# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from models.modeling_flux_quant import (
    FluxTransformer2DModel as FluxTransformer2DModelQuant,
)
from diffusers import FluxPipeline, DiffusionPipeline
import copy
import torch
import transformers
from utils import utils
from utils import datautils
from utils import trainer

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer

from utils.prompt_list import get_default_prompts, get_prompt

from pathlib import Path
from tqdm import tqdm
import gc
from torch.utils.data import Subset

log = utils.get_logger("clm")


def load_quantized_model(model_args, training_args, cache_dir: Path, w_bits=16):
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    model = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
        w_bits=w_bits
    )

    if not model_args.contain_weight_clip_val:
        weight_clip_val_dict = {}
        for name, param in tqdm(model.named_parameters(), desc="Initializing weight_clip_val"):
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)

                if w_bits == 1:
                    scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
                elif w_bits == 0 or w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif 3 <= w_bits and w_bits <= 8:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (w_bits - 1) - 1
                    scale = xmax / maxq
                else:
                    raise NotImplementedError
                    
                weight_clip_val_dict[name] = scale
        model.load_state_dict(weight_clip_val_dict, assign=True, strict=False)
    
    return model

def train(debug=False):
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()
    
    
    ### Dataset Generation
    ## Configuration
    dataset_dir = Path(training_args.output_dir) / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    ## Load Prompt List
    prompt_list = get_prompt(3000) if not debug else get_default_prompts()

    ## Dataset Generation
    if not (dataset_dir / f'{len(prompt_list)-1}_{prompt_list[-1].replace(" ", "_")}').exists():
        ## Load Model
        dtype = torch.bfloat16 if training_args.bf16 else torch.float
        pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_args.input_model_filename, torch_dtype=dtype).to('cuda')
        org_model = FluxTransformer2DModelQuant.from_pretrained(
                pretrained_model_name_or_path=model_args.input_model_filename,
                subfolder="transformer",
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                device_map=None,
                w_bits=16,
                dataset_collect=True,
            ).to('cuda')
        pipe.transformer = org_model

        ## Dataset Collection
        for idx, prompt in tqdm(enumerate(prompt_list)): 
            pipe([prompt])
            torch.save(pipe.transformer.dataset_dict, dataset_dir / f'{idx}_{prompt.replace(" ", "_")}')
            pipe.transformer.clear_dataset()
            torch.cuda.empty_cache()
        
        log.info("Clear Cache Memory")
        del pipe
        del org_model
        torch.cuda.empty_cache()
    
    ### Quantization Aware Training
    ## Load Model
    log.info(f"load_model, w_bits: {model_args.w_bits}")
    cache_dir = Path(training_args.output_dir) / "cache" / f"bits_{model_args.w_bits}" 
    q_model = load_quantized_model(model_args, training_args, cache_dir, w_bits=model_args.w_bits).to('cuda')
    if training_args.gradient_checkpointing:
        log.info("Gradient Checkpointing Enable")
        training_args.gradient_checkpointing = False
        q_model.enable_gradient_checkpointing()

    ## Sanity Check
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_args.input_model_filename, torch_dtype=torch.bfloat16).to('cuda')
    pipe.transformer = q_model
    utils.generate_images(pipe, 
                        ["Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render"], 
                        1, 
                        Path(training_args.output_dir) / "eval" / f"sanity",
                        'cuda',
                        seed=42)

    del pipe
    log.info("Complete model loading...")

    ## Load Dataset
    dataset = trainer.TorchFileDataset(dataset_dir, debug=debug, group_size=training_args.train_batch_size)
    train_data = dataset
    valid_data = Subset(dataset, list(range(28 * 4)))

    log.info(f"train dataset size: {len(train_data)}")

    ## Apply LoRA
    ## Freeze non-LoRA params
    total_params = list(map(lambda x: x[0], q_model.named_parameters()))
    for name, param in q_model.named_parameters():
        if name.split(".")[-1] == "weight":
            # For LoRA Weight... unable real_weight update
            tmp = name.split(".")
            tmp[-1] = "lora_A"

            if ".".join(tmp) in total_params:
                param.requires_grad = False
        

    ## Load Trainer
    mytrainer = trainer.FluxQATTrainer(
        model = q_model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=trainer.custom_collate_fn,
        callbacks=[trainer.EmptyCacheCallback(save_path=training_args.output_dir)],
    )
    
    # Do Train
    if training_args.do_train:
        gc.collect()
        torch.cuda.empty_cache()
        train_result = mytrainer.train(
            resume_from_checkpoint=True
        )
        mytrainer.save_state()
        utils.safe_save_model_for_hf_trainer(mytrainer, model_args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        utils.generate_images(pipe, ["A lone violinist playing on top of a sinking airship during golden hour, with sheet music flying into the wind and glowing birds circling above"], 
                            1, out_dir=training_args.output_dir / "eval" / f"last")

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
