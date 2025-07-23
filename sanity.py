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

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer

from utils.prompt_list import get_default_prompts

from pathlib import Path
from tqdm import tqdm

import torch.nn as nn

log = utils.get_logger("clm")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not 'transformer' in name:
                continue
            # if 'norm' in name:
            #     continue
            linear_params += sum(p.numel() for p in module.parameters())

    ratio = linear_params / total_params if total_params > 0 else 0

    print(f"Total parameters: {total_params}")
    print(f"Linear layer parameters: {linear_params}")
    print(f"Linear layer parameter ratio: {ratio:.4f}")

    return {
        "total": total_params,
        "linear": linear_params,
        "ratio": ratio
    }

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


def sanity(debug=False):
    # dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()
    if debug: print(model_args, data_args, training_args)

    prompts = get_default_prompts()

    # Sanity Check Full Precision
    dtype = torch.bfloat16 if training_args.bf16 else torch.float
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_args.input_model_filename, torch_dtype=dtype).to('cuda')

    count_parameters(pipe.transformer)

    samples_dir = Path(training_args.output_dir) / "samples" / "bf16"
    print(f"Generating 2 sample images …")
    utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")

    for w_bits in [16, 8, 4, 2, 0]:
        # load model
        log.info(f"Start to load model... w_bits: {w_bits}")
        cache_dir = Path(training_args.output_dir) / "cache" / f"bits_{w_bits}"
        model = load_quantized_model(model_args, training_args, cache_dir, w_bits=w_bits)
        model.cuda()
        pipe.transformer = model
        log.info("Complete model loading...")
        
        # inference model
        samples_dir = Path(training_args.output_dir) / "samples" / f"bits_{w_bits}"
        print(f"Generating 2 sample images …")
        utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
        print(f"Samples saved to '{samples_dir}'")

        # save and remove model
        model.save_pretrained(cache_dir)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sanity(debug=True)
