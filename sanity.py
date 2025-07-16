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


log = utils.get_logger("clm")

def load_quantized_model(model_args, training_args, w_bits=16):
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    model = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        w_bits=w_bits
    )

    if not model_args.contain_weight_clip_val:
        for name, param in model.named_parameters():
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)

                if model_args.w_bits == 1:
                    scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
                elif model_args.w_bits == 0 or model_args.w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif model_args.w_bits == 3 or model_args.w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (model_args.w_bits - 1) - 1
                    scale = xmax / maxq
                else:
                    raise NotImplementedError

                param.data.copy_(scale)
    return model


def sanity(debug=False):
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    prompts = get_default_prompts()

    # Sanity Check Full Precision
    dtype = torch.bfloat16 if training_args.bf16 else torch.float
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_args.input_model_filename, torch_dtype=dtype).to('cuda')

    samples_dir = Path(training_args.output_dir) / "samples" / "bf16"
    print(f"Generating 2 sample images …")
    utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")

    # Sanity Check bf 16
    log.info("Start to load model...")
    model = load_quantized_model(model_args, training_args, w_bits=16)
    model.cuda()
    log.info("Complete model loading...")

    pipe.transformer = model
    
    samples_dir = Path(training_args.output_dir) / "samples" / "bits_16"
    print(f"Generating 2 sample images …")
    utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")

    # Sanity Check int 8
    log.info("Start to load model...")
    model = load_quantized_model(model_args, training_args, w_bits=8)
    model.cuda()
    log.info("Complete model loading...")

    pipe.transformer = model
    
    samples_dir = Path(training_args.output_dir) / "samples" / "bits_8"
    print(f"Generating 2 sample images …")
    utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")




if __name__ == "__main__":
    sanity(debug=True)
