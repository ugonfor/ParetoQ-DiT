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


log = utils.get_logger("clm")

def sanity(debug=False):
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    model = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        w_bits=16
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

    model.cuda()
    log.info("Complete model loading...")

    pipe: FluxPipeline = DiffusionPipeline.from_pretrained(model_args.input_model_filename, torch_dtype=dtype)


    prompts = get_default_prompts()

    samples_dir = "." / "samples"
    print(f"Generating 4 sample images …")
    utils.generate_images(pipe, prompts, 4, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")



if __name__ == "__main__":
    sanity(debug=True)
