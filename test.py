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
from safetensors.torch import load_file

log = utils.get_logger("clm")

# def load_quantized_model(checkpoint_path: Path, w_bits=16):
#     dtype = torch.bfloat16

#     log.info(f"Loading quantized model from cache directory: {checkpoint_path} | w_bits = {w_bits}")
#     model = FluxTransformer2DModelQuant.from_pretrained(
#         pretrained_model_name_or_path=checkpoint_path,
#         torch_dtype=dtype,
#         low_cpu_mem_usage=True,
#         device_map=None,
#         w_bits=w_bits
#     )
#     return model


def sanity(checkpoint_path, w_bits):
    output_path = "./test/"

    prompts = ["A cat made of sea water walking in a library.", "A television on the top of a bird"]

    # Sanity Check Full Precision
    dtype = torch.bfloat16
    pipe: FluxPipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype).to('cuda')

    samples_dir = Path(output_path) / "samples" / "bf16"
    print(f"Generating 2 sample images …")
    utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")


    # load model
    log.info(f"Start to load model... w_bits: {w_bits}")
    pipe.transformer = FluxTransformer2DModelQuant.from_pretrained(
        pretrained_model_name_or_path=f"./output/cache/bits_{w_bits}",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
        w_bits=w_bits
    )
    log.info(f"Start to load weight... w_bits: {w_bits}")
    pipe.transformer.load_state_dict(load_file(Path(checkpoint_path) / "model.safetensors"), strict=False)
    pipe = pipe.to('cuda')
    log.info("Complete model loading...")
    
    # inference model
    samples_dir = Path(output_path) / "samples" / f"bits_{w_bits}"
    print(f"Generating 2 sample images …")
    utils.generate_images(pipe, prompts, 2, samples_dir, 'cuda', seed=42)
    print(f"Samples saved to '{samples_dir}'")



if __name__ == "__main__":
    import sys
    checkpoint_path = sys.argv[1]
    w_bits = sys.argv[2]
    sanity(checkpoint_path, int(w_bits))
