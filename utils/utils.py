# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict

import torch
import transformers

from diffusers import DiffusionPipeline
from pathlib import Path
from tqdm import tqdm
from typing import List


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name):
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {}
        for key in state_dict.keys():
            if "teacher" in key:
                continue
            cpu_state_dict[key] = state_dict[key]
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_local_rank():
    if os.environ["LOCAL_RANK"]:
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


###############################################################################
# Image generation utilities
###############################################################################

def generate_images(
    pipe: DiffusionPipeline,
    prompts: List[str],
    num_images: int,
    out_dir: Path,
    device: str,
    seed: int | None = None,
) -> None:
    """Generate `num_images` images and save as PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    for i in tqdm(range(num_images), desc="Generating"):
        prompt = prompts[i % len(prompts)]
        with torch.amp.autocast(enabled=pipe.torch_dtype == torch.float16):
            image = pipe(prompt).images[0]
        image.save(out_dir / f"sample_{i:03d}.png")