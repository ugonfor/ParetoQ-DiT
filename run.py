# quantize_flux.py
"""
Reproduce **1.58‑bit FLUX** quantisation *with* Diffusers integration.

This standalone script now:

1. **Loads** any Diffusers checkpoint (default: `black-forest-labs/FLUX.1-dev`).
2. **Ternary‑quantises** **all** `nn.Linear` layers inside the UNet **and** the
   text‑encoder to {-1, 0, +1} following the magnitude threshold rule described
   in *1.58‑bit FLUX* (Yang et al., 2024). No calibration data required.
3. **Saves** the compact pipeline (`pipe.save_pretrained`).
4. **Optionally** runs a smoke‑test that generates a user‑defined number of
   images from a diverse prompt list (100 prompts by default).

---
**Quickstart**
```bash
pip install torch>=2.2 diffusers transformers accelerate safetensors pillow tqdm

python quantize_flux.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --output_path ./flux_1p58bit \
  --threshold_ratio 0.05 \
  --num_images 10             # set 0 to skip generation
```
The script writes quantised weights to `./flux_1p58bit/` and, if generation is
enabled, PNG samples to `./flux_1p58bit/samples/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm

from prompt_list import get_default_prompts  # Custom module to load default prompts

###############################################################################
# 1‑bit‑plus‑sign (≈1.58‑bit) Linear‑layer quantisation helpers
###############################################################################

def _quantize_linear(layer: torch.nn.Linear, threshold_ratio: float = 0.05) -> None:
    """In‑place ternary quantisation of a single `nn.Linear` layer.

    Args:
        layer: The Linear layer to quantise.
        threshold_ratio: |w| values **below** `ratio * max(|w|)` become 0.
    """
    with torch.no_grad():
        W = layer.weight.data
        t = threshold_ratio * W.abs().max()
        q = torch.where(W.abs() < t, torch.zeros_like(W), torch.sign(W))

        # Store compact int8 tensor + scale; drop original weight param
        layer.register_buffer("qweight", q.to(torch.int8))
        layer.register_buffer("scale", W.abs().max())
        delattr(layer, "weight")

        def _forward(x):  # type: ignore
            w = layer.qweight.float() * layer.scale
            return torch.nn.functional.linear(x, w, layer.bias)

        layer.forward = _forward  # type‑patch to bypass weight attr


def quantize_model(model: torch.nn.Module, threshold_ratio: float = 0.05) -> None:
    """Recursively apply ternary quantisation to **all** Linear layers."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            _quantize_linear(mod, threshold_ratio)


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
        with torch.cuda.amp.autocast(enabled=pipe.torch_dtype == torch.float16):
            image = pipe(prompt).images[0]
        image.save(out_dir / f"sample_{i:03d}.png")

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Ternary‑quantise a Diffusers FLUX model")
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev", help="Hugging Face model ID or local path")
    p.add_argument("--output_path", type=str, required=True, help="Where to save the quantised pipeline")
    p.add_argument("--threshold_ratio", type=float, default=0.05, help="|w| below ratio*max -> 0")
    p.add_argument("--num_images", type=int, default=0, help="Number of test images to generate (0 = skip)")
    p.add_argument("--prompts_json", type=str, default=None, help="Optional JSON file containing a list of prompts")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Computation device")
    p.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    output_path = Path(args.output_path)

    print(f"Loading pipeline '{args.model_id}' …")
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    breakpoint()
    print("Quantising text‑encoder …")
    quantize_model(pipe.text_encoder, args.threshold_ratio)

    print("Quantising UNet …")
    quantize_model(pipe.unet, args.threshold_ratio)

    print(f"Saving quantised checkpoint to '{output_path}' …")
    pipe.save_pretrained(output_path)

    # Optional image generation ------------------------------------------------
    if args.num_images > 0:
        if args.prompts_json is not None:
            with open(args.prompts_json, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            assert isinstance(prompts, list) and all(isinstance(s, str) for s in prompts), "JSON file must be a list of strings"
        else:
            prompts = get_default_prompts()

        samples_dir = output_path / "samples"
        print(f"Generating {args.num_images} sample images …")
        generate_images(pipe, prompts, args.num_images, samples_dir, device, seed=args.seed)
        print(f"Samples saved to '{samples_dir}'")

    print("Done ✔️")


if __name__ == "__main__":
    main()
