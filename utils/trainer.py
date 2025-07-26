from transformers import Trainer, TrainerCallback
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms
import torch.nn.functional as F

from utils import utils
from enum import IntEnum

from diffusers import FluxPipeline, DiffusionPipeline
from pathlib import Path

from logging import getLogger

# import ImageReward as RM
# reward_model = RM.load("ImageReward-v1.0")

logger = getLogger()

class InputIndex(IntEnum):
    HIDDEN_STATES = 0               # torch.Size([B, 4096, 64])
    ENCODER_HIDDEN_STATES = 1       # torch.Size([B, 512, 4096])
    POOLED_PROJECTIONS = 2          # torch.Size([B, 768])
    TIMESTEP = 3                    # torch.Size([B])
    IMG_IDS = 4                     # torch.Size([4096, 3])
    TXT_IDS = 5                     # torch.Size([512, 3])
    GUIDANCE = 6                    # torch.Size([B])
    JOINT_ATTENTION_KWARGS = 7      # empty dict: {}
    CONTROLNET_BLOCK_SAMPLES = 8    # None
    CONTROLNET_SINGLE_BLOCK_SAMPLES = 9  # None
    RETURN_DICT = 10               # bool
    CONTROLNET_BLOCKS_REPEAT = 11  # bool

def compute_psnr(pred, target, max_pixel_value=1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    psnr = 20 * torch.log10(max_pixel_value) - 10 * torch.log10(mse)
    return psnr.item()

def custom_collate_fn(batch):
    input_len = len(batch[0]["input"])
    input_elements = [[] for _ in range(input_len)]
    outputs = []

    for item in batch:
        for i in range(input_len):
            input_elements[i].append(item["input"][i])
        outputs.append(item["output"])

    collated_input = []

    for i, elems in enumerate(input_elements):
        example = elems[0]

        idx = InputIndex(i)

        if idx in [InputIndex.HIDDEN_STATES, InputIndex.ENCODER_HIDDEN_STATES, 
                    InputIndex.POOLED_PROJECTIONS, InputIndex.TIMESTEP, 
                    InputIndex.GUIDANCE]:
            collated_input.append(torch.concat(elems))  # e.g. [B, ...]
        elif idx in [InputIndex.IMG_IDS, InputIndex.TXT_IDS]:
            collated_input.append(elems[0])
        elif idx in [InputIndex.JOINT_ATTENTION_KWARGS]:
            # empty dict case: just return empty dict
            if all(isinstance(e, dict) and len(e) == 0 for e in elems):
                collated_input.append({})
            else:
                # if dicts have keys (not empty), stack per key
                collated_input.append({
                    key: torch.stack([e[key] for e in elems])
                    for key in example.keys()
                })
        elif idx in [InputIndex.CONTROLNET_BLOCK_SAMPLES, InputIndex.CONTROLNET_SINGLE_BLOCK_SAMPLES]:
            collated_input.append(None)
        elif idx in [InputIndex.RETURN_DICT, InputIndex.CONTROLNET_BLOCKS_REPEAT]:
            if not all(isinstance(e, bool) for e in elems):
                raise TypeError(f"Expected bool at index {i}, got: {[type(e) for e in elems]}")
            collated_input.append(elems[0])  # single bool (same across batch)
        else:
            raise TypeError(f"Unsupported type or unexpected index: {i} ({type(example)})")

    collated_output = torch.concat(outputs)

    return {
        "input": tuple(collated_input),
        "output": collated_output
    }

class TorchFileDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, debug=False, group_size=4): # batch size
        self.file_paths = sorted([
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
        ])

        logger.info(f"debug mode: {debug}")
        if debug: self.file_paths = self.file_paths[:10]

        self.total_data = [torch.load(path) for path in self.file_paths]
        self.group_size = group_size
        self.seq_len = len(self.total_data[0]["input"])   # 예: 28
        self.num_groups = (len(self.total_data) + group_size - 1) // group_size

    def __len__(self):
        return len(self.total_data) * 28

    def __getitem__(self, idx):
        """
        idx 가 0‥N*28-1 을 돌 때[]
        ├─ group_idx      : 몇 번째 4-개 묶음인지
        ├─ inner_idx      : 시퀀스 안쪽 번호 (0‥27)
        └─ pos_in_group   : 4-개 묶음 안에서의 위치 (0‥3)
        """
        
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))  # slice 객체 → 인덱스 리스트
            return [self[i] for i in indices]  # 재귀적으로 self[i] 호출
        
        # ① 시퀀스 안쪽 번호 (0‥27) — 4칸씩 모은 뒤 28에서 한 바퀴
        inner_idx = (idx // self.group_size) % self.seq_len

        # ② 몇 번째 4-개 묶음인지
        group_idx = idx // (self.group_size * self.seq_len)

        # ③ 4-개 묶음 안에서의 위치
        pos_in_group = idx % self.group_size

        # ④ 실제 sample 인덱스 = 묶음 시작 위치 + pos_in_group
        sample_idx = group_idx * self.group_size + pos_in_group

        # (마지막 묶음이 4개보다 적을 수 있으니 범위 체크)
        if sample_idx >= len(self.total_data):
            raise IndexError("index exceeds available samples")

        sample = self.total_data[sample_idx]
        return {
            "input":  sample["input"][inner_idx],      # tuple of tensors
            "output": sample["output"][inner_idx][0],  # tensor
        }        
class FluxQATTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_prompt = "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render"

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(*inputs["input"])
            pred = pred[0] # unveil tuple 
            
            label = inputs["output"]

            loss = F.mse_loss(pred, label)

            return (loss, pred) if return_outputs else loss

        
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

    def _move_to_device(self, input_tuple, device):
        moved = []
        for x in input_tuple:
            if isinstance(x, torch.Tensor):
                moved.append(x.to(device))
            elif isinstance(x, dict):
                moved.append({k: v.to(device) for k, v in x.items()})
            else:
                moved.append(x)
        return tuple(moved)

class EmptyCacheCallback(TrainerCallback):
    def __init__(self, save_path="./output/"):
        self.best_reward = float("-inf")
        self.last_generated_dir = None 
        self.save_path = save_path

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # 50스텝마다
            torch.cuda.empty_cache()
            pipe : FluxPipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
            model = kwargs['model'].eval()
            pipe.transformer = model.to(torch.bfloat16)
    
            utils.generate_images(pipe, 
                                ["Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
                                "A cat made of sea water walking in a library.", "A television on the top of a bird"], 
                                3, 
                                Path(self.save_path) / "eval" / f"step_{state.global_step}",
                                'cuda',
                                seed=42)
            
            del pipe
            torch.cuda.empty_cache()
            
            out_dir = Path(self.save_path) / "eval" / f"step_{state.global_step}"
            self.last_generated_dir = out_dir  # 여기 저장!


if __name__ == "__main__":
    dataset = TorchFileDataset(folder_path="./output/dataset")
    breakpoint()