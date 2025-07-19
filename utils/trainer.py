from transformers import Trainer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms
import torch.nn.functional as F

from utils import utils
from enum import IntEnum

from logging import getLogger
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

class MyCollator:
    def __call__(self, batch):
        return custom_collate_fn(batch)

class TorchFileDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, debug=False):
        self.file_paths = sorted([
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
        ])

        logger.info(f"debug mode: {debug}")
        if debug: self.file_paths = self.file_paths[:10]

        self.total_data = [torch.load(path) for path in self.file_paths]

    def __len__(self):
        return len(self.total_data) * 28

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))  # slice 객체 → 인덱스 리스트
            return [self[i] for i in indices]  # 재귀적으로 self[i] 호출
            
        sample = self.total_data[idx//28]
        return {
            "input": sample["input"][idx%28], # type: tuple(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict, None, None, bool, bool)
            "output": sample["output"][idx%28][0], # type: torch.Tensor
        }
        
class FluxQATTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_prompt = "A lone violinist playing on top of a sinking airship during golden hour, with sheet music flying into the wind and glowing birds circling above"

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(*inputs["input"])
            pred = pred[0] # unveil tuple 
            
            label = inputs["output"]

            loss = F.mse_loss(pred, label)

            return (loss, pred) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self.model
        model.eval()
        device = model.device

        psnr_list = []

        self.pipe.transformer = self.org_model
        self.pipe = self.pipe.to(device)
        gt = self.pipe(self.test_prompt)

        self.pipe.transformer = model
        self.pipe = self.pipe.to(device)
        pred = self.pipe(self.test_prompt)

        utils.generate_images(pipe, [self.test_prompt], 
                            1, out_dir=training_args.output_dir / "eval" / f"step_{self.state.global_step}")
    
        psnr = compute_psnr(pred, gt)
        psnr_list.append(psnr)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        print(f"[Evaluate] PSNR from FluxPipeline: {avg_psnr:.2f} dB")

        return {f"{metric_key_prefix}_psnr": avg_psnr}
        
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
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

if __name__ == "__main__":
    dataset = TorchFileDataset(folder_path="./output/dataset")
    breakpoint()