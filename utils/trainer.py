from transformers import Trainer
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
import torch.nn.functional as F

from utils import utils

def compute_psnr(pred, target, max_pixel_value=1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    psnr = 20 * torch.log10(max_pixel_value) - 10 * torch.log10(mse)
    return psnr.item()
    
def custom_collate_fn(batch):
    input_len = len(batch[0]["input"])  # 보통 12
    input_elements = [[] for _ in range(input_len)]
    outputs = []

    for item in batch:
        for i in range(input_len):
            input_elements[i].append(item["input"][i])
        outputs.append(item["output"])

    collated_input = []

    for i, elems in enumerate(input_elements):
        example = elems[0]

        if isinstance(example, torch.Tensor):
            collated_input.append(torch.stack(elems))  # (B, ...)
        elif isinstance(example, dict):
            # dict: key마다 collate
            collated_dict = {
                key: torch.stack([e[key] for e in elems])
                for key in example.keys()
            }
            collated_input.append(collated_dict)
        elif example is None:
            collated_input.append([None for _ in elems])  # 유지
        elif isinstance(example, bool):
            collated_input.append(torch.tensor(elems))  # bool → tensor
        else:
            raise TypeError(f"Unsupported type at input[{i}]: {type(example)}")

    collated_output = torch.stack(outputs)

    return {
        "input": tuple(collated_input),
        "output": collated_output
    }

class TorchFileDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.file_paths = sorted([
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
        ])

        self.total_data = [torch.load(path) for path in self.file_paths]

    def __len__(self):
        return len(self.total_data) * 28

    def __getitem__(self, idx):
        sample = self.total_data[idx//28]
        return {
            "input": sample["input"][idx%28], # type: tuple(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict, None, None, bool, bool)
            "output": sample["output"][idx%28][0], # type: torch.Tensor
        }
        
class FluxQATTrainer(Trainer):
    def __init__(self, *args, flux_pipeline=None, org_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = flux_pipeline  # 사용자가 전달해야 함
        self.org_model = org_model
        self.test_prompt = "A lone violinist playing on top of a sinking airship during golden hour, with sheet music flying into the wind and glowing birds circling above"

    def compute_loss(self, model, inputs, return_outputs=False):
        pred = model(*inputs["input"])
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