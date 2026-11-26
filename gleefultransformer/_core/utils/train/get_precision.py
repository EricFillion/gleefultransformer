import torch


def get_precision(device) -> str:
    if device.type != "cuda":
        return "fp32"
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return "bf16"
    return "fp16"
