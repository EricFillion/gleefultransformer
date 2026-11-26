import os

import torch
from torch import device

from gleefultransformer._core.exceptions import GleefulDeviceError


def get_device() -> device:
    try:
        d = None
        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                d = torch.device("mps")

        if torch.cuda.is_available():
            d = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

        if not d:
            d = torch.device("cpu")

        return d
    except Exception as e:
        raise GleefulDeviceError(f"Device selection failed: {e}")
