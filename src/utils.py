"""Reproducibility helpers. Every run must record seed, device and library versions."""
import json
import os
import platform
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def environment_info() -> dict:
    """Hardware and versions, to paste into the experiment log."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    set_seed()
    print(json.dumps(environment_info(), indent=2))
