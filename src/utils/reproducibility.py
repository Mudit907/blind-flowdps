"""
Reproducibility utilities.

Every experiment sets a seed via seed_everything() at startup.
This ensures runs are deterministic and results are comparable across machines.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers Python, NumPy, PyTorch (CPU and CUDA), and cuDNN.
    Call this at the top of every script before any model or data loading.

    Args:
        seed: Integer seed value. Default 42 is used throughout this project.
              Change per-experiment via the YAML config, not here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism — slight performance cost, necessary for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
