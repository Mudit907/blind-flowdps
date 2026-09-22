"""
Image I/O utilities.

Centralised loading and saving so every script handles images consistently.
All tensors in this project are float32 in [0, 1], shape (B, C, H, W) or (C, H, W).
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image


def load_image(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load an image from disk as a float32 tensor in [0, 1].

    Args:
        path: Path to image file (PNG, JPG, etc.)
        size: Optional (H, W) to resize. If None, keeps original size.
        device: Target device for the tensor.

    Returns:
        Float32 tensor of shape (1, C, H, W), values in [0, 1].
        Batch dim is always added for downstream compatibility.
    """
    img = Image.open(path).convert("RGB")

    if size is not None:
        img = img.resize((size[1], size[0]), Image.LANCZOS)  # PIL takes (W, H)

    tensor = TF.to_tensor(img)  # (C, H, W), float32, [0, 1]
    return tensor.unsqueeze(0).to(device)  # (1, C, H, W)


def save_image(
    tensor: torch.Tensor,
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save a tensor as an image file.

    Args:
        tensor: Float32 tensor of shape (B, C, H, W) or (C, H, W), values in [0, 1].
                If batch size > 1, saves only the first image.
        path: Output path. Extension determines format (PNG recommended).
        quality: JPEG quality if saving as JPEG (ignored for PNG).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle batch dim
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Clamp and convert to uint8
    tensor = tensor.detach().cpu().clamp(0, 1)
    img = TF.to_pil_image(tensor)
    img.save(path, quality=quality)


def tensor_to_numpy(tensor: torch.Tensor) -> "np.ndarray":
    """Convert (B,C,H,W) or (C,H,W) tensor to (H,W,C) uint8 numpy array."""
    import numpy as np

    if tensor.dim() == 4:
        tensor = tensor[0]

    return (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
