"""
Image quality metrics.

Single source of truth for all metric computation in this project.
Every evaluation script imports from here — no inline metric code anywhere.

Metrics implemented:
    PSNR  — Peak Signal-to-Noise Ratio (ISO standard, primary metric)
    SSIM  — Structural Similarity Index (Wang & Bovik 2004, primary metric)
    KernelError — MSE between estimated and true blur parameters

NOT implemented (intentionally):
    Gap closure % — mathematically broken (produces >100% when recon > oracle PSNR)
    LPIPS — requires additional network; add in v2 if needed
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PSNRResult:
    value: float          # dB
    mse: float            # raw MSE for debugging
    is_valid: bool        # False if images have different shapes or are all-black


@dataclass
class SSIMResult:
    value: float          # [0, 1]
    is_valid: bool


@dataclass
class ImageMetrics:
    """All metrics for a single image pair."""
    psnr: PSNRResult
    ssim: SSIMResult
    input_psnr: Optional[PSNRResult] = None     # PSNR of degraded input (for improvement calc)

    @property
    def psnr_improvement(self) -> Optional[float]:
        """PSNR improvement over degraded input (dB). None if input_psnr not set."""
        if self.input_psnr is None:
            return None
        return self.psnr.value - self.input_psnr.value


@dataclass
class KernelMetrics:
    """Metrics for kernel estimation accuracy."""
    sigma_x_error: float    # |σ_x_estimated - σ_x_true|
    sigma_y_error: float    # |σ_y_estimated - σ_y_true|
    sigma_x_estimated: float
    sigma_y_estimated: float
    sigma_x_true: float
    sigma_y_true: float
    converged: bool


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    reduction: str = "mean",
) -> PSNRResult:
    """
    Compute Peak Signal-to-Noise Ratio.

    Formula: PSNR = 10 * log10(max_val² / MSE)

    Args:
        pred: Predicted/reconstructed image (B, C, H, W) or (C, H, W)
        target: Ground truth image, same shape as pred
        max_val: Maximum possible pixel value (1.0 for [0,1], 255.0 for uint8)
        reduction: 'mean' averages over batch; 'none' returns per-image values

    Returns:
        PSNRResult with value in dB

    Notes:
        Higher is better. Common reference points:
        - 30 dB: acceptable quality
        - 35 dB: good quality
        - 40+ dB: very high quality (near lossless)
    """
    if pred.shape != target.shape:
        return PSNRResult(value=float("nan"), mse=float("nan"), is_valid=False)

    # Ensure float computation
    pred = pred.float()
    target = target.float()

    mse = F.mse_loss(pred, target, reduction="mean").item()

    if mse == 0.0:
        # Perfect reconstruction — return a large finite value
        return PSNRResult(value=100.0, mse=0.0, is_valid=True)

    import math
    psnr_val = 10.0 * math.log10(max_val**2 / mse)

    return PSNRResult(value=psnr_val, mse=mse, is_valid=True)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> SSIMResult:
    """
    Compute Structural Similarity Index (Wang & Bovik, 2004).

    Uses a Gaussian-weighted local window to compare luminance, contrast,
    and structure between pred and target.

    Args:
        pred: Predicted image (B, C, H, W) or (C, H, W), values in [0, max_val]
        target: Ground truth image, same shape
        window_size: Size of Gaussian window (default 11, per Wang et al.)
        sigma: Gaussian std for window (default 1.5, per Wang et al.)
        max_val: Dynamic range of images

    Returns:
        SSIMResult with value in [0, 1]. 1.0 = identical images.
    """
    if pred.shape != target.shape:
        return SSIMResult(value=float("nan"), is_valid=False)

    # Add batch dim if needed
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    pred = pred.float()
    target = target.float()

    # Convert to grayscale for SSIM (standard practice)
    if pred.shape[1] == 3:
        pred_gray = rgb_to_grayscale(pred)
        target_gray = rgb_to_grayscale(target)
    else:
        pred_gray = pred
        target_gray = target

    # Build Gaussian window
    window = _gaussian_window(window_size, sigma, device=pred.device)

    # SSIM constants (stabilise division when denominator near 0)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = F.conv2d(pred_gray, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(target_gray, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_gray ** 2, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target_gray ** 2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(pred_gray * target_gray, window, padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return SSIMResult(value=ssim_map.mean().item(), is_valid=True)


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Build a normalised 1D Gaussian and outer-product to 2D."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


# ---------------------------------------------------------------------------
# Kernel estimation accuracy
# ---------------------------------------------------------------------------


def compute_kernel_metrics(
    sigma_x_estimated: float,
    sigma_y_estimated: float,
    sigma_x_true: float,
    sigma_y_true: float,
    converged: bool = True,
) -> KernelMetrics:
    """
    Measure how accurately blind kernel estimation recovered the true parameters.

    Args:
        sigma_x_estimated: Estimated blur std in x from estimate_params()
        sigma_y_estimated: Estimated blur std in y
        sigma_x_true: Ground truth σ_x used to generate degraded image
        sigma_y_true: Ground truth σ_y

    Returns:
        KernelMetrics with absolute errors and raw values
    """
    return KernelMetrics(
        sigma_x_error=abs(sigma_x_estimated - sigma_x_true),
        sigma_y_error=abs(sigma_y_estimated - sigma_y_true),
        sigma_x_estimated=sigma_x_estimated,
        sigma_y_estimated=sigma_y_estimated,
        sigma_x_true=sigma_x_true,
        sigma_y_true=sigma_y_true,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------


def evaluate_image_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    degraded: Optional[torch.Tensor] = None,
) -> ImageMetrics:
    """
    Compute all image quality metrics for a reconstruction/target pair.

    Args:
        pred: Reconstructed image (B, C, H, W) or (C, H, W)
        target: Ground truth clean image
        degraded: Original degraded input (optional; enables improvement calculation)

    Returns:
        ImageMetrics with PSNR, SSIM, and optional improvement over input
    """
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)

    input_psnr = None
    if degraded is not None:
        input_psnr = compute_psnr(degraded, target)

    return ImageMetrics(psnr=psnr, ssim=ssim, input_psnr=input_psnr)
