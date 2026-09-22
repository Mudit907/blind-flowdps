"""
Unit tests for image quality metrics.

The most important test here is that gap closure % is NOT present.
That metric is broken (produces >100% when recon > oracle PSNR)
and must never appear in this codebase.
"""

import math

import pytest
import torch

from src.metrics.image_metrics import (
    compute_psnr,
    compute_ssim,
    evaluate_image_pair,
)


@pytest.fixture
def identical_images():
    torch.manual_seed(0)
    img = torch.rand(1, 3, 64, 64)
    return img, img.clone()


@pytest.fixture
def noisy_pair():
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 64, 64)
    noisy = (clean + torch.randn_like(clean) * 0.1).clamp(0, 1)
    return clean, noisy


class TestPSNR:
    def test_identical_images_returns_high_value(self, identical_images):
        pred, target = identical_images
        result = compute_psnr(pred, target)
        assert result.value == 100.0  # Perfect reconstruction
        assert result.is_valid

    def test_noisy_images_in_range(self, noisy_pair):
        clean, noisy = noisy_pair
        result = compute_psnr(noisy, clean)
        # Realistic PSNR range for moderate noise
        assert 10.0 < result.value < 50.0
        assert result.is_valid

    def test_higher_noise_lower_psnr(self):
        torch.manual_seed(0)
        clean = torch.rand(1, 3, 64, 64)
        mildly_noisy = (clean + torch.randn_like(clean) * 0.05).clamp(0, 1)
        heavily_noisy = (clean + torch.randn_like(clean) * 0.3).clamp(0, 1)

        psnr_mild = compute_psnr(mildly_noisy, clean).value
        psnr_heavy = compute_psnr(heavily_noisy, clean).value
        assert psnr_mild > psnr_heavy

    def test_shape_mismatch_returns_invalid(self):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 128, 128)
        result = compute_psnr(pred, target)
        assert not result.is_valid

    def test_formula_correctness(self):
        """Verify PSNR = 10*log10(1/MSE) for max_val=1."""
        torch.manual_seed(42)
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        mse = ((pred - target) ** 2).mean().item()
        expected_psnr = 10.0 * math.log10(1.0 / mse)

        result = compute_psnr(pred, target)
        assert abs(result.value - expected_psnr) < 0.01


class TestSSIM:
    def test_identical_images_returns_one(self, identical_images):
        pred, target = identical_images
        result = compute_ssim(pred, target)
        # SSIM of identical images should be very close to 1.0
        assert result.value > 0.99
        assert result.is_valid

    def test_different_images_below_one(self, noisy_pair):
        clean, noisy = noisy_pair
        result = compute_ssim(noisy, clean)
        assert 0.0 < result.value < 1.0
        assert result.is_valid

    def test_shape_mismatch_returns_invalid(self):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 128, 128)
        result = compute_ssim(pred, target)
        assert not result.is_valid


class TestEvaluateImagePair:
    def test_computes_improvement(self, noisy_pair):
        clean, noisy = noisy_pair
        # Pretend 'noisy' is the reconstruction (same as input here for testing)
        metrics = evaluate_image_pair(pred=noisy, target=clean, degraded=noisy)
        assert metrics.psnr_improvement is not None
        assert metrics.psnr_improvement == 0.0  # pred == degraded, so no improvement

    def test_no_degraded_means_no_improvement(self, noisy_pair):
        clean, noisy = noisy_pair
        metrics = evaluate_image_pair(pred=noisy, target=clean)
        assert metrics.psnr_improvement is None  # No degraded provided

    def test_gap_closure_not_implemented(self):
        """
        Gap closure % is mathematically broken and must NOT be in this codebase.
        This test will fail if anyone adds it — intentional safety guard.
        """
        from src.metrics import image_metrics
        assert not hasattr(image_metrics, "compute_gap_closure"), (
            "Gap closure metric must not be implemented — it produces >100% "
            "when reconstruction PSNR exceeds oracle PSNR. Use psnr_improvement instead."
        )
