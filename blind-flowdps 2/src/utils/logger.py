"""
Experiment logger backed by Weights & Biases.

Every evaluation run logs to W&B automatically. This gives you:
  - Sigma trajectory plots during diffusion
  - PSNR/SSIM per image
  - Kernel estimation convergence curves
  - Side-by-side image comparisons (input, recon, ground truth)
  - Full hyperparameter tracking (every run is reproducible by config)

If W&B is unavailable (no API key), falls back to local JSON logging.
This means the evaluation pipeline never breaks in offline environments.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class RunConfig:
    """
    Complete configuration for one experiment run.
    Logged to W&B at run start — every result is traceable to its config.
    """
    # Experiment identity
    project: str = "blind-flowdps"
    run_name: str = "unnamed"
    seed: int = 42

    # Task
    task: str = "deblur_gauss"
    dataset: str = "div2k"

    # Degradation
    deg_scale: float = 8.0
    sigma_true: Optional[float] = None  # None = blind (unknown)

    # Diffusion
    NFE: int = 28
    cfg_scale: float = 2.0
    img_size: int = 256

    # Data consistency
    dc_stepsize: float = 15.0
    dc_num_iters: int = 7

    # Kernel estimation (blind mode)
    kernel_iterations: int = 15
    kernel_lr: float = 0.3
    sigma_init: tuple = (10.0, 10.0)
    sigma_min: float = 0.5
    sigma_max: float = 25.0


class ExperimentLogger:
    """
    Logs experiment results to W&B with local JSON fallback.

    Usage:
        logger = ExperimentLogger(config)
        logger.log_image_result(image_id, metrics, images)
        logger.log_sigma_trajectory(image_id, sigma_history)
        logger.finish()
    """

    def __init__(self, config: RunConfig, output_dir: str = "experiments/results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[Dict] = []
        self._wandb_available = False
        self._run = None

        self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialise W&B run, falling back to offline mode if unavailable."""
        try:
            import wandb

            self._run = wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                config=asdict(self.config),
            )
            self._wandb_available = True
            print(f"W&B logging enabled. Run: {self._run.url}")

        except Exception as e:
            print(f"W&B unavailable ({e}). Logging to local JSON only.")
            self._wandb_available = False

    def log_image_result(
        self,
        image_id: str,
        psnr_input: float,
        psnr_recon: float,
        ssim_recon: float,
        sigma_estimated: Optional[Dict] = None,
        images: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Log metrics for one reconstructed image.

        Args:
            image_id: Image filename or identifier (e.g. '0882')
            psnr_input: PSNR of degraded input vs ground truth (dB)
            psnr_recon: PSNR of reconstruction vs ground truth (dB)
            ssim_recon: SSIM of reconstruction vs ground truth
            sigma_estimated: Dict with 'sigma_x', 'sigma_y' from kernel estimation
            images: Dict with 'input', 'recon', 'target' tensors for visual logging
        """
        result = {
            "image_id": image_id,
            "psnr_input": psnr_input,
            "psnr_recon": psnr_recon,
            "psnr_improvement": psnr_recon - psnr_input,
            "ssim_recon": ssim_recon,
        }

        if sigma_estimated:
            result.update({
                "sigma_x_est": sigma_estimated.get("sigma_x"),
                "sigma_y_est": sigma_estimated.get("sigma_y"),
                "kernel_converged": sigma_estimated.get("converged", False),
            })

        self._results.append(result)

        if self._wandb_available and self._run:
            import wandb

            log_dict = {f"image/{k}": v for k, v in result.items()}

            # Log side-by-side image comparison
            if images:
                log_dict["visuals"] = wandb.Image(
                    self._make_comparison_grid(images),
                    caption=f"{image_id} | PSNR: {psnr_recon:.2f} dB (+{psnr_recon-psnr_input:.2f})"
                )

            self._run.log(log_dict)

    def log_sigma_trajectory(
        self,
        image_id: str,
        diffusion_step: int,
        sigma_history: List[float],  # sigma values across kernel estimation steps
        sigma_x_final: float,
        sigma_y_final: float,
    ) -> None:
        """
        Log sigma convergence during kernel estimation for one diffusion step.

        This is the data needed to analyse: does sigma converge? How fast?
        Does it converge to the right value? Logged per diffusion timestep.
        """
        if self._wandb_available and self._run:
            import wandb

            self._run.log({
                "kernel_est/image_id": image_id,
                "kernel_est/diffusion_step": diffusion_step,
                "kernel_est/sigma_x_final": sigma_x_final,
                "kernel_est/sigma_y_final": sigma_y_final,
                "kernel_est/final_loss": sigma_history[-1] if sigma_history else None,
                "kernel_est/loss_reduction": (
                    (sigma_history[0] - sigma_history[-1]) / sigma_history[0]
                    if sigma_history and sigma_history[0] > 0
                    else 0.0
                ),
            })

    def log_summary(self) -> Dict[str, float]:
        """
        Compute and log aggregate metrics across all images.

        Returns:
            Dict with mean/std PSNR and SSIM across the evaluation set.
        """
        if not self._results:
            return {}

        improvements = [r["psnr_improvement"] for r in self._results]
        psnr_recons = [r["psnr_recon"] for r in self._results]
        ssims = [r["ssim_recon"] for r in self._results]

        import numpy as np

        summary = {
            "mean_psnr_improvement": float(np.mean(improvements)),
            "std_psnr_improvement": float(np.std(improvements)),
            "mean_psnr_recon": float(np.mean(psnr_recons)),
            "mean_ssim": float(np.mean(ssims)),
            "num_images": len(self._results),
        }

        if self._wandb_available and self._run:
            self._run.summary.update(summary)

        return summary

    def save_results(self) -> Path:
        """Write all results to a structured JSON file."""
        output_path = self.output_dir / f"{self.config.run_name}_results.json"

        output = {
            "config": asdict(self.config),
            "summary": self.log_summary(),
            "per_image": self._results,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_path}")
        return output_path

    def finish(self) -> None:
        """Finalise logging, save results, close W&B run."""
        path = self.save_results()

        if self._wandb_available and self._run:
            self._run.finish()

        print(f"Run complete. Results at: {path}")

    @staticmethod
    def _make_comparison_grid(images: Dict[str, torch.Tensor]) -> "np.ndarray":
        """Concatenate input/recon/target horizontally for visual logging."""
        import numpy as np

        frames = []
        for key in ["input", "recon", "target"]:
            if key in images:
                img = images[key]
                if img.dim() == 4:
                    img = img[0]
                frame = (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(frame)

        return np.concatenate(frames, axis=1) if frames else np.zeros((256, 256, 3), dtype=np.uint8)
