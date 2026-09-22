"""
Main evaluation pipeline for Blind-FlowDPS.

Runs blind Gaussian deblurring on a dataset and logs results to W&B.

Usage:
    python scripts/evaluate.py --config experiments/configs/div2k_eval.yaml
    python scripts/evaluate.py --config experiments/configs/div2k_eval.yaml --num_images 3
    python scripts/evaluate.py --image data/test.png --deg_scale 8

The config YAML is the single source of truth for hyperparameters.
Command-line args override config values (useful for quick tests).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm

# Ensure src is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.image_metrics import evaluate_image_pair, compute_kernel_metrics
from src.operators.gaussian_blur import GaussianBlurConfig, GaussianBlurOperator
from src.utils.image_io import load_image, save_image
from src.utils.logger import ExperimentLogger, RunConfig
from src.utils.reproducibility import seed_everything


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind-FlowDPS evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/div2k_eval.yaml",
        help="Path to YAML experiment config",
    )

    # Overrides (take priority over config)
    parser.add_argument("--image", type=str, help="Single image path (overrides dataset)")
    parser.add_argument("--data_dir", type=str, help="Dataset directory override")
    parser.add_argument("--num_images", type=int, help="Number of images override")
    parser.add_argument("--deg_scale", type=float, help="True blur sigma override")
    parser.add_argument("--NFE", type=int, help="Diffusion steps override")
    parser.add_argument("--output_dir", type=str, help="Output directory override")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose kernel estimation output for debugging",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(args: argparse.Namespace) -> dict:
    """Load YAML config and apply command-line overrides."""
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply overrides — command-line args win over YAML
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.num_images:
        config["data"]["num_images"] = args.num_images
    if args.deg_scale:
        config["degradation"]["deg_scale"] = args.deg_scale
    if args.NFE:
        config["diffusion"]["NFE"] = args.NFE
    if args.output_dir:
        config["logging"]["output_dir"] = args.output_dir
    if args.no_wandb:
        config["logging"]["log_wandb"] = False

    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def get_image_paths(config: dict, single_image: Optional[str] = None):
    """Return list of (image_id, clean_image_path) tuples."""
    if single_image:
        return [("single", Path(single_image))]

    data_dir = Path(config["data"]["data_dir"])
    num_images = config["data"]["num_images"]

    # Sort to ensure consistent ordering across runs
    all_images = sorted(data_dir.glob("*.png")) + sorted(data_dir.glob("*.jpg"))

    if not all_images:
        raise FileNotFoundError(f"No images found in {data_dir}")

    selected = all_images[:num_images]
    return [(p.stem, p) for p in selected]


# ---------------------------------------------------------------------------
# Degradation pipeline
# ---------------------------------------------------------------------------


def degrade_image(
    clean: torch.Tensor,
    operator: GaussianBlurOperator,
    deg_scale: float,
    noise_level: float,
    seed: int,
) -> torch.Tensor:
    """
    Apply Gaussian blur + additive noise to a clean image.

    Uses a fixed seed so the same degradation is applied reproducibly.
    """
    torch.manual_seed(seed)

    # Blur
    degraded = operator.forward(clean, sigma_x=deg_scale, sigma_y=deg_scale)

    # Noise
    noise = torch.randn_like(degraded) * noise_level
    degraded = (degraded + noise).clamp(0, 1)

    return degraded


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------


def run_solver(
    degraded: torch.Tensor,
    operator: GaussianBlurOperator,
    config: dict,
    device: str,
    debug: bool = False,
) -> tuple[torch.Tensor, list]:
    """
    Run the FlowDPS solver with blind kernel estimation.

    This is where the FlowDPS solver integrates. Currently returns a
    placeholder — replace with actual solver call once SD3 weights are loaded.

    Args:
        degraded: Degraded measurement y, shape (1, C, H, W)
        operator: BlindDeblurring operator with estimate_params()
        config: Full experiment config
        device: CUDA device string
        debug: Print sigma trajectories during solving

    Returns:
        Tuple of (reconstructed image tensor, list of sigma estimates per step)
    """
    # -----------------------------------------------------------------------
    # INTEGRATION POINT
    # Replace everything between these comments with the actual FlowDPS call.
    #
    # The solver should:
    # 1. Accept operator.estimate_params() as the kernel estimation function
    # 2. Call it inside the data_consistency loop at each diffusion timestep
    # 3. Pass estimated sigma_x, sigma_y to operator.forward() and operator.adjoint()
    # 4. Return the final reconstructed image
    #
    # See docs/method.md for the full pseudocode.
    #
    # Example (once solver is integrated):
    #
    #   from src.solvers.flowdps import FlowDPSSolver
    #   solver = FlowDPSSolver.from_pretrained("sd3-medium", device=device)
    #   recon, sigma_history = solver.sample(
    #       measurement=degraded,
    #       operator=operator,
    #       blind=True,
    #       NFE=config["diffusion"]["NFE"],
    #       cfg_scale=config["diffusion"]["cfg_scale"],
    #       dc_stepsize=config["data_consistency"]["stepsize"],
    #       dc_num_iters=config["data_consistency"]["num_iters"],
    #       kernel_iterations=config["kernel_estimation"]["iterations"],
    #       kernel_lr=config["kernel_estimation"]["lr"],
    #       debug=debug,
    #   )
    #   return recon, sigma_history
    # -----------------------------------------------------------------------

    # PLACEHOLDER — remove when solver is integrated
    print("  [NOTE] Solver not yet integrated. Returning degraded image as placeholder.")
    return degraded.clone(), []


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    config = load_config(args)

    seed = config["experiment"]["seed"]
    seed_everything(seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build operator
    op_config = GaussianBlurConfig(
        kernel_size=33,
        sigma_min=config["kernel_estimation"]["sigma_min"],
        sigma_max=config["kernel_estimation"]["sigma_max"],
        sigma_init=tuple(config["kernel_estimation"]["sigma_init"]),
        device=device,
    )
    operator = GaussianBlurOperator(op_config)

    # Build logger
    run_config = RunConfig(
        project=config["experiment"]["project"],
        run_name=config["experiment"]["run_name"],
        seed=seed,
        task=config["degradation"]["task"],
        dataset=config["data"]["dataset"],
        deg_scale=config["degradation"]["deg_scale"],
        NFE=config["diffusion"]["NFE"],
        cfg_scale=config["diffusion"]["cfg_scale"],
        dc_stepsize=config["data_consistency"]["stepsize"],
        dc_num_iters=config["data_consistency"]["num_iters"],
        kernel_iterations=config["kernel_estimation"]["iterations"],
        kernel_lr=config["kernel_estimation"]["lr"],
        sigma_init=tuple(config["kernel_estimation"]["sigma_init"]),
    )

    logger = ExperimentLogger(
        config=run_config,
        output_dir=config["logging"]["output_dir"],
    )

    # Get image paths
    image_paths = get_image_paths(config, args.image)
    print(f"\nEvaluating {len(image_paths)} images...")

    deg_scale = config["degradation"]["deg_scale"]
    noise_level = config["degradation"]["noise_level"]
    save_images = config["logging"].get("save_images", True)
    output_dir = Path(config["logging"]["output_dir"])

    # ---------------------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------------------

    for idx, (image_id, image_path) in enumerate(tqdm(image_paths, desc="Evaluating")):
        print(f"\n[{idx+1}/{len(image_paths)}] Image: {image_id}")

        # Load clean image
        clean = load_image(image_path, size=(256, 256), device=device)

        # Degrade with known sigma (sigma NOT given to solver — blind setting)
        degraded = degrade_image(
            clean, operator, deg_scale, noise_level, seed=seed + idx
        )

        # Run solver (blind — no sigma given)
        recon, sigma_history = run_solver(
            degraded, operator, config, device, debug=args.debug
        )

        # Compute metrics
        metrics = evaluate_image_pair(pred=recon, target=clean, degraded=degraded)

        print(
            f"  Input PSNR:  {metrics.input_psnr.value:.2f} dB\n"
            f"  Recon PSNR:  {metrics.psnr.value:.2f} dB\n"
            f"  Improvement: {metrics.psnr_improvement:+.2f} dB\n"
            f"  SSIM:        {metrics.ssim.value:.4f}"
        )

        # Log to W&B
        logger.log_image_result(
            image_id=image_id,
            psnr_input=metrics.input_psnr.value,
            psnr_recon=metrics.psnr.value,
            ssim_recon=metrics.ssim.value,
            images={
                "input": degraded,
                "recon": recon,
                "target": clean,
            },
        )

        # Save output images
        if save_images:
            save_image(recon, output_dir / "recon" / f"{image_id}.png")
            save_image(degraded, output_dir / "input" / f"{image_id}.png")
            save_image(clean, output_dir / "target" / f"{image_id}.png")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    summary = logger.log_summary()
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Images evaluated:     {summary.get('num_images')}")
    print(f"  Mean PSNR improvement: {summary.get('mean_psnr_improvement', 0):+.2f} dB")
    print(f"  Std:                  ±{summary.get('std_psnr_improvement', 0):.2f} dB")
    print(f"  Mean recon PSNR:      {summary.get('mean_psnr_recon', 0):.2f} dB")
    print(f"  Mean SSIM:            {summary.get('mean_ssim', 0):.4f}")
    print("=" * 50)

    logger.finish()


if __name__ == "__main__":
    main()
