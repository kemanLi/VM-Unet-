"""One-batch CUDA forward/backward smoke test for the full Sobel VM-UNet."""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import build_model  # noqa: E402
from models.vmunet.sobel_guidance import available_sobel_variants  # noqa: E402
from utils import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("DRIVE", "STARE"), default="DRIVE")
    parser.add_argument(
        "--operator",
        choices=available_sobel_variants(),
        default="s3_d2",
    )
    parser.add_argument(
        "--q-stats-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "q_stats",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The full model smoke test requires CUDA")
    statistics = json.loads(
        (args.q_stats_dir / "{}.json".format(args.dataset)).read_text(
            encoding="utf-8"
        )
    )
    q_value = float(statistics["variants"][args.operator]["q"])
    set_seed(42)

    model = build_model(
        "vmunet_highres_sobel",
        {
            "num_classes": 1,
            "input_channels": 3,
            "depths": [2, 2, 2, 2],
            "depths_decoder": [2, 2, 2, 1],
            "drop_path_rate": 0.2,
            "load_ckpt_path": None,
            "shallow_channels": (24, 48),
            "fusion_depth": 2,
            "upsample_mode": "bilinear",
            "sobel_operator": args.operator,
            "sobel_q": q_value,
            "guidance_strength": 1.0,
            "sobel_eps": 1e-6,
            "fov_erosion_radius": 2,
        },
    ).cuda()
    model.train()
    image = torch.randn(1, 3, 192, 192, device="cuda")
    safe_fov = torch.ones(1, 1, 192, 192, device="cuda")
    output = model(image, safe_fov_mask=safe_fov)
    if output.shape != (1, 1, 192, 192):
        raise RuntimeError("Unexpected output shape: {}".format(tuple(output.shape)))
    loss = output.mean()
    loss.backward()
    if not torch.isfinite(output).all():
        raise RuntimeError("Full-model smoke test produced non-finite output")
    print(
        "CUDA smoke test passed: dataset={}, operator={}, q={:.10g}, "
        "output_shape={}, loss={:.6f}".format(
            args.dataset,
            args.operator,
            q_value,
            tuple(output.shape),
            loss.item(),
        )
    )


if __name__ == "__main__":
    main()
