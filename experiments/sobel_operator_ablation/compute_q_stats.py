"""Compute leakage-free q_0.99 statistics on unaugmented training images."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vmunet.sobel_guidance.guidance import (  # noqa: E402
    SobelGuidance,
    erode_binary_mask,
)
from models.vmunet.sobel_guidance.kernels import get_variant_spec  # noqa: E402
from variants import OPERATOR_NAMES  # noqa: E402


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _training_pairs(data_root, dataset):
    split_root = Path(data_root) / dataset / "train"
    image_dir = split_root / "images"
    fov_dir = split_root / "fov_masks"
    images = {path.stem: path for path in image_dir.glob("*.png")}
    fov_masks = {path.stem: path for path in fov_dir.glob("*.png")}
    if not images:
        raise FileNotFoundError("No training PNG images found in {}".format(image_dir))
    if images.keys() != fov_masks.keys():
        raise ValueError(
            "Training image/FOV names differ for {}: missing FOV={}, extra FOV={}".format(
                dataset,
                sorted(images.keys() - fov_masks.keys()),
                sorted(fov_masks.keys() - images.keys()),
            )
        )
    return [(images[name], fov_masks[name]) for name in sorted(images)]


def _load_normalized_image(path):
    with Image.open(path) as image_file:
        image = np.array(image_file.convert("RGB"), dtype=np.uint8, copy=True)
    if image.shape[:2] != (576, 576):
        raise ValueError(
            "q statistics require 576x576 training images; {} is {}".format(
                path,
                image.shape[:2],
            )
        )
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _load_fov(path):
    with Image.open(path) as mask_file:
        mask = np.array(mask_file.convert("L"), dtype=np.uint8, copy=True)
    if mask.shape != (576, 576):
        raise ValueError(
            "q statistics require 576x576 FOV masks; {} is {}".format(
                path,
                mask.shape,
            )
        )
    return torch.from_numpy(mask >= 128).unsqueeze(0).unsqueeze(0).float()


def compute_dataset_statistics(
    data_root,
    dataset,
    quantile=0.99,
    fov_erosion_radius=2,
    eps=1e-6,
):
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    pairs = _training_pairs(data_root, dataset)
    values_by_operator = {name: [] for name in OPERATOR_NAMES}
    modules = {
        name: SobelGuidance(
            variant=name,
            q_value=1.0,
            strength=1.0,
            eps=eps,
            fov_erosion_radius=fov_erosion_radius,
        ).eval()
        for name in OPERATOR_NAMES
    }

    with torch.no_grad():
        for image_path, fov_path in pairs:
            normalized_image = _load_normalized_image(image_path)
            fov_mask = _load_fov(fov_path)
            safe_fov = erode_binary_mask(fov_mask, fov_erosion_radius).bool()
            if not torch.any(safe_fov):
                raise ValueError("Empty safe FOV for {}".format(image_path))
            for operator, module in modules.items():
                raw_energy = module.compute_raw_energy(normalized_image)
                values_by_operator[operator].append(
                    raw_energy[safe_fov].cpu().numpy()
                )

    variant_statistics = {}
    for operator, value_parts in values_by_operator.items():
        values = np.concatenate(value_parts)
        spec = get_variant_spec(operator)
        variant_statistics[operator] = {
            "q": float(np.quantile(values, quantile)),
            "kernel_size": spec["kernel_size"],
            "directions": list(spec["directions"]),
            "valid_pixel_count": int(values.size),
        }

    return {
        "dataset": dataset,
        "source_split": "train",
        "unaugmented_full_image_size": 576,
        "quantile": quantile,
        "fov_erosion_radius": fov_erosion_radius,
        "eps": eps,
        "kernel_normalization": "per-direction L2",
        "padding": "reflect",
        "variants": variant_statistics,
    }


def write_statistics(statistics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "retinal_576",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("DRIVE", "STARE"),
        default=("DRIVE", "STARE"),
    )
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--fov-erosion-radius", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "q_stats",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        statistics = compute_dataset_statistics(
            data_root=args.data_root,
            dataset=dataset,
            quantile=args.quantile,
            fov_erosion_radius=args.fov_erosion_radius,
            eps=args.eps,
        )
        output_path = args.output_dir / "{}.json".format(dataset)
        write_statistics(statistics, output_path)
        print("Wrote {}".format(output_path))
        for operator in OPERATOR_NAMES:
            print(
                "  {} q={:.10g}".format(
                    operator,
                    statistics["variants"][operator]["q"],
                )
            )


if __name__ == "__main__":
    main()
