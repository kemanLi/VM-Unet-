"""Prepare leakage-safe 576x576 DRIVE/STARE splits.

The source layout is expected to be:

    data/<DATASET>/train/{images,masks}
    data/<DATASET>/val/{images,masks}

The source ``train`` directory is split at original-image level into 80% train
and 20% validation. The source ``val`` directory is preserved as the final
test set. Images are resized with bilinear interpolation and masks with nearest
neighbor interpolation.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from datasets.retinal_fov import estimate_fov_mask
except ModuleNotFoundError:
    from retinal_fov import estimate_fov_mask


def paired_files(split_dir):
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    images = {path.stem: path for path in image_dir.iterdir() if path.is_file()}
    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    if images.keys() != masks.keys():
        raise ValueError(
            f"Unpaired files in {split_dir}: "
            f"missing masks={sorted(images.keys() - masks.keys())}, "
            f"missing images={sorted(masks.keys() - images.keys())}"
        )
    return [(images[name], masks[name], name) for name in sorted(images)]


def save_pair(pair, destination, size):
    image_path, mask_path, name = pair
    image_output = destination / "images" / f"{name}.png"
    mask_output = destination / "masks" / f"{name}.png"
    fov_output = destination / "fov_masks" / f"{name}.png"
    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    fov_output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        original_size = image.size
        resized_image = image.resize((size, size), Image.Resampling.BILINEAR)
        fov_mask = estimate_fov_mask(
            np.array(resized_image, dtype=np.uint8, copy=True)
        )
        resized_image.save(image_output)
        Image.fromarray(fov_mask * 255).save(fov_output)

    with Image.open(mask_path) as mask_file:
        mask = mask_file.convert("L").resize((size, size), Image.Resampling.NEAREST)
        mask = mask.point(lambda value: 255 if value >= 128 else 0)
        mask.save(mask_output)

    return {"case_name": name, "original_size": list(original_size)}


def prepare_dataset(source_root, output_root, dataset_name, size, seed, overwrite):
    source = source_root / dataset_name
    destination = output_root / dataset_name
    if destination.exists() and any(destination.rglob("*")) and not overwrite:
        raise FileExistsError(
            f"{destination} is not empty. Use --overwrite to regenerate it."
        )
    if destination.exists() and overwrite:
        resolved_output = output_root.resolve()
        resolved_destination = destination.resolve()
        if resolved_destination.parent != resolved_output:
            raise ValueError(f"Unsafe output destination: {resolved_destination}")
        shutil.rmtree(resolved_destination)

    train_pool = paired_files(source / "train")
    test_pairs = paired_files(source / "val")
    names = [pair[2] for pair in train_pool]
    shuffled_names = names.copy()
    random.Random(seed).shuffle(shuffled_names)
    validation_count = max(1, round(len(shuffled_names) * 0.2))
    validation_names = set(shuffled_names[:validation_count])
    train_pairs = [pair for pair in train_pool if pair[2] not in validation_names]
    validation_pairs = [pair for pair in train_pool if pair[2] in validation_names]

    manifest = {
        "dataset": dataset_name,
        "image_size": [size, size],
        "seed": seed,
        "source_train_count": len(train_pool),
        "source_test_count": len(test_pairs),
        "splits": {},
    }
    for split_name, pairs in (
        ("train", train_pairs),
        ("val", validation_pairs),
        ("test", test_pairs),
    ):
        records = [
            save_pair(pair, destination / split_name, size)
            for pair in pairs
        ]
        manifest["splits"][split_name] = records

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-root", type=Path, default=Path("./data/retinal_576"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("DRIVE", "STARE"),
        default=("DRIVE", "STARE"),
    )
    parser.add_argument("--size", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        manifest = prepare_dataset(
            source_root=args.source_root,
            output_root=args.output_root,
            dataset_name=dataset_name,
            size=args.size,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        split_counts = {
            name: len(records) for name, records in manifest["splits"].items()
        }
        print(f"{dataset_name}: {split_counts}")


if __name__ == "__main__":
    main()
