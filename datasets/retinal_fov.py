"""Generate deterministic field-of-view masks for retinal fundus images."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_FOV_THRESHOLD = 50


def _largest_connected_component(binary_mask):
    """Return the largest 4-connected foreground component."""
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    largest_component = []

    for start_y, start_x in zip(*np.nonzero(binary_mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and binary_mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    queue.append((neighbor_y, neighbor_x))

        if len(component) > len(largest_component):
            largest_component = component

    result = np.zeros_like(binary_mask, dtype=bool)
    if largest_component:
        y_coordinates, x_coordinates = zip(*largest_component)
        result[y_coordinates, x_coordinates] = True
    return result


def _fill_holes(binary_mask):
    """Fill background regions that do not connect to an image border."""
    height, width = binary_mask.shape
    exterior = np.zeros_like(binary_mask, dtype=bool)
    queue = deque()

    for x in range(width):
        for y in (0, height - 1):
            if not binary_mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not binary_mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for neighbor_y, neighbor_x in (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        ):
            if (
                0 <= neighbor_y < height
                and 0 <= neighbor_x < width
                and not binary_mask[neighbor_y, neighbor_x]
                and not exterior[neighbor_y, neighbor_x]
            ):
                exterior[neighbor_y, neighbor_x] = True
                queue.append((neighbor_y, neighbor_x))

    return binary_mask | (~binary_mask & ~exterior)


def estimate_fov_mask(rgb_image, threshold=DEFAULT_FOV_THRESHOLD):
    """Estimate the valid retinal field without using the vessel annotation."""
    image = np.asarray(rgb_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")

    foreground = np.max(image, axis=2) > threshold
    field_of_view = _fill_holes(_largest_connected_component(foreground))
    if not np.any(field_of_view):
        raise ValueError("Unable to identify a retinal field of view")
    return field_of_view.astype(np.uint8)


def add_fov_masks(data_root, datasets):
    """Create one PNG FOV mask beside every prepared retinal image."""
    data_root = Path(data_root)
    written = 0
    for dataset_name in datasets:
        dataset_root = data_root / dataset_name
        for split_name in ("train", "val", "test"):
            image_dir = dataset_root / split_name / "images"
            output_dir = dataset_root / split_name / "fov_masks"
            output_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(image_dir.glob("*.png")):
                with Image.open(image_path) as image_file:
                    rgb_image = np.array(
                        image_file.convert("RGB"), dtype=np.uint8, copy=True
                    )
                fov_mask = estimate_fov_mask(rgb_image)
                Image.fromarray(fov_mask * 255).save(
                    output_dir / image_path.name
                )
                written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("./data/retinal_576")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("DRIVE", "STARE"),
        default=("DRIVE", "STARE"),
    )
    args = parser.parse_args()
    written = add_fov_masks(args.data_root, args.datasets)
    print(f"Wrote {written} FOV masks under {args.data_root}")


if __name__ == "__main__":
    main()
