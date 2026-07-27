import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _sliding_positions(image_size, patch_size, stride):
    positions = list(range(0, image_size - patch_size + 1, stride))
    last_position = image_size - patch_size
    if not positions or positions[-1] != last_position:
        positions.append(last_position)
    return positions


def _paired_paths(root):
    root = Path(root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    fov_dir = root / "fov_masks"
    images = {path.stem: path for path in image_dir.iterdir() if path.is_file()}
    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    fov_masks = {
        path.stem: path for path in fov_dir.iterdir() if path.is_file()
    } if fov_dir.exists() else {}

    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    missing_fov_masks = sorted(images.keys() - fov_masks.keys())
    extra_fov_masks = sorted(fov_masks.keys() - images.keys())
    if missing_masks or missing_images or missing_fov_masks or extra_fov_masks:
        raise ValueError(
            f"Image/mask/FOV mismatch in {root}. "
            f"Missing masks: {missing_masks}; missing images: {missing_images}"
            f"; missing FOV masks: {missing_fov_masks}; "
            f"extra FOV masks: {extra_fov_masks}. Run "
            f"`python datasets/retinal_fov.py` to generate FOV masks."
        )
    if not images:
        raise ValueError(f"No paired samples found in {root}")
    return [
        (images[name], masks[name], fov_masks[name], name)
        for name in sorted(images)
    ]


def _load_resized_triplet(image_path, mask_path, fov_path):
    with Image.open(image_path) as image_file:
        image = np.array(image_file.convert("RGB"), dtype=np.uint8, copy=True)
    with Image.open(mask_path) as mask_file:
        mask = np.array(mask_file.convert("L"), dtype=np.uint8, copy=True)
    with Image.open(fov_path) as fov_file:
        fov_mask = np.array(fov_file.convert("L"), dtype=np.uint8, copy=True)
    mask = (mask >= 128).astype(np.uint8)
    fov_mask = (fov_mask >= 128).astype(np.uint8)
    return image, mask, fov_mask


def _image_to_tensor(image):
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _mask_to_tensor(mask):
    return torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()


def _erode_fov_mask(fov_mask, radius):
    """Erode the full-image FOV before cropping any training patch."""
    if radius <= 0:
        return np.ascontiguousarray(fov_mask)
    mask = _mask_to_tensor(fov_mask).unsqueeze(0)
    background = F.pad(
        1.0 - mask,
        (radius, radius, radius, radius),
        mode="constant",
        value=1.0,
    )
    safe_mask = 1.0 - F.max_pool2d(
        background,
        kernel_size=2 * radius + 1,
        stride=1,
    )
    return safe_mask.squeeze(0).squeeze(0).numpy().astype(np.uint8)


class RetinalPatchDataset(Dataset):
    """Online random-patch dataset for DRIVE/STARE training."""

    def __init__(
        self,
        split_dir,
        samples_per_epoch=4800,
        patch_size=192,
        patch_stride=48,
        positive_crop_probability=0.5,
        min_fov_fraction=0.0,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
        rotation_probability=0.5,
        photometric_probability=0.3,
        gamma_probability=0.3,
        return_fov_mask=False,
        fov_erosion_radius=2,
    ):
        self.samples = _paired_paths(split_dir)
        self.samples_per_epoch = samples_per_epoch
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.positive_crop_probability = positive_crop_probability
        self.min_fov_fraction = min_fov_fraction
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotation_probability = rotation_probability
        self.photometric_probability = photometric_probability
        self.gamma_probability = gamma_probability
        self.return_fov_mask = bool(return_fov_mask)
        self.fov_erosion_radius = int(fov_erosion_radius)

        if patch_stride <= 0 or patch_stride > patch_size:
            raise ValueError(
                "patch_stride must be positive and no larger than patch_size"
            )

        self.cached = []
        self.patch_candidates = []
        self.positive_patch_candidates = []
        for image_path, mask_path, fov_path, name in self.samples:
            image, mask, fov_mask = _load_resized_triplet(
                image_path, mask_path, fov_path
            )
            height, width = mask.shape
            if height < patch_size or width < patch_size:
                raise ValueError(
                    f"{name} has size {width}x{height}, smaller than patch size {patch_size}"
                )
            sample_index = len(self.cached)
            self.cached.append(
                {
                    "image": image,
                    "mask": mask,
                    "fov_mask": fov_mask,
                    "safe_fov_mask": (
                        _erode_fov_mask(fov_mask, self.fov_erosion_radius)
                        if self.return_fov_mask
                        else None
                    ),
                    "name": name,
                }
            )
            y_positions = _sliding_positions(height, patch_size, patch_stride)
            x_positions = _sliding_positions(width, patch_size, patch_stride)
            for y in y_positions:
                for x in x_positions:
                    fov_patch = fov_mask[
                        y : y + patch_size, x : x + patch_size
                    ]
                    if (
                        min_fov_fraction > 0
                        and np.mean(fov_patch) < min_fov_fraction
                    ):
                        continue
                    candidate = (sample_index, y, x)
                    self.patch_candidates.append(candidate)
                    mask_patch = mask[y : y + patch_size, x : x + patch_size]
                    if np.any(mask_patch):
                        self.positive_patch_candidates.append(candidate)

        if not self.patch_candidates:
            raise ValueError(
                f"No valid {patch_size}x{patch_size} patch candidates found in "
                f"{split_dir}"
            )

    def __len__(self):
        return self.samples_per_epoch

    def _augment(self, image, mask, safe_fov_mask=None):
        if random.random() < self.horizontal_flip_probability:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            if safe_fov_mask is not None:
                safe_fov_mask = np.flip(safe_fov_mask, axis=1)
        if random.random() < self.vertical_flip_probability:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
            if safe_fov_mask is not None:
                safe_fov_mask = np.flip(safe_fov_mask, axis=0)

        if random.random() < self.rotation_probability:
            rotation_k = random.choice((1, 2, 3))
            image = np.rot90(image, rotation_k)
            mask = np.rot90(mask, rotation_k)
            if safe_fov_mask is not None:
                safe_fov_mask = np.rot90(safe_fov_mask, rotation_k)

        image = image.astype(np.float32)
        if random.random() < self.photometric_probability:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            channel_mean = image.mean(axis=(0, 1), keepdims=True)
            image = (image - channel_mean) * contrast + channel_mean
            image = image * brightness
        image = np.clip(image, 0.0, 255.0)

        if random.random() < self.gamma_probability:
            gamma = random.uniform(0.8, 1.2)
            image = 255.0 * np.power(image / 255.0, gamma)

        augmented = (
            np.ascontiguousarray(image.astype(np.uint8)),
            np.ascontiguousarray(mask),
        )
        if safe_fov_mask is None:
            return augmented
        return augmented + (np.ascontiguousarray(safe_fov_mask),)

    def __getitem__(self, _):
        use_positive = (
            self.positive_patch_candidates
            and random.random() < self.positive_crop_probability
        )
        candidates = (
            self.positive_patch_candidates if use_positive else self.patch_candidates
        )
        sample_index, y, x = random.choice(candidates)
        sample = self.cached[sample_index]
        image = sample["image"][y : y + self.patch_size, x : x + self.patch_size]
        mask = sample["mask"][y : y + self.patch_size, x : x + self.patch_size]
        if not self.return_fov_mask:
            image, mask = self._augment(image, mask)
            return _image_to_tensor(image), _mask_to_tensor(mask)

        safe_fov_mask = sample["safe_fov_mask"][
            y : y + self.patch_size,
            x : x + self.patch_size,
        ]
        image, mask, safe_fov_mask = self._augment(
            image,
            mask,
            safe_fov_mask,
        )
        return {
            "image": _image_to_tensor(image),
            "mask": _mask_to_tensor(mask),
            "safe_fov_mask": _mask_to_tensor(safe_fov_mask),
        }


class RetinalFullImageDataset(Dataset):
    """Full resized images for deterministic sliding-window validation/testing."""

    def __init__(self, split_dir):
        self.samples = _paired_paths(split_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, fov_path, name = self.samples[index]
        image, mask, fov_mask = _load_resized_triplet(
            image_path, mask_path, fov_path
        )
        return {
            "image": _image_to_tensor(image),
            "mask": _mask_to_tensor(mask),
            "fov_mask": _mask_to_tensor(fov_mask),
            "case_name": name,
        }
