"""Sobel response, robust normalization, safe-FOV masking and pyramids."""

import torch
from torch import nn
from torch.nn import functional as F

from .kernels import get_normalized_kernels, get_variant_spec


IMAGENET_GREEN_MEAN = 0.456
IMAGENET_GREEN_STD = 0.224


def restore_green_channel(normalized_rgb):
    """Recover the augmented green channel in [0, 1] from ImageNet RGB input."""
    if normalized_rgb.ndim != 4 or normalized_rgb.size(1) < 2:
        raise ValueError(
            "Expected normalized RGB input with shape Bx3xHxW, got {}".format(
                tuple(normalized_rgb.shape)
            )
        )
    green = normalized_rgb[:, 1:2] * IMAGENET_GREEN_STD + IMAGENET_GREEN_MEAN
    return green.clamp(0.0, 1.0)


def erode_binary_mask(mask, radius=2):
    """Erode a BCHW mask, treating pixels outside the image as invalid."""
    if mask.ndim != 4 or mask.size(1) != 1:
        raise ValueError("Expected a Bx1xHxW FOV mask")
    mask = (mask > 0.5).to(dtype=torch.float32)
    if radius <= 0:
        return mask
    background = 1.0 - mask
    background = F.pad(
        background,
        (radius, radius, radius, radius),
        mode="constant",
        value=1.0,
    )
    dilated_background = F.max_pool2d(
        background,
        kernel_size=2 * radius + 1,
        stride=1,
    )
    return 1.0 - dilated_background


def apply_spatial_guidance(features, guidance, strength=1.0):
    """Apply ``F * (1 + strength * G)`` to NCHW or NHWC features."""
    if guidance.ndim != 4 or guidance.size(1) != 1:
        raise ValueError("Guidance must have shape Bx1xHxW")
    if features.ndim != 4:
        raise ValueError("Features must be a four-dimensional tensor")

    if features.shape[-2:] == guidance.shape[-2:]:
        gate = guidance
    elif features.shape[1:3] == guidance.shape[-2:]:
        gate = guidance.permute(0, 2, 3, 1)
    else:
        raise ValueError(
            "Feature resolution {} does not match guidance {}".format(
                tuple(features.shape), tuple(guidance.shape)
            )
        )
    gate = gate.to(device=features.device, dtype=features.dtype)
    return features * (1.0 + float(strength) * gate)


class SobelGuidance(nn.Module):
    """Fixed, non-trainable Sobel guidance for one operator variant."""

    def __init__(
        self,
        variant,
        q_value,
        strength=1.0,
        eps=1e-6,
        fov_erosion_radius=2,
    ):
        super().__init__()
        if q_value is None or float(q_value) <= 0:
            raise ValueError("q_value must be a positive training-set statistic")
        if float(strength) < 0:
            raise ValueError("Sobel guidance strength must be non-negative")
        if float(eps) <= 0:
            raise ValueError("eps must be positive")

        spec = get_variant_spec(variant)
        self.variant = variant
        self.kernel_size = spec["kernel_size"]
        self.num_directions = len(spec["directions"])
        self.strength = float(strength)
        self.eps = float(eps)
        self.fov_erosion_radius = int(fov_erosion_radius)
        self.register_buffer("kernels", get_normalized_kernels(variant))
        self.register_buffer("q_value", torch.tensor(float(q_value)))

    def extra_repr(self):
        return (
            "variant={!r}, directions={}, kernel_size={}, q_value={:.8g}, "
            "strength={}, eps={}, fov_erosion_radius={}".format(
                self.variant,
                self.num_directions,
                self.kernel_size,
                self.q_value.item(),
                self.strength,
                self.eps,
                self.fov_erosion_radius,
            )
        )

    def compute_raw_energy(self, normalized_rgb):
        """Compute the unnormalized directional RMS energy on restored green."""
        green = restore_green_channel(normalized_rgb)
        radius = self.kernel_size // 2
        padded = F.pad(green, (radius, radius, radius, radius), mode="reflect")
        kernels = self.kernels.to(device=green.device, dtype=green.dtype)
        responses = F.conv2d(padded, kernels)
        energy = torch.sqrt(
            self.eps
            + (2.0 / float(self.num_directions))
            * responses.square().sum(dim=1, keepdim=True)
        )
        return energy

    def normalize_energy(self, raw_energy, safe_fov_mask=None):
        q_value = self.q_value.to(device=raw_energy.device, dtype=raw_energy.dtype)
        guidance = (raw_energy / q_value).clamp(0.0, 1.0)
        if safe_fov_mask is not None:
            if safe_fov_mask.shape != guidance.shape:
                raise ValueError(
                    "Safe FOV shape {} does not match guidance {}".format(
                        tuple(safe_fov_mask.shape), tuple(guidance.shape)
                    )
                )
            guidance = guidance * (safe_fov_mask > 0.5).to(guidance.dtype)
        return guidance

    def prepare_full_image(self, normalized_rgb, fov_mask):
        """Compute full-image guidance once before validation/test window crops."""
        safe_fov = erode_binary_mask(fov_mask, self.fov_erosion_radius)
        raw_energy = self.compute_raw_energy(normalized_rgb)
        return self.normalize_energy(raw_energy, safe_fov), safe_fov

    def prepare_training_patch(self, normalized_rgb, safe_fov_mask):
        """Compute online guidance after augmentation for one training patch."""
        raw_energy = self.compute_raw_energy(normalized_rgb)
        return self.normalize_energy(raw_energy, safe_fov_mask)

    @staticmethod
    def build_pyramid(guidance, safe_fov_mask, levels=6):
        """Build max-pooled guidance and conservative min-pooled FOV masks."""
        if levels < 1:
            raise ValueError("levels must be at least one")
        if guidance.shape != safe_fov_mask.shape:
            raise ValueError("Guidance and safe FOV must have the same shape")

        current_guidance = guidance
        current_mask = (safe_fov_mask > 0.5).to(guidance.dtype)
        guidance_levels = [current_guidance * current_mask]
        mask_levels = [current_mask]
        for _ in range(levels - 1):
            if min(current_guidance.shape[-2:]) < 2:
                raise ValueError("Cannot downsample the requested number of levels")
            next_mask = 1.0 - F.max_pool2d(
                1.0 - current_mask,
                kernel_size=2,
                stride=2,
            )
            next_guidance = F.max_pool2d(
                current_guidance,
                kernel_size=2,
                stride=2,
            ) * next_mask
            guidance_levels.append(next_guidance)
            mask_levels.append(next_mask)
            current_guidance = next_guidance
            current_mask = next_mask
        return tuple(guidance_levels), tuple(mask_levels)

    def gate(self, features, guidance):
        return apply_spatial_guidance(features, guidance, self.strength)
