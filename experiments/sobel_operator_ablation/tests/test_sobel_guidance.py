import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vmunet.sobel_guidance.guidance import (  # noqa: E402
    SobelGuidance,
    apply_spatial_guidance,
    erode_binary_mask,
    restore_green_channel,
)
from models.vmunet.sobel_guidance.kernels import (  # noqa: E402
    available_sobel_variants,
    get_normalized_kernels,
)


class SobelGuidanceTests(unittest.TestCase):
    def test_kernel_invariants(self):
        expected_counts = {
            "s3_d2": 2,
            "s3_d4": 4,
            "s5_d2": 2,
            "s5_d4": 4,
            "s5_d8": 8,
        }
        for variant in available_sobel_variants():
            kernels = get_normalized_kernels(variant)
            self.assertEqual(kernels.size(0), expected_counts[variant])
            self.assertTrue(
                torch.allclose(
                    kernels.flatten(1).sum(1),
                    torch.zeros(kernels.size(0)),
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    kernels.flatten(1).norm(dim=1),
                    torch.ones(kernels.size(0)),
                )
            )

    def test_inverse_imagenet_green(self):
        raw = torch.rand(2, 3, 13, 15)
        normalized = raw.clone()
        normalized[:, 0] = (raw[:, 0] - 0.485) / 0.229
        normalized[:, 1] = (raw[:, 1] - 0.456) / 0.224
        normalized[:, 2] = (raw[:, 2] - 0.406) / 0.225
        restored = restore_green_channel(normalized)
        self.assertTrue(torch.allclose(restored[:, 0], raw[:, 1], atol=1e-6))

    def test_safe_fov_erosion(self):
        mask = torch.ones(1, 1, 9, 9)
        safe = erode_binary_mask(mask, radius=2)
        self.assertEqual(int(safe.sum().item()), 25)
        self.assertTrue(torch.all(safe[:, :, 2:7, 2:7] == 1))

    def test_six_level_pyramid(self):
        guidance = torch.rand(1, 1, 192, 192)
        safe = torch.ones_like(guidance)
        levels, masks = SobelGuidance.build_pyramid(guidance, safe, levels=6)
        expected = (192, 96, 48, 24, 12, 6)
        self.assertEqual(tuple(item.shape[-1] for item in levels), expected)
        self.assertEqual(tuple(item.shape[-1] for item in masks), expected)

    def test_lambda_one_gate(self):
        features = torch.ones(1, 3, 4, 4)
        guidance = torch.full((1, 1, 4, 4), 0.5)
        guided = apply_spatial_guidance(features, guidance, strength=1.0)
        self.assertTrue(torch.allclose(guided, torch.full_like(features, 1.5)))


if __name__ == "__main__":
    unittest.main()
