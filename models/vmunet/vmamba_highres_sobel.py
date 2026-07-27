"""Scheme-2 Sobel-guided high-resolution VM-UNet backbone.

The Sobel branch is deliberately isolated from ``vmamba_highres.py``. Encoder
guidance is applied after every VSS stage. The guided post-VSS tensors feed both
Patch Merging and the matching decoder skip, so all E0-E3 stages share one
consistent semantic definition.
"""

import torch
from torch.utils import checkpoint

from .sobel_guidance import SobelGuidance
from .vmamba_highres import HighResolutionVSSM


def _run_stage_blocks(layer, x):
    """Run only the VSS blocks, leaving sampling to the caller."""
    for block in layer.blocks:
        if layer.use_checkpoint:
            x = checkpoint.checkpoint(block, x)
        else:
            x = block(x)
    return x


class SobelGuidedHighResolutionVSSM(HighResolutionVSSM):
    """High-resolution VSSM using post-VSS Sobel guidance and guided skips."""

    def __init__(
        self,
        sobel_operator,
        sobel_q,
        guidance_strength=1.0,
        sobel_eps=1e-6,
        fov_erosion_radius=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sobel_guidance = SobelGuidance(
            variant=sobel_operator,
            q_value=sobel_q,
            strength=guidance_strength,
            eps=sobel_eps,
            fov_erosion_radius=fov_erosion_radius,
        )

    @property
    def sobel_operator(self):
        return self.sobel_guidance.variant

    def prepare_full_guidance(self, normalized_rgb, fov_mask):
        """Prepare one 576x576 map before validation/test sliding windows."""
        return self.sobel_guidance.prepare_full_image(normalized_rgb, fov_mask)

    def _resolve_patch_guidance(
        self,
        normalized_rgb,
        safe_fov_mask=None,
        guidance_map=None,
    ):
        batch, _, height, width = normalized_rgb.shape
        if safe_fov_mask is None:
            safe_fov_mask = normalized_rgb.new_ones((batch, 1, height, width))
        else:
            safe_fov_mask = safe_fov_mask.to(
                device=normalized_rgb.device,
                dtype=normalized_rgb.dtype,
            )

        if guidance_map is None:
            guidance_map = self.sobel_guidance.prepare_training_patch(
                normalized_rgb,
                safe_fov_mask,
            )
        else:
            guidance_map = guidance_map.to(
                device=normalized_rgb.device,
                dtype=normalized_rgb.dtype,
            )
            if guidance_map.shape != safe_fov_mask.shape:
                raise ValueError(
                    "Guidance patch {} and safe FOV {} must match".format(
                        tuple(guidance_map.shape), tuple(safe_fov_mask.shape)
                    )
                )
            guidance_map = guidance_map * (safe_fov_mask > 0.5).to(
                guidance_map.dtype
            )

        return self.sobel_guidance.build_pyramid(
            guidance_map,
            safe_fov_mask,
            levels=6,
        )[0]

    def forward_features_guided(self, f1, guidance_levels):
        """Encode with post-VSS guidance at 48, 24, 12 and 6 pixels."""
        x = self.patch_embed(f1)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        guided_skips = []
        for stage_index, layer in enumerate(self.layers):
            x = _run_stage_blocks(layer, x)
            x = self.sobel_guidance.gate(
                x,
                guidance_levels[stage_index + 2],
            )
            if layer.downsample is not None:
                guided_skips.append(x)
                x = layer.downsample(x)
        return x, guided_skips

    def forward_features_up_guided(self, x, guided_skips):
        """Decode as PatchExpand -> add same-resolution skip -> decoder VSS."""
        if len(guided_skips) != self.num_layers - 1:
            raise ValueError(
                "Expected {} guided skips, received {}".format(
                    self.num_layers - 1,
                    len(guided_skips),
                )
            )

        for decoder_index, layer_up in enumerate(self.layers_up):
            if layer_up.upsample is not None:
                x = layer_up.upsample(x)
                x = x + guided_skips[-decoder_index]
            x = _run_stage_blocks(layer_up, x)
        return x

    def forward(
        self,
        normalized_rgb,
        safe_fov_mask=None,
        guidance_map=None,
    ):
        guidance_levels = self._resolve_patch_guidance(
            normalized_rgb,
            safe_fov_mask=safe_fov_mask,
            guidance_map=guidance_map,
        )

        # Raw F0 produces raw F1. Only the dedicated decoder F0 branch is gated.
        f0, f1 = self.high_resolution_module.extract(normalized_rgb)
        guided_f0 = self.sobel_guidance.gate(f0, guidance_levels[0])
        guided_f1 = self.sobel_guidance.gate(f1, guidance_levels[1])

        x, guided_skips = self.forward_features_guided(
            guided_f1,
            guidance_levels,
        )
        x = self.forward_features_up_guided(x, guided_skips)
        return self.forward_final(x, guided_f0, guided_f1)
