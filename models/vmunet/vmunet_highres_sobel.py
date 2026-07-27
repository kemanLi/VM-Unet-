"""Public wrapper for the scheme-2 Sobel-guided high-resolution VM-UNet."""

from pathlib import Path

import torch
from torch import nn

from .vmamba_highres_sobel import SobelGuidedHighResolutionVSSM


class SobelGuidedHighResolutionVMUNet(nn.Module):
    """Ablation-level model kept separate from baseline and high-resolution code."""

    supports_sobel_guidance = True

    def __init__(
        self,
        input_channels=3,
        num_classes=1,
        depths=(2, 2, 9, 2),
        depths_decoder=(2, 9, 2, 2),
        drop_path_rate=0.2,
        shallow_channels=(24, 48),
        fusion_depth=2,
        upsample_mode="bilinear",
        sobel_operator="s3_d2",
        sobel_q=None,
        guidance_strength=1.0,
        sobel_eps=1e-6,
        fov_erosion_radius=2,
        load_ckpt_path=None,
    ):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.vmunet = SobelGuidedHighResolutionVSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
            shallow_channels=shallow_channels,
            fusion_depth=fusion_depth,
            upsample_mode=upsample_mode,
            sobel_operator=sobel_operator,
            sobel_q=sobel_q,
            guidance_strength=guidance_strength,
            sobel_eps=sobel_eps,
            fov_erosion_radius=fov_erosion_radius,
        )

    def prepare_full_guidance(self, normalized_rgb, fov_mask):
        return self.vmunet.prepare_full_guidance(normalized_rgb, fov_mask)

    def forward(self, x, safe_fov_mask=None, guidance_map=None):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(
            x,
            safe_fov_mask=safe_fov_mask,
            guidance_map=guidance_map,
        )
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return logits

    def load_from(self):
        if self.load_ckpt_path is None:
            return

        checkpoint_path = Path(self.load_ckpt_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "VMamba pre-trained checkpoint not found: {}".format(
                    checkpoint_path
                )
            )

        try:
            checkpoint_data = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint_data = torch.load(str(checkpoint_path), map_location="cpu")

        pretrained_encoder = checkpoint_data.get("model", checkpoint_data)
        if not isinstance(pretrained_encoder, dict):
            raise ValueError("Unsupported checkpoint format in {}".format(checkpoint_path))
        model_dict = self.vmunet.state_dict()

        def compatible_weights(weights):
            return {
                key: value
                for key, value in weights.items()
                if key in model_dict and model_dict[key].shape == value.shape
            }

        encoder_weights = compatible_weights(pretrained_encoder)
        if not encoder_weights:
            raise RuntimeError(
                "No compatible VMamba weights were found in {}".format(
                    checkpoint_path
                )
            )
        model_dict.update(encoder_weights)

        decoder_candidates = {}
        layer_mapping = {
            "layers.0": "layers_up.3",
            "layers.1": "layers_up.2",
            "layers.2": "layers_up.1",
            "layers.3": "layers_up.0",
        }
        for key, value in pretrained_encoder.items():
            for encoder_name, decoder_name in layer_mapping.items():
                if encoder_name in key:
                    decoder_candidates[
                        key.replace(encoder_name, decoder_name)
                    ] = value
                    break
        decoder_weights = compatible_weights(decoder_candidates)
        model_dict.update(decoder_weights)
        self.vmunet.load_state_dict(model_dict)
        print(
            "Loaded VMamba pre-training from {}: encoder {} tensors, "
            "decoder {} tensors.".format(
                checkpoint_path,
                len(encoder_weights),
                len(decoder_weights),
            )
        )
