from pathlib import Path

import torch
from torch import nn

from .vmamba_highres import HighResolutionVSSM


class HighResolutionVMUNet(nn.Module):
    """VM-UNet with the combined F0/F1 high-resolution improvement."""

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
        load_ckpt_path=None,
    ):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = HighResolutionVSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
            shallow_channels=shallow_channels,
            fusion_depth=fusion_depth,
            upsample_mode=upsample_mode,
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return logits

    def load_from(self):
        if self.load_ckpt_path is None:
            return

        checkpoint_path = Path(self.load_ckpt_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"VMamba pre-trained checkpoint not found: {checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

        pretrained_encoder = checkpoint.get("model", checkpoint)
        if not isinstance(pretrained_encoder, dict):
            raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

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
                f"No compatible VMamba weights were found in {checkpoint_path}"
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

        loaded_patch_embed_keys = sorted(
            key
            for key in encoder_weights
            if key.startswith("patch_embed.")
        )
        patch_projection_loaded = (
            "patch_embed.proj.weight" in encoder_weights
        )
        print(
            "Loaded VMamba pre-training from {}: encoder {} tensors, "
            "decoder {} tensors, patch projection weight loaded: {}.".format(
                checkpoint_path,
                len(encoder_weights),
                len(decoder_weights),
                patch_projection_loaded,
            )
        )
        print(
            "Compatible Patch Embedding tensors loaded: {}.".format(
                loaded_patch_embed_keys or "none"
            )
        )
        if not patch_projection_loaded:
            print(
                "High-resolution patch projection weight was initialized "
                "separately because its shape differs from the RGB "
                "pre-trained projection."
            )
