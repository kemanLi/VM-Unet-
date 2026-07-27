import torch
from torch import nn
from torch.nn import functional as F

from .vmamba import VSSM


def _group_count(channels, maximum_groups=8):
    """Choose the largest valid GroupNorm group count up to maximum_groups."""
    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class ShallowFeatureExtractor(nn.Module):
    """Produce the full-resolution F0 and half-resolution F1 features."""

    def __init__(self, in_channels=3, shallow_channels=(24, 48)):
        super().__init__()
        if len(shallow_channels) != 2:
            raise ValueError("shallow_channels must contain the F0 and F1 widths")

        f0_channels, f1_channels = shallow_channels
        self.full_resolution = nn.Sequential(
            ConvNormAct(in_channels, f0_channels),
            ConvNormAct(f0_channels, f0_channels),
        )
        self.half_resolution = ConvNormAct(
            f0_channels,
            f1_channels,
            stride=2,
        )

    def forward(self, x):
        f0 = self.full_resolution(x)
        f1 = self.half_resolution(f0)
        return f0, f1


class UpFuseBlock(nn.Module):
    """Upsample a decoder feature, concatenate one shallow skip, then fuse."""

    _ALIGN_CORNERS_MODES = {"linear", "bilinear", "bicubic", "trilinear"}

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        fusion_depth=2,
        upsample_mode="bilinear",
    ):
        super().__init__()
        if fusion_depth < 1:
            raise ValueError("fusion_depth must be at least 1")

        self.upsample_mode = upsample_mode
        self.channel_projection = ConvNormAct(in_channels, out_channels)

        fusion_layers = [
            ConvNormAct(out_channels + skip_channels, out_channels)
        ]
        fusion_layers.extend(
            ConvNormAct(out_channels, out_channels)
            for _ in range(fusion_depth - 1)
        )
        self.fusion = nn.Sequential(*fusion_layers)

    def _resize(self, x, spatial_size):
        interpolate_kwargs = {
            "size": spatial_size,
            "mode": self.upsample_mode,
        }
        if self.upsample_mode in self._ALIGN_CORNERS_MODES:
            interpolate_kwargs["align_corners"] = False
        return F.interpolate(x, **interpolate_kwargs)

    def forward(self, x, skip):
        x = self._resize(x, skip.shape[-2:])
        x = self.channel_projection(x)
        x = torch.cat((x, skip), dim=1)
        return self.fusion(x)


class HighResolutionFusionModule(nn.Module):
    """Internal two-stage progressive fusion head for F1 followed by F0."""

    def __init__(
        self,
        decoder_channels=96,
        shallow_channels=(24, 48),
        fusion_depth=2,
        upsample_mode="bilinear",
    ):
        super().__init__()
        f0_channels, f1_channels = shallow_channels
        self.fuse_f1 = UpFuseBlock(
            in_channels=decoder_channels,
            skip_channels=f1_channels,
            out_channels=f1_channels,
            fusion_depth=fusion_depth,
            upsample_mode=upsample_mode,
        )
        self.fuse_f0 = UpFuseBlock(
            in_channels=f1_channels,
            skip_channels=f0_channels,
            out_channels=f0_channels,
            fusion_depth=fusion_depth,
            upsample_mode=upsample_mode,
        )

    def forward(self, x, f0, f1):
        x = self.fuse_f1(x, f1)
        x = self.fuse_f0(x, f0)
        return x


class HighResolutionModule(nn.Module):
    """Own the F0/F1 extraction and fusion as one ablation-level module."""

    def __init__(
        self,
        input_channels=3,
        decoder_channels=96,
        shallow_channels=(24, 48),
        fusion_depth=2,
        upsample_mode="bilinear",
    ):
        super().__init__()
        self.shallow_features = ShallowFeatureExtractor(
            in_channels=input_channels,
            shallow_channels=shallow_channels,
        )
        self.fusion_head = HighResolutionFusionModule(
            decoder_channels=decoder_channels,
            shallow_channels=shallow_channels,
            fusion_depth=fusion_depth,
            upsample_mode=upsample_mode,
        )

    def extract(self, x):
        return self.shallow_features(x)

    def reconstruct(self, x, f0, f1):
        return self.fusion_head(x, f0, f1)


class HighResolutionVSSM(VSSM):
    """VM-UNet backbone with one combined F0/F1 high-resolution module."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1,
        depths=(2, 2, 9, 2),
        depths_decoder=(2, 9, 2, 2),
        dims=(96, 192, 384, 768),
        dims_decoder=(768, 384, 192, 96),
        shallow_channels=(24, 48),
        fusion_depth=2,
        upsample_mode="bilinear",
        drop_path_rate=0.1,
        **kwargs,
    ):
        if len(shallow_channels) != 2:
            raise ValueError("shallow_channels must contain exactly two values")
        if dims[0] != dims_decoder[-1]:
            raise ValueError(
                "The last decoder width must match the first encoder width"
            )

        shallow_channels = tuple(shallow_channels)

        # F1 is half resolution, so a stride-2 patch embedding preserves the
        # original VM-UNet deep feature resolutions (H/4, H/8, H/16, H/32).
        super().__init__(
            patch_size=2,
            in_chans=shallow_channels[1],
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            dims=dims,
            dims_decoder=dims_decoder,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

        # The original one-shot x4 Final_PatchExpand is intentionally replaced
        # by the two-stage F1/F0 fusion head below.
        del self.final_up

        self.input_channels = in_chans
        self.shallow_channels = shallow_channels
        self.high_resolution_module = HighResolutionModule(
            input_channels=in_chans,
            decoder_channels=dims_decoder[-1],
            shallow_channels=shallow_channels,
            fusion_depth=fusion_depth,
            upsample_mode=upsample_mode,
        )
        self.high_resolution_module.apply(
            self._init_high_resolution_modules
        )

    @staticmethod
    def _init_high_resolution_modules(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_final(self, x, f0, f1):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.high_resolution_module.reconstruct(x, f0, f1)
        return self.final_conv(x)

    def forward(self, x):
        f0, f1 = self.high_resolution_module.extract(x)
        x, skip_list = self.forward_features(f1)
        x = self.forward_features_up(x, skip_list)
        return self.forward_final(x, f0, f1)
