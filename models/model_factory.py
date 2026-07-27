from models.vmunet.vmunet import VMUNet
from models.vmunet.vmunet_highres import HighResolutionVMUNet
from models.vmunet.vmunet_highres_sobel import (
    SobelGuidedHighResolutionVMUNet,
)


_MODEL_NAMES = ("vmunet", "vmunet_highres", "vmunet_highres_sobel")


def available_model_names():
    return _MODEL_NAMES


def build_model(model_name, model_config):
    common_kwargs = {
        "num_classes": model_config["num_classes"],
        "input_channels": model_config["input_channels"],
        "depths": model_config["depths"],
        "depths_decoder": model_config["depths_decoder"],
        "drop_path_rate": model_config["drop_path_rate"],
        "load_ckpt_path": model_config["load_ckpt_path"],
    }

    if model_name == "vmunet":
        return VMUNet(**common_kwargs)

    if model_name == "vmunet_highres":
        return HighResolutionVMUNet(
            **common_kwargs,
            shallow_channels=model_config.get(
                "shallow_channels",
                (24, 48),
            ),
            fusion_depth=model_config.get("fusion_depth", 2),
            upsample_mode=model_config.get(
                "upsample_mode",
                "bilinear",
            ),
        )

    if model_name == "vmunet_highres_sobel":
        return SobelGuidedHighResolutionVMUNet(
            **common_kwargs,
            shallow_channels=model_config.get("shallow_channels", (24, 48)),
            fusion_depth=model_config.get("fusion_depth", 2),
            upsample_mode=model_config.get("upsample_mode", "bilinear"),
            sobel_operator=model_config["sobel_operator"],
            sobel_q=model_config["sobel_q"],
            guidance_strength=model_config.get("guidance_strength", 1.0),
            sobel_eps=model_config.get("sobel_eps", 1e-6),
            fov_erosion_radius=model_config.get("fov_erosion_radius", 2),
        )

    raise ValueError(
        "Unknown model {!r}. Available models: {}".format(
            model_name,
            ", ".join(_MODEL_NAMES),
        )
    )
