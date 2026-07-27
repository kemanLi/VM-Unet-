"""Canonical fixed kernels used by the Sobel operator ablation.

Each returned directional kernel is normalized independently to unit L2 norm.
This is required because the supplied 5x5 templates do not share one norm.
"""

from collections import OrderedDict

import torch


_SOBEL_3X3 = OrderedDict(
    (
        (
            "0",
            (
                (1, 2, 1),
                (0, 0, 0),
                (-1, -2, -1),
            ),
        ),
        (
            "45",
            (
                (2, 1, 0),
                (1, 0, -1),
                (0, -1, -2),
            ),
        ),
        (
            "90",
            (
                (1, 0, -1),
                (2, 0, -2),
                (1, 0, -1),
            ),
        ),
        (
            "135",
            (
                (0, -1, -2),
                (1, 0, -1),
                (2, 1, 0),
            ),
        ),
    )
)


_SOBEL_5X5 = OrderedDict(
    (
        (
            "0",
            (
                (0, 0, 0, 0, 0),
                (-1, -2, -4, -2, -1),
                (0, 0, 0, 0, 0),
                (1, 2, 4, 2, 1),
                (0, 0, 0, 0, 0),
            ),
        ),
        (
            "22.5",
            (
                (0, 0, 0, 0, 0),
                (0, -2, -4, -2, 0),
                (-1, -4, 0, 4, 1),
                (0, 2, 4, 2, 0),
                (0, 0, 0, 0, 0),
            ),
        ),
        (
            "45",
            (
                (0, 0, 0, -1, 0),
                (0, -2, -4, 0, 1),
                (0, -4, 0, 4, 0),
                (-1, 0, 4, 2, 0),
                (0, 1, 0, 0, 0),
            ),
        ),
        (
            "67.5",
            (
                (0, 0, -1, 0, 0),
                (0, -2, -4, 2, 0),
                (0, -4, 0, 4, 0),
                (0, -2, 4, 2, 0),
                (0, 0, 1, 0, 0),
            ),
        ),
        (
            "90",
            (
                (0, -1, 0, 1, 0),
                (0, -2, 0, 2, 0),
                (0, -4, 0, 4, 0),
                (0, -2, 0, 2, 0),
                (0, -1, 0, 1, 0),
            ),
        ),
        (
            "112.5",
            (
                (0, 0, 1, 0, 0),
                (0, -2, 4, 2, 0),
                (0, -4, 0, 4, 0),
                (0, -2, -4, 2, 0),
                (0, 0, -1, 0, 0),
            ),
        ),
        (
            "135",
            (
                (0, 1, 0, 0, 0),
                (-1, 0, 4, 2, 0),
                (0, -4, 0, 4, 0),
                (0, -2, -4, 0, 1),
                (0, 0, 0, -1, 0),
            ),
        ),
        (
            "157.5",
            (
                (0, 0, 0, 0, 0),
                (0, 2, 4, 2, 0),
                (-1, -4, 0, 4, 1),
                (0, -2, -4, -2, 0),
                (0, 0, 0, 0, 0),
            ),
        ),
    )
)


_VARIANTS = OrderedDict(
    (
        ("s3_d2", (3, ("0", "90"))),
        ("s3_d4", (3, ("0", "45", "90", "135"))),
        ("s5_d2", (5, ("0", "90"))),
        ("s5_d4", (5, ("0", "45", "90", "135"))),
        (
            "s5_d8",
            (
                5,
                ("0", "22.5", "45", "67.5", "90", "112.5", "135", "157.5"),
            ),
        ),
    )
)


def available_sobel_variants():
    """Return stable CLI names for the five operator comparisons."""
    return tuple(_VARIANTS.keys())


def get_variant_spec(variant):
    if variant not in _VARIANTS:
        raise ValueError(
            "Unknown Sobel variant {!r}. Available variants: {}".format(
                variant,
                ", ".join(available_sobel_variants()),
            )
        )
    kernel_size, directions = _VARIANTS[variant]
    return {
        "name": variant,
        "kernel_size": kernel_size,
        "directions": directions,
    }


def get_normalized_kernels(variant, dtype=torch.float32):
    """Return an ``N x 1 x k x k`` tensor of unit-L2 kernels."""
    spec = get_variant_spec(variant)
    kernel_bank = _SOBEL_3X3 if spec["kernel_size"] == 3 else _SOBEL_5X5
    kernels = torch.tensor(
        [kernel_bank[direction] for direction in spec["directions"]],
        dtype=dtype,
    ).unsqueeze(1)
    norms = kernels.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
    if torch.any(norms == 0):
        raise ValueError("A Sobel direction contains an all-zero kernel")
    return kernels / norms
