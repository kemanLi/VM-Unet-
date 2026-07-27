"""Locked experiment matrix for selecting one Sobel operator."""


EXPERIMENTS = (
    {
        "operator": "s3_d2",
        "kernel_size": 3,
        "directions": ("0", "90"),
    },
    {
        "operator": "s3_d4",
        "kernel_size": 3,
        "directions": ("0", "45", "90", "135"),
    },
    {
        "operator": "s5_d2",
        "kernel_size": 5,
        "directions": ("0", "90"),
    },
    {
        "operator": "s5_d4",
        "kernel_size": 5,
        "directions": ("0", "45", "90", "135"),
    },
    {
        "operator": "s5_d8",
        "kernel_size": 5,
        "directions": (
            "0",
            "22.5",
            "45",
            "67.5",
            "90",
            "112.5",
            "135",
            "157.5",
        ),
    },
)

OPERATOR_NAMES = tuple(item["operator"] for item in EXPERIMENTS)
