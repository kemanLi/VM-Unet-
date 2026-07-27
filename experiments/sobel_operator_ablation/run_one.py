"""Launch one locked Sobel-operator screening run."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from variants import OPERATOR_NAMES  # noqa: E402


def _load_q(q_stats_path, dataset, operator):
    if not q_stats_path.is_file():
        raise FileNotFoundError(
            "Missing {}. Run compute_q_stats.py before training.".format(
                q_stats_path
            )
        )
    statistics = json.loads(q_stats_path.read_text(encoding="utf-8"))
    if statistics.get("dataset") != dataset:
        raise ValueError("q statistics dataset does not match {}".format(dataset))
    return float(statistics["variants"][operator]["q"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("DRIVE", "STARE"), required=True)
    parser.add_argument("--operator", choices=OPERATOR_NAMES, required=True)
    parser.add_argument(
        "--q-stats-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "q_stats",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "sobel_operator_ablation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    q_value = _load_q(
        args.q_stats_dir / "{}.json".format(args.dataset),
        args.dataset,
        args.operator,
    )
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "train.py"),
        "--dataset",
        args.dataset,
        "--model",
        "vmunet_highres_sobel",
        "--initialization",
        "scratch",
        "--sobel-operator",
        args.operator,
        "--sobel-q",
        repr(q_value),
        "--guidance-strength",
        "1.0",
        "--seed",
        "42",
        "--run-tag",
        "operator_{}".format(args.operator),
        "--result-root",
        str(args.result_root),
    ]
    print("Locked settings: scratch, lambda=1.0, seed=42")
    print("Command: {}".format(subprocess.list2cmdline(command)))
    if not args.dry_run:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
