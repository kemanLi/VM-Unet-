"""Promote one screened operator into a single downstream configuration."""

import argparse
import json
from pathlib import Path

from variants import OPERATOR_NAMES


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=OPERATOR_NAMES, required=True)
    parser.add_argument(
        "--q-stats-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "q_stats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sobel_selected.json",
    )
    args = parser.parse_args()

    selected = {
        "operator": args.operator,
        "initialization": "scratch",
        "guidance_strength": 1.0,
        "seed": 42,
        "q_by_dataset": {},
    }
    for dataset in ("DRIVE", "STARE"):
        source = args.q_stats_dir / "{}.json".format(dataset)
        statistics = json.loads(source.read_text(encoding="utf-8"))
        selected["q_by_dataset"][dataset] = float(
            statistics["variants"][args.operator]["q"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Selected {} -> {}".format(args.operator, args.output))


if __name__ == "__main__":
    main()
