from pathlib import Path
from pinder.eval.dockq.method import MethodMetrics, summarize_valid_systems


"""
Example eval directory structure:
eval_example/
└── some_method
    ├── 1an1__A1_P00761--1an1__B1_P80424
    │   ├── apo_decoys
    │   │   ├── model_1.pdb
    │   │   └── model_2.pdb
    │   ├── holo_decoys
    │   │   ├── model_1.pdb
    │   │   └── model_2.pdb
    │   └── predicted_decoys
    │       ├── model_1.pdb
    │       └── model_2.pdb
    └── 1b8m__A1_P23560--1b8m__B1_P34130
        ├── holo_decoys
        │   ├── model_1.pdb
        │   └── model_2.pdb
        └── predicted_decoys
            ├── model_1.pdb
            └── model_2.pdb
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        "-f",
        help="Path to eval",
        type=str,
        metavar="eval_dir",
        required=True,
    )
    parser.add_argument(
        "--serial",
        "-s",
        help="Whether to disable parallel eval over systems",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--method_name",
        "-m",
        "-n",
        help="Optional name for output csv",
        type=str,
        metavar="method_name",
        required=False,
    )
    parser.add_argument(
        "--allow_missing",
        "-a",
        help="Whether to allow missing systems for a given pinder-set + monomer",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--custom_index",
        "-c",
        help="Optional local filepath or GCS uri to a custom index with non-pinder splits. Note: must still follow the pinder index schema and define test holdout sets, but does not need to share the same split members.",
        default="",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--max_workers",
        "-w",
        help="Optional maximum number of processes to spawn in multiprocessing. Default is None (all available cores).",
        default=None,
        required=False,
        type=int,
    )
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir).absolute()
    parallel = not args.serial

    summarize_valid_systems(eval_dir, custom_index=args.custom_index)

    mm = MethodMetrics(
        eval_dir,
        parallel=parallel,
        allow_missing_systems=args.allow_missing,
        custom_index=args.custom_index,
        max_workers=args.max_workers,
    )
    metrics = mm.metrics
    method_name = args.method_name or eval_dir.stem.split("_eval")[0]
    metric_csv = eval_dir / f"{method_name}_metrics.csv"
    metrics.to_csv(metric_csv, index=False)
    leaderboard_row = mm.get_leaderboard_entry()
    leaderboard_csv = eval_dir / f"{method_name}_leaderboard.csv"
    leaderboard_row.to_csv(leaderboard_csv, index=False)


if __name__ == "__main__":
    main()
