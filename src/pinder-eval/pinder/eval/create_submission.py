from __future__ import annotations
import shutil
import sys
from pathlib import Path
from pinder.eval.dockq.method import summarize_valid_systems


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


def main(argv: list[str] | None = None) -> None:
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
        "--submission_name",
        "-s",
        "-n",
        help="Optional name for submission",
        type=str,
        metavar="submission_name",
        required=False,
    )
    args = parser.parse_args(sys.argv if argv is None else argv)
    eval_dir = Path(args.eval_dir).absolute()

    summarize_valid_systems(eval_dir, argv is None)
    zip_name = args.submission_name or eval_dir.stem
    base_name = eval_dir.parent / zip_name
    eval_zip = Path(
        shutil.make_archive(str(base_name), "zip", eval_dir.parent, eval_dir.stem)
    )

    print(f"Created submission archive at: {eval_zip}")


if __name__ == "__main__":
    main()
