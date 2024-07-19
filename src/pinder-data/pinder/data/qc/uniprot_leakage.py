from __future__ import annotations

import pandas as pd
from pinder.data.qc.annotation_check import get_paired_uniprot_intersection
from pinder.data.qc.utils import load_index


def uniprot_leakage_main(
    index_path: str | None = None,
    split: str = "test",
) -> pd.DataFrame:
    pindex = load_index(index_path)
    all_splits = ["train", "test", "val"]
    pindex_all_splits = pindex[
        (pindex["split"].isin(all_splits))
        & ~(pindex["cluster_id"].str.contains("-1", regex=False))
        & ~(pindex["cluster_id"].str.contains("p_p", regex=False))
    ]

    uniprot_problems = get_paired_uniprot_intersection(pindex_all_splits, against=split)
    return uniprot_problems


def report_uniprot_test_val_leakage(index_path: str | None = None) -> pd.DataFrame:
    _, percent_test = uniprot_leakage_main(index_path, split="test")
    _, percent_val = uniprot_leakage_main(index_path, split="val")
    report = pd.DataFrame(
        [
            {
                "Measure": "Leakage",
                "Metric": "Uniprot pair",
                "Test": percent_test * 100,
                "Val": percent_val * 100,
            },
        ]
    )
    return report
