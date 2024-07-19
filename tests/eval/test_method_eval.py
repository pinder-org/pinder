import pandas as pd
import pytest

from pinder.core import get_index
from pinder.core.index.utils import IndexEntry
from pinder.eval.dockq.method import (
    download_entry,
    get_valid_eval_systems,
    summarize_valid_systems,
    validate_system_id,
    MethodMetrics,
)

pindex = get_index()


def test_validate_system_id(pinder_temp_dir):
    with pytest.raises(ValueError) as exc_info:
        validate_system_id(pinder_temp_dir, set())
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == f"Invalid directory structure! {pinder_temp_dir.stem} is not a pinder ID"
    )


# Marking skipif for now until we set the pinder_xl/s boolean in index
@pytest.mark.skipif(
    pindex.pinder_xl.sum() == 0,
    reason="not possible to get method metrics without pinder_xl/s/af2 test sets defined!",
)
def test_method_metrics(pinder_eval_dir):
    mm = MethodMetrics(pinder_eval_dir, allow_missing_systems=True, parallel=False)
    metrics = mm.metrics
    assert isinstance(metrics, pd.DataFrame)
    assert metrics.shape == (8, 28)
    oracle = mm.system_oracle()
    assert oracle.shape == (4, 10)
    assert mm.top_k(1).shape == (4, 28)
    assert mm.median_capri_hit_rate() == 0.25
    oracle_summary = mm.oracle_median_summary()
    assert oracle_summary.shape == (4, 9)
    leaderboard_row = mm.get_leaderboard_entry()
    assert leaderboard_row.shape == (4, 41)


def test_summarize_valid_systems(monkeypatch, pinder_eval_dir):
    # iterator for input mock values
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda x: next(answers))
    with pytest.raises(RuntimeError) as exc_info:
        summarize_valid_systems(pinder_eval_dir)
    assert exc_info.type is RuntimeError
    assert exc_info.value.args[0] == "Method rejected! Missing systems!"


def test_eval_format_validation(pinder_eval_dir, pinder_method_test_dir):
    expected1 = [
        {
            "system": "1ldt__A1_P00761--1ldt__B1_P80424",
            "method_name": "some_method",
        },
        {
            "system": "1b8m__B1_P34130--1b8m__A1_P23560",
            "method_name": "some_method",
        },
    ]
    expected2 = [
        {
            "system": "5o2z__A1_P06396--5o2z__B1_P06396",
            "method_name": "diffdock-pp",
        },
        {
            "system": "6s0a__A1_P27918--6s0a__B1_P27918",
            "method_name": "dockgpt",
        },
        {
            "system": "6x1g__A1_B3CVM3--6x1g__C1_P63000",
            "method_name": "hdock",
        },
        {
            "system": "2e31__A1_Q80UW2--2e31__B1_P63208",
            "method_name": "geodock",
        },
    ]
    eval_systems1 = get_valid_eval_systems(pinder_eval_dir)
    assert isinstance(eval_systems1, list)
    assert len(eval_systems1) == 2
    for eval_system in eval_systems1:
        eval_system["system"] = eval_system["system"].stem
        assert eval_system == expected1[0] or eval_system == expected1[1]

    eval_systems2 = get_valid_eval_systems(pinder_method_test_dir)
    assert isinstance(eval_systems2, list)
    assert len(eval_systems2) == 4
    for eval_system in eval_systems2:
        eval_system["system"] = eval_system["system"].stem
        assert eval_system in expected2


def test_download_entry():
    entry_id = "1zop__A1_P20701--1zop__B1_P20701"
    row = pindex.query(f'id == "{entry_id}"')
    entry = IndexEntry(**row.to_dict(orient="records")[0])
    assert download_entry(entry) is None
