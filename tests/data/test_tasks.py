import pytest
import shutil
from unittest.mock import patch


@patch("pinder.data.rcsb_rsync.get_rsync_directories")
def test_two_char_code_rsync(mock_get_rsync_directories, pinder_data_cp):
    mock_get_rsync_directories.return_value = ["foo", "bar"]
    from pinder.data.pipeline import tasks

    data_dir = pinder_data_cp / "nextgen_rcsb"
    assert set(tasks.two_char_code_rsync(data_dir)) == {"foo", "bar"}


def test_foldseek_pdb_glob(pinder_data_cp):
    from pinder.data.pipeline import tasks

    pdb_dir = pinder_data_cp / "pinder/pdbs"
    assert len(tasks.foldseek_pdb_glob(pdb_dir)) == 30


def test_plm_embedding_pairs(tmp_path, pinder_data_cp):
    from pinder.data.pipeline import tasks

    assert len(tasks.plm_embedding_pairs(pinder_data_cp)) == 0
    embed_files = tasks.plm_embedding_glob(pinder_data_cp)
    assert len(embed_files) == 1
    npz = embed_files[0]
    shutil.copy(npz, tmp_path / npz.name)
    shutil.copy(npz, tmp_path / "embeddings_1.npz")
    assert len(tasks.plm_embedding_pairs(tmp_path)) == 1


@pytest.mark.parametrize(
    "step_name, input_type, data_subdir, cache_func, cache_kwargs, scatter_func, batch_size, expected_batches",
    [
        (
            "ingest_rcsb_files",
            "cif",
            "nextgen_rcsb",
            "get_uningested_mmcif",
            {},
            "chunk_collection",
            2,
            4,
        ),
        (
            "ingest_rcsb_files",
            "cif",
            "nextgen_rcsb_typo",
            "get_uningested_mmcif",
            {},
            "chunk_collection",
            2,
            1,
        ),
        (
            "get_pisa_annotations",
            "cif",
            "nextgen_rcsb",
            "get_pisa_unannotated",
            {},
            "chunk_collection",
            2,
            6,
        ),
        (
            "get_rcsb_annotations",
            "pdb_ids",
            "nextgen_rcsb",
            "get_rcsb_unannotated",
            {"pinder_dir": ""},
            "chunk_collection",
            1,
            13,
        ),
    ],
)
def test_get_stage_tasks(
    step_name,
    input_type,
    data_subdir,
    cache_func,
    cache_kwargs,
    scatter_func,
    batch_size,
    expected_batches,
    pinder_data_cp,
):
    from pinder.data.pipeline import cache, tasks, scatter

    data_dir = pinder_data_cp / data_subdir
    if "pinder_dir" in cache_kwargs:
        cache_kwargs["pinder_dir"] = pinder_data_cp / "pinder"

    scatter_batches = tasks.get_stage_tasks(
        data_dir,
        input_type=input_type,
        batch_size=batch_size,
        cache_func=getattr(cache, cache_func),
        cache_kwargs=cache_kwargs,
        scatter_method=getattr(scatter, scatter_func),
    )
    assert len(scatter_batches) == expected_batches


def test_run_task(pinder_data_cp):
    from pinder.data.pipeline import tasks
    from pinder.data import rcsb_rsync

    tasks.run_task(
        rcsb_rsync.download_two_char_codes,
        task_input=dict(codes=["1a"], data_dir=pinder_data_cp, redirect_stdout=True),
        iterable_kwarg="codes",
    )
    assert (pinder_data_cp / "1a").is_dir()
