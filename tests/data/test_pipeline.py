import pytest
from pinder.data.pipeline.data_pipeline import DataIngestPipeline


@pytest.mark.parametrize(
    "stage, run_specific_step, skip_specific_step, expected_n_batches, expected_batch_size, expected_output_dir",
    [
        ("download_rcsb_files", "download_rcsb_files", "", 1, 1, "data/1a"),
        ("download_rcsb_files", "", "download_rcsb_files", 1, 0, None),
    ],
)
def test_data_pipeline_run_stage(
    stage,
    run_specific_step,
    skip_specific_step,
    expected_n_batches,
    expected_batch_size,
    expected_output_dir,
    pinder_data_cp,
):
    local_pinder_mount = pinder_data_cp / "pinder-data-pipeline"
    if not local_pinder_mount.is_dir():
        local_pinder_mount.mkdir(parents=True)

    pipe = DataIngestPipeline(
        image="local",
        pinder_mount_point=str(local_pinder_mount),
        skip_specific_step=skip_specific_step,
        run_specific_step=run_specific_step,
        two_char_code="1a",
        use_cache=True,
    )
    pipe.run_stage(stage)
    assert len(pipe.scatter_batches) == expected_n_batches
    assert len(pipe.scatter_batches[0]) == expected_batch_size
    if expected_output_dir:
        assert (local_pinder_mount / expected_output_dir).is_dir()


@pytest.mark.parametrize(
    "run_specific_step, skip_specific_step, expected_n_batches, expected_batch_size, expected_output_dir",
    [
        ("download_rcsb_files", "", 1, 0, "data/1a"),
        ("download_rcsb_files", "download_rcsb_files", 1, 0, None),
    ],
)
def test_data_pipeline_run(
    run_specific_step,
    skip_specific_step,
    expected_n_batches,
    expected_batch_size,
    expected_output_dir,
    pinder_data_cp,
):
    local_pinder_mount = pinder_data_cp / "pinder-data-pipeline"
    if not local_pinder_mount.is_dir():
        local_pinder_mount.mkdir(parents=True)

    pipe = DataIngestPipeline(
        image="local",
        pinder_mount_point=str(local_pinder_mount),
        skip_specific_step=skip_specific_step,
        run_specific_step=run_specific_step,
        two_char_code="1a",
        use_cache=True,
    )
    pipe.run()
    if expected_output_dir:
        assert (local_pinder_mount / expected_output_dir).is_dir()
    assert len(pipe.scatter_batches) == expected_n_batches
    assert len(pipe.scatter_batches[0]) == expected_batch_size
