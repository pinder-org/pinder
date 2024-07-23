import pytest
from pinder.methods.preprocess import prepare_inference_inputs


@pytest.mark.parametrize(
    "ids, pairs, expected_num_configs",
    [
        (["7zj1__B1_P55265--7zj1__A1_P55265"], "all", 3),  # all three monomer types
        (
            ["7uln__A1_P09514--7uln__B1_P09514", "7dnu__A1_P32092--7dnu__A2_P32092"],
            "all",
            2,
        ),  # only holo
    ],
)
def test_prepare_inference_inputs(tmp_path, ids, pairs, expected_num_configs):
    inference_config = prepare_inference_inputs(
        method_dir=tmp_path, ids=ids, pairs=pairs
    )
    assert len(inference_config) == expected_num_configs
