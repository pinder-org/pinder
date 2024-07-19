import pytest
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

import biotite.structure.io as strucio
import biotite.sequence.io.fasta as fasta
from biotite.sequence import ProteinSequence
import numpy as np
import pandas as pd

from pinder.core import PinderSystem
from pinder.data.config import FoldseekConfig, MMSeqsConfig
from pinder.data.foldseek_utils import (
    alignment_to_parquet,
    create_dbs,
    create_dbs_and_run,
    create_fasta_from_foldseek_inputs,
    create_fasta_from_systems,
    create_foldseek_input_dir,
    fasta2dict,
    run_mmseqs,
    run_foldseek,
    run_db_vs_db,
    run_foldseek_on_pinder_chains,
    run_mmseqs_on_pinder_chains,
)


def test_fasta2dict(pinder_data_cp):
    pdb_dir = pinder_data_cp / "pinder/pdbs"
    foldseek_fasta = pinder_data_cp / "foldseek.fasta"
    create_fasta_from_foldseek_inputs(
        foldseek_dir=pdb_dir,
        fasta_file=foldseek_fasta,
        parallel=True,
        max_workers=2,
    )
    fasta_dict = fasta2dict(foldseek_fasta)
    assert isinstance(fasta_dict, dict)
    assert len(fasta_dict.keys()) == 30
    headers = {p.stem for p in pdb_dir.glob("*.pdb")}
    assert set(fasta_dict.keys()) == headers


def test_get_dev_systems(pinder_data_cp):
    from pinder.data.system import get_dev_systems
    from pinder.core.index.system import PinderSystem

    pindex_fname = pinder_data_cp / "pinder/index.1.csv.gz"
    metadata_fname = pinder_data_cp / "pinder/metadata.1.csv.gz"

    systems = list(
        get_dev_systems(
            pindex_fname, metadata_fname, dataset_path=pinder_data_cp / "pinder"
        )
    )
    assert len(systems) > 1
    assert isinstance(systems[0], PinderSystem)


def test_create_fasta_from_systems(pinder_data_cp):
    """Test the create_fasta_from_systems function"""

    pids = ["5jlh__F1_Q7Z406--5jlh__C1_P63261", "7uxc__H1_Q9UHA4--7uxc__F1_Q6IAA8"]
    systems = [PinderSystem(pid) for pid in pids]
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        fasta_file = tmp_dir_path / "output.fasta"
        create_fasta_from_systems(systems, fasta_file)
        assert fasta_file.is_file()
        output_fasta = fasta.FastaFile.read(fasta_file)

        # Assuming your test file is located directly in the tests/ directory
        # Adjust the path to test_data/expected_output.fasta as necessary
        expected_fasta_path = pinder_data_cp / "foldseek/fasta_from_systems.fasta"
        expected_fasta = fasta.FastaFile.read(expected_fasta_path)

        for expected_header, expected_sequence in expected_fasta.items():
            assert (
                expected_header in output_fasta
            ), f"Expected header {expected_header} not found in output"
            assert expected_sequence == output_fasta[expected_header], (
                f"Output {expected_header} sequence {output_fasta[expected_header]}"
                + f"not equal to expected sequence {expected_sequence}"
            )


def test_create_fasta_from_dev_systems(pinder_data_cp):
    from pinder.data.system import get_dev_systems
    from pinder.data.foldseek_utils import create_fasta_from_systems

    pindex_fname = pinder_data_cp / "pinder/index.1.csv.gz"
    metadata_fname = pinder_data_cp / "pinder/metadata.1.csv.gz"

    systems = list(
        get_dev_systems(pindex_fname, metadata_fname, pinder_data_cp / "pinder")
    )
    output_dir = pinder_data_cp / "fasta_folder"
    output_dir.mkdir(exist_ok=True)

    fasta_file = output_dir / "fasta_from_systems.fasta"
    create_fasta_from_systems(systems, fasta_file)
    assert fasta_file.is_file()


@pytest.mark.parametrize(
    "db_size, expected_sub_dirs",
    [
        (1, {"00000", "00001", "00002", "00003", "00004"}),
        (2, {"00000", "00001", "00002"}),
    ],
)
def test_create_dbs(pinder_data_cp, db_size, expected_sub_dirs):
    """
    Test the create_dbs function
    """
    mock_foldseek_db_path = pinder_data_cp / "foldseek/mock_foldseek_db"
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        # create the databases
        create_dbs(
            tmp_dir_path,
            mock_foldseek_db_path,
            db_size=db_size,
            parallel=True,
            max_workers=2,
        )

        pdb_paths = list(tmp_dir_path.rglob("*.pdb"))
        for pdb_path in pdb_paths:
            sub_dir = pdb_path.relative_to(pdb_path.parents[2]).parents[1].as_posix()
            assert (
                sub_dir in expected_sub_dirs
            ), f"Unexpected sub dir {sub_dir} in not in sub dirs, for db_size={db_size}"
            db_pdbs = list(pdb_path.parent.glob("*.pdb"))
            assert len(db_pdbs) <= db_size


def test_create_fasta_from_foldseek_inputs(pinder_data_cp):
    """
    Test the create_fasta_from_foldseek_inputs function

    Tests using pdbs with missing residues(aug_mock_foldseek_db), show that
    create_fasta_from_foldseek_inputs creates a fasta file without placeholders
    not sure if expected, or the implications for downstream analyses
    """

    # Convert three-letter codes to one-letter codes
    three_to_one = ProteinSequence.convert_letter_3to1

    mock_foldseek_db_path = pinder_data_cp / "foldseek/mock_foldseek_db"

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        # create the foldseek fasta
        foldseek_fasta = tmp_dir_path / "foldseek.fasta"
        create_fasta_from_foldseek_inputs(
            mock_foldseek_db_path, foldseek_fasta, parallel=True, max_workers=2
        )
        # read in the generated fasta file
        assert foldseek_fasta.is_file()
        foldseek_fasta = fasta.FastaFile.read(foldseek_fasta)

        # get the expected output
        for pdb_path in mock_foldseek_db_path.glob("*.pdb"):
            header = pdb_path.stem
            atom_array = strucio.load_structure(pdb_path)
            res_id_names = {res.res_id: res.res_name for res in atom_array}
            sequence = "".join(
                [three_to_one(res_name) for _, res_name in sorted(res_id_names.items())]
            )

            assert (
                header in foldseek_fasta
            ), f"Expected header {header} not found in output"
            assert (
                sequence == foldseek_fasta[header]
            ), f"Expected sequence {sequence} not equal to expected sequence {foldseek_fasta[header]}"


def test_run_mmseqs(pinder_data_cp):
    """
    Test the run_mmseqs function
    """

    def get_mmseqs_aln_dict(file_path):
        with open(file_path, "r", encoding="utf-8") as result_file:
            output = result_file.read()

        # create query and target dict
        query_target_dict = dict()
        for line in output.split("\n"):
            if line == "":
                continue
            line_split = line.split("\t")
            # dict[(Query, Target)] = Identity
            query, target, pident = line_split[:3]
            query_target_dict[(query, target)] = float(pident)

        return query_target_dict

    # Set up the input and output paths
    input_fasta = pinder_data_cp / "foldseek/mock_foldseek_db/mock_foldseek_db.fasta"
    expected_alignment = (
        pinder_data_cp / "foldseek/mock_foldseek_db/mmseqs_test_aln.txt"
    )
    expected_scores = get_mmseqs_aln_dict(expected_alignment)

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        # call the pinder function
        run_mmseqs(input_fasta, tmp_dir_path)
        alignment_path = tmp_dir_path / "mmseqs2/alignment.txt"
        assert alignment_path.is_file(), "Actual alignment output file does not exist"

        actual_scores = get_mmseqs_aln_dict(alignment_path)

        assert set(actual_scores.keys()) == set(
            expected_scores.keys()
        ), f"Actual score dict keys do not match expected score dict keys"
        for pdb_pair, score in expected_scores.items():
            assert (
                actual_scores[pdb_pair] == score
            ), f"Expected score {score} not equal to actual score {actual_scores[pdb_pair]}"


@pytest.mark.parametrize(
    "sensitivity, evalue, alignment_type, ref_file",
    [
        (11.0, 0.05, 2, "foldseek_alignment_2_s_11_e_p05.txt"),
        (11.0, 0.05, 1, "foldseek_alignment_1_s_11_e_p05.txt"),
        (11.0, 0.5, 2, "foldseek_alignment_2_s_11_e_p5.txt"),
        (11.0, 0.5, 1, "foldseek_alignment_1_s_11_e_p5.txt"),
        (6.0, 0.05, 2, "foldseek_alignment_2_s_6_e_p05.txt"),
        (6.0, 0.05, 1, "foldseek_alignment_1_s_6_e_p05.txt"),
        (6.0, 0.5, 2, "foldseek_alignment_2_s_6_e_p5.txt"),
        (6.0, 0.5, 1, "foldseek_alignment_1_s_6_e_p5.txt"),
    ],
)
def test_run_foldseek(pinder_data_cp, sensitivity, evalue, alignment_type, ref_file):
    """
    Test the run_foldseek function
    """

    def get_foldseek_aln_dict(file_path):
        with open(file_path, "r", encoding="utf-8") as result_file:
            output = result_file.read()

        # create query and target dict
        query_target_dict = dict()
        for line in output.split("\n"):
            if line == "":
                continue
            line_split = line.split("\t")
            # dict[(Query, Target)] = Identity
            query, target, lddt = line_split[:3]
            # Newer foldseek version removes .pdb from filename
            query = query.split(".pdb")[0]
            target = target.split(".pdb")[0]
            query_target_dict[(query, target)] = {
                "lddt": float(lddt),
            }

        return query_target_dict

    # the mock foldseek db path
    mock_foldseek_db_path = pinder_data_cp / "foldseek/mock_foldseek_db"
    ref_alignment_path = pinder_data_cp / f"foldseek/foldseek_aln/{ref_file}"

    # read in the expected output as a dict
    expected_scores = get_foldseek_aln_dict(ref_alignment_path)

    # set the foldseek config
    config = FoldseekConfig()
    config.sensitivity = sensitivity
    config.evalue = evalue
    config.alignment_type = alignment_type

    # run pinder foldseek
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        output_dir = tmp_dir_path / "foldseek_output"
        run_foldseek(
            mock_foldseek_db_path, output_dir, mock_foldseek_db_path, config=config
        )

        # Now, check if the output file exists
        alignment_path = output_dir / "alignment.txt"
        assert alignment_path.is_file(), "Expected output file does not exist"

        actual_scores = get_foldseek_aln_dict(alignment_path)

        assert set(actual_scores.keys()) == set(
            expected_scores.keys()
        ), "Actual score dict keys do not match expected score dict keys"

        for pdb_pair, expected_score in expected_scores.items():
            # approx(expected_score, rel=0.05), \
            assert (
                actual_scores[pdb_pair] == expected_score
            ), f"Expected score {expected_score} not equal to actual score {actual_scores[pdb_pair]}"


def test_run_db_vs_db(pinder_data_cp):
    """
    Testing the connection between run_db_vs_db and run_foldseek explicitly
    is how I discovered a bug in run_db_vs_db, so including this test as
    well as the mock test.
    """

    mock_foldseek_db_path = pinder_data_cp / "foldseek/mock_foldseek_db"

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        # create the databases
        tmp_foldseek_db_path = tmp_dir_path / "foldseek_db"
        create_dbs(
            tmp_foldseek_db_path,
            mock_foldseek_db_path,
            db_size=1,
            parallel=True,
            max_workers=2,
        )

        # Call the function with the mock_db_path
        i, j = 0, 4  # we have 5 total pdb files in the mock_foldseek_db
        result_path = run_db_vs_db(tmp_foldseek_db_path, i, j)

        # Now, check if the expected output file exists
        assert result_path.is_file(), "Expected output file does not exist"


# Assuming your logging is setup like this, adjust if necessary
@patch("pinder.data.foldseek_utils.log")
def test_run_db_vs_db_mock(_mock_log, tmp_path):
    """
    Mock version of test_run_db_vs_db function, which tests run_db_vs_db
    without actually running run_foldseek
    """
    with patch("pinder.data.foldseek_utils.run_foldseek") as mock_run_foldseek:
        db_path = tmp_path
        i, j = 1, 2
        config = FoldseekConfig()
        use_cache = True

        # Setup: make it look like the alignment file already exists (to test caching behavior)
        output_dir = db_path / f"{i:05d}" / f"{j:05d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        alignments = output_dir / "alignment.txt"
        alignments.touch()  # Create the file

        # Test when use_cache is True and file exists
        result = run_db_vs_db(db_path, i, j, config, use_cache)
        mock_run_foldseek.assert_not_called()
        assert result == alignments

        # Test when use_cache is False or file doesn't exist
        alignments.unlink()  # Remove the file to simulate it doesn't exist
        use_cache = False  # Change use_cache to False
        result = run_db_vs_db(db_path, i, j, config, use_cache)
        mock_run_foldseek.assert_called_once_with(
            db_path / f"{i:05d}" / "db",
            output_dir,
            db_path / f"{j:05d}" / "db",
            config=config,
        )


def test_create_foldseek_input_dir(pinder_data_cp):
    """
    Test the create_foldseek_input_dir function

    The follow keyword arguments are not used in the create_foldseek_input_dir function.
    It's possible there are plans to use these in the future, including here for reference:
        dataset_path: Path | None = None,
        parallel: bool = False,
    """
    pindex_fname = pinder_data_cp / "pinder/index.1.csv.gz"
    pdb_dir = pinder_data_cp / "pinder/pdbs"
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        foldseek_dir = tmp_dir_path / "foldseek_dir"
        create_foldseek_input_dir(
            index=pindex_fname,
            foldseek_dir=foldseek_dir,
            pdb_dir=pdb_dir,
            parallel=False,
        )
        assert foldseek_dir.is_dir()
        expected_pdbs = [
            "7nsg_A.pdb",
            "7nsg_C.pdb",
            "7nsg_B.pdb",
            "4wwi_A.pdb",
            "4wwi_D.pdb",
            "6wwe_B.pdb",
            "6wwe_C.pdb",
            "6wwe_A.pdb",
            "7cma_A.pdb",
            "7cma_B.pdb",
        ]
        created_pdbs = [p.name for p in foldseek_dir.glob("*.pdb")]
        for p in expected_pdbs:
            assert (
                foldseek_dir / p
            ).is_file(), f"Expected pdb {p} not found in foldseek dir: {created_pdbs}"


@pytest.mark.parametrize("db_size, num_chains", [(1, 4), (2, 4), (1, 5), (2, 5)])
def test_create_dbs_and_run(tmp_path, db_size, num_chains):
    # Setup paths
    fold_db_path = tmp_path / "fold_db"
    chains_path = tmp_path / "chains"
    chains_path.mkdir()
    fold_db_path.mkdir()

    num_db_vs_db_calls = 0
    for i in range(0, num_chains, db_size):
        for j in range(0, num_chains, db_size):
            if j < i:
                continue
            num_db_vs_db_calls += 1

    for i in range(num_chains):
        (chains_path / f"chain_{i}.pdb").touch()

    config = FoldseekConfig()

    # Function to create dummy alignment files
    def mock_run_db_vs_db(fold_db_path, i, j, config):
        alignment_file = fold_db_path / f"{i}_{j}_alignment.txt"
        alignment_file.write_text("alignment data\n")
        return alignment_file

    with patch("pinder.data.foldseek_utils.create_dbs") as mock_create_dbs, patch(
        "pinder.data.foldseek_utils.run_db_vs_db", side_effect=mock_run_db_vs_db
    ):
        # Run the function under test
        create_dbs_and_run(fold_db_path, chains_path, db_size, config)

        # Verify `create_dbs` was called correctly
        mock_create_dbs.assert_called_once_with(fold_db_path, chains_path, db_size)

        # Check the final alignment.txt content
        final_alignment_path = fold_db_path / "alignment.txt"
        assert final_alignment_path.exists()
        with final_alignment_path.open("r") as f:
            final_content = f.read()
            assert (
                final_content.count("alignment data") == num_db_vs_db_calls
            )  # or another logic based on db_size and n_chains

    # Verify temporary alignment files were cleaned up
    alignment_files = list(fold_db_path.glob("*_alignment.txt"))
    assert not alignment_files, "Temporary alignment files were not cleaned up."


@pytest.mark.parametrize("pdb_dir_exists", [True, False])
def test_run_foldseek_on_pinder_chains(pdb_dir_exists):
    pdb_dir = Path("/fake/pdb_dir")
    foldseek_dir = Path("/tmp/foldseek")
    index = "index.1.csv.gz"

    mock_config = FoldseekConfig()

    # Setup the mocks
    with patch.object(
        Path, "is_dir", return_value=pdb_dir_exists
    ) as mock_is_dir, patch(
        "pinder.data.foldseek_utils.create_foldseek_input_dir"
    ) as mock_create_foldseek_input_dir, patch(
        "pinder.data.foldseek_utils.create_dbs_and_run"
    ) as mock_create_dbs_and_run, patch(
        "pinder.data.foldseek_utils.log.error"
    ) as mock_log_error:
        # Call the function under test
        run_foldseek_on_pinder_chains(pdb_dir, index, foldseek_dir, config=mock_config)

        # Check if the directory existence check was performed
        mock_is_dir.assert_called_once()

        if pdb_dir_exists:
            # Verify create_foldseek_input_dir was called with correct arguments
            mock_create_foldseek_input_dir.assert_called_once_with(
                index, foldseek_dir, pdb_dir=pdb_dir
            )

            # Verify create_dbs_and_run was called with correct arguments
            mock_create_dbs_and_run.assert_called_once_with(
                foldseek_dir / "foldseek_dbs", foldseek_dir, config=mock_config
            )

            # Ensure the error log was not called
            mock_log_error.assert_not_called()
        else:
            # Ensure that the function exits gracefully if the pdb_dir doesn't exist
            mock_log_error.assert_called_once_with(
                f"Input PDB directory {pdb_dir} does not exist."
            )
            mock_create_foldseek_input_dir.assert_not_called()
            mock_create_dbs_and_run.assert_not_called()


@pytest.mark.parametrize(
    "pdb_dir_exists,output_dir_exists,use_cache",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
    ],
)
def test_run_mmseqs_on_pinder_chains(
    tmp_path, pdb_dir_exists, output_dir_exists, use_cache
):
    pdb_dir = tmp_path / "pdb_dir"
    if pdb_dir_exists:
        pdb_dir.mkdir()
    output_dir = tmp_path / "output_dir"
    if output_dir_exists:
        output_dir.mkdir()
    index = "index.1.csv.gz"
    fasta_file = output_dir / "input.fasta"

    # Setup the mocks
    with patch("pinder.data.foldseek_utils.log.error") as mock_log_error, patch(
        "pinder.data.foldseek_utils.log.warning"
    ) as mock_log_warning, patch(
        "pinder.data.foldseek_utils.log.info"
    ) as mock_log_info, patch(
        "pinder.data.foldseek_utils.create_foldseek_input_dir"
    ) as mock_create_foldseek_input_dir, patch(
        "pinder.data.foldseek_utils.create_fasta_from_foldseek_inputs"
    ) as mock_create_fasta, patch(
        "pinder.data.foldseek_utils.run_mmseqs"
    ) as mock_run_mmseqs:
        # Call the function under test with use_cache parameter
        run_mmseqs_on_pinder_chains(pdb_dir, index, output_dir, use_cache=use_cache)

        if not pdb_dir_exists:
            # If pdb_dir doesn't exist, verify logging error and exit
            mock_log_error.assert_called_once_with(
                f"Input directory {pdb_dir} does not exist."
            )
            mock_create_foldseek_input_dir.assert_not_called()
            mock_create_fasta.assert_not_called()
            mock_run_mmseqs.assert_not_called()
            return

        if not output_dir_exists:
            # If output_dir doesn't exist, verify warning and call to create_foldseek_input_dir
            mock_log_warning.assert_called_once()
            mock_create_foldseek_input_dir.assert_called_once_with(
                index,
                output_dir,
                pdb_dir=pdb_dir,
                use_cache=use_cache,
            )
        else:
            # Verify create_foldseek_input_dir is not called when output_dir exists
            mock_create_foldseek_input_dir.assert_not_called()

        # Verify create_fasta_from_foldseek_inputs and run_mmseqs are always called with use_cache
        mock_create_fasta.assert_called_once_with(
            output_dir, fasta_file, use_cache=use_cache
        )
        mock_run_mmseqs.assert_called_once_with(
            fasta_file, output_dir, use_cache=use_cache, config=MMSeqsConfig()
        )
        mock_log_info.assert_called_with(f"Created {fasta_file} for mmseqs2.")


def test_alignment_to_parquet(pinder_data_cp):
    """
    Test the alignment_to_parquet function
    """
    alignment_file = pinder_data_cp / "foldseek/foldseek_dbs/alignment.txt"
    alignment_to_parquet(
        alignment_file=alignment_file,
        alignment_type="foldseek",
        remove_original=False,
        use_cache=True,
    )
    filtered_pqt = alignment_file.parent / "filtered_alignment.parquet"
    aln_pqt = alignment_file.parent / f"{alignment_file.stem}.parquet"
    assert aln_pqt.is_file()
    assert filtered_pqt.is_file()
    aln = pd.read_parquet(aln_pqt)
    filtered_aln = pd.read_parquet(filtered_pqt)
    assert aln.shape == (10, 10)
    assert filtered_aln.shape == (4, 10)
    float_cols = ["lddt"]
    int_cols = ["qstart", "qend", "qlen", "tstart", "tend", "tlen", "alnlen"]
    for col in float_cols:
        assert aln.dtypes[col] == np.dtype("float64")
    for col in int_cols:
        assert aln.dtypes[col] == np.dtype("int64")
    assert set(filtered_aln["query"]) == {
        "7nsg_B.pdb",
        "6wwe_A.pdb",
        "7nsg_A.pdb",
        "6wwe_B.pdb",
    }
