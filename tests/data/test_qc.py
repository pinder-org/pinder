from __future__ import annotations

from pathlib import Path
import random
from itertools import product
from unittest.mock import MagicMock
import string

from biotite.sequence import ProteinSequence
import pytest
import pandas as pd

import pinder.data.qc.similarity_check as sc
from pinder.data.qc.similarity_check import (
    align_sequences,
    generate_alignments,
)
from pinder.core.index.utils import IndexEntry
import pinder.data.qc.annotation_check as ac
import pinder.data.qc.pfam_diversity as pfam
import pinder.data.qc.uniprot_leakage as ul


random.seed(42)


@pytest.mark.parametrize(
    "a, b, expect_leak",
    [
        # Expect this to match exactly
        (
            ProteinSequence("AQWERTYIPLSDFHKLCVNM" * 3),
            ProteinSequence("AQWERTYIPLSDFHKLCVNM" * 3),
            True,
        ),
        # Too short, but matches exactly
        (
            ProteinSequence("AQWERTYIPLSDFHKLCVNM"),
            ProteinSequence("AQWERTYIPLSDFHKLCVNM"),
            False,
        ),
        # Random sequences should not match
        (
            ProteinSequence(
                random.choices(ProteinSequence.alphabet.get_symbols(), k=50)
            ),
            ProteinSequence(
                random.choices(ProteinSequence.alphabet.get_symbols(), k=50)
            ),
            False,
        ),
        # Too short, and random
        (
            ProteinSequence(
                random.choices(ProteinSequence.alphabet.get_symbols(), k=30)
            ),
            ProteinSequence(
                random.choices(ProteinSequence.alphabet.get_symbols(), k=30)
            ),
            False,
        ),
    ],
)
def test_align_sequences(a: ProteinSequence, b: ProteinSequence, expect_leak):
    _, _, normed_score, leak = align_sequences(
        (("ABCD", "A"), ("WXYZ", "B"), a, b),
        Path("/dev/null"),
    )
    assert leak == expect_leak

    if leak:
        assert normed_score > 0.3 and len(a) > 40 and len(b) > 40
    else:
        assert normed_score < 0.3 or len(a) < 40 or len(b) < 40


@pytest.mark.parametrize("num_leaky", [0, 1, 2, 4, 8, 16, 20])
def test_generate_alignments(num_leaky, monkeypatch):
    """Calls to `align_sequences` and `write_alignment`, so the only thing this function
    does is:
        1. Generate the product of the test_vs_train
        2. Schedule the multiprocessing (We don't test this here)
        3. Filter the output by whether they are leaked
    """
    # Define inputs and whether an input pair is a leak
    seqs_a = {
        f"{random.choices(string.ascii_letters, k=4)}_{random.choice(string.ascii_letters)}": ProteinSequence(
            random.choices(ProteinSequence.alphabet.get_symbols(), k=5)
        )
        for _ in range(4)
    }
    seqs_b = {
        f"{random.choices(string.ascii_letters, k=4)}_{random.choice(string.ascii_letters)}": ProteinSequence(
            random.choices(ProteinSequence.alphabet.get_symbols(), k=5)
        )
        for _ in range(5)
    }

    leaks = random.sample(list(product(seqs_a.values(), seqs_b.values())), k=num_leaky)

    # Patch out callouts and multiprocessing
    def mock_align_sequences(tup: tuple, *args, **kwargs):
        """Mock the return value of align_sequences"""
        _, _, a, b = tup
        if (a, b) in leaks:
            return MagicMock(), Path("/dev/null"), 1.0, True
        else:
            return MagicMock(), Path("/dev/null"), 0.0, False

    def mock_write_alignment(*args, **kwargs):
        return

    monkeypatch.setattr(sc, "align_sequences", mock_align_sequences)
    monkeypatch.setattr(sc, "write_alignment", mock_write_alignment)

    # Begin test
    result = generate_alignments(seqs_a, seqs_b, Path("/dev/null"), 1, parallel=False)

    assert len(result) == num_leaky


def test_sequence_leakage_main(splits_data_cp):
    metadata_file = splits_data_cp / "metadata.2.csv.gz"
    chk_dir = splits_data_cp / "cluster/f6e35584321f647887eacb8ee369305f"
    index_file = chk_dir / "pindex.4.csv"
    test_table_file = chk_dir / "test_subset_file.csv"
    sc.sequence_leakage_main(
        index_file=index_file,
        metadata_file=metadata_file,
        test_table_file=test_table_file,
        pinder_dir=splits_data_cp,
        cache_path=splits_data_cp / "data/similarity-cache",
        num_cpu=1,
        n_chunks=1,
        chain_overlap_threshold=0.0,
    )


def test_get_processed_alignments(splits_data_cp):
    # Try to get alignments
    cache_path = splits_data_cp / "data/similarity-cache"
    alignments_path = cache_path / "alignments"
    alignments = sc.get_processed_alignments(alignments_path)
    assert len(alignments) == 30


def test_binding_leakage_main(test_dir):
    """Integration test for ECOD leakage and diversity calculations."""
    pinder_dir = test_dir / "pinder_data/mini"
    metadata_file = pinder_dir / "metadata_with_supplementary.parquet"
    index_file = pinder_dir / "index.parquet"

    dfs = ac.binding_leakage_main(index_file=index_file, metadata_file=metadata_file)
    assert len(dfs) == 4
    assert all(isinstance(df, pd.DataFrame) for df in dfs)
    report = dfs[-1]
    assert report.shape == (2, 4)


def test_pfam_diversity_main(test_dir):
    """Integration test for Pfam clan diversity calculations."""
    pinder_dir = test_dir / "pinder_data/mini"
    metadata_file = pinder_dir / "metadata_with_supplementary.parquet"
    index_file = pinder_dir / "index.parquet"
    pfam.pfam_diversity_main(
        index_file=index_file,
        metadata_file=metadata_file,
        pfam_file=pinder_dir / "PDBfam.parquet",
    )


@pytest.mark.parametrize("split", ["test", "val"])
def test_uniprot_leakage_main(test_dir, split):
    """Integration test for uniprot pair leakage calculations."""
    pinder_dir = test_dir / "pinder_data/mini"
    index_file = pinder_dir / "index.parquet"
    leakage_info = ul.uniprot_leakage_main(
        index_path=index_file,
        split=split,
    )
    assert isinstance(leakage_info, tuple)
    leakage_df, percentage = leakage_info
    assert isinstance(percentage, float)
    assert isinstance(leakage_df, pd.DataFrame)
    index_schema = IndexEntry.__annotations__
    index_fields = index_schema.keys()
    expected_columns = set(index_fields)
    expected_columns.add("uniprot_pairs")
    assert set(leakage_df.columns) == expected_columns
