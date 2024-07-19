"""Utility functions for working with pdbs

Adapted from Raptorx3DModelling/Common/PDBUtils.py
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from biotite.sequence.io.fasta import FastaFile


from pinder.core.loader.structure import Structure
from pinder.core.structure.atoms import get_seq_identity
from pinder.core.utils import setup_logger

import pinder.core.utils.constants as pc


log = setup_logger(__name__)

VALID_AA_3_LETTER_SET = set(pc.three_to_one_noncanonical_mapping.keys())


def three_to_one(x: str) -> str:
    one: str = (
        pc.three_to_one_noncanonical_mapping[x]
        if x in pc.three_to_one_noncanonical_mapping
        else "X"
    )
    return one


def load_fasta_file(seq_file: str | Path, as_str: bool = True) -> str | FastaFile:
    """Load a fasta file.

    Parameters
    ----------
    seq_file : str | Path
        file to read (fasta) sequence from.
    as_str : bool
        whether to return string representation (default) or
        biotite.sequence.io.fasta.FastaFile.

    Returns
    -------
    str | FastaFile
        Sequence as string or biotite.sequence.io.fasta.FastaFile.
    """
    if not isinstance(seq_file, Path):
        seq_file = Path(seq_file)
    if not seq_file.is_file():
        raise FileNotFoundError(seq_file)

    fasta_file = FastaFile.read(seq_file)
    for header, string in fasta_file.items():
        print("Header:", header)
        print(len(string))
        print("Sequence:", string[:50], "...")
        print("Sequence length:", len(string))
    if as_str:
        return list(fasta_file.values())[0]
    return fasta_file


def extract_pdb_seq_from_pdb_file(
    pdb_path: Path,
    chain_id: str | None = None,
) -> tuple[list[str], list[list[int]], list[str]]:
    # this replaces pdb_utils.extract_pdb_seq_from_pdb_file
    struct = Structure(pdb_path)
    if chain_id:
        struct.filter("chain_id", [chain_id], copy=False)

    chain_seq = struct.chain_sequence

    chains = list(chain_seq.keys())
    pdb_seqs = ["".join(slist) for slist in chain_seq.values()]
    res_lists = [
        df["res_id"].tolist() for ch, df in struct.dataframe.groupby("chain_id")
    ]
    return pdb_seqs, res_lists, chains


@lru_cache(16)
def safe_load_sequence(
    seq_path: str | None = None,
    pdb_path: str | None = None,
    chain_id: str | None = None,
) -> str:
    """Loads sequence, either from fasta or given pdb file

    seq_path takes priority over pdb_path. pdb_path or seq_path
    must be provided.

    Parameters:
        seq_path (optional): path to sequence fasta file
        pdb_path (optional): path to pdb file
        chain_id (optional): chain to load sequence from in
        provided pdb file
    """
    if seq_path:
        pdbseqs = [load_fasta_file(seq_path)]
    else:
        assert isinstance(pdb_path, str)
        pdbseqs, *_ = extract_pdb_seq_from_pdb_file(Path(pdb_path), chain_id=chain_id)
    if len(pdbseqs) > 1 and not chain_id:
        log.warning(f"Multiple chains found for pdb: {pdb_path}")
    return pdbseqs[0]


def is_homodimer(
    chain_1_seq: str,
    chain_2_seq: str,
    min_seq_id: float = 0.9,
) -> bool:
    """Whether the two sequences have a similarity above threshold."""
    min_len = min(len(chain_1_seq), len(chain_2_seq))
    max_len = max(len(chain_1_seq), len(chain_2_seq))
    if (min_len / max_len) <= min_seq_id:
        return False

    identity = get_seq_identity(chain_1_seq, chain_2_seq, gap_penalty=(-5.0, -0.2))
    homodimer: bool = identity >= min_seq_id
    return homodimer
