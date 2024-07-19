from __future__ import annotations
import pytest
import shutil
from pathlib import Path
from tempfile import mkdtemp

from biotite.structure.io.pdb import PDBFile
from torch import Size

from pinder.core import PinderSystem
from pinder.core.loader.structure import backbone_atom_tensor, Structure
from pinder.core.utils import pdb_utils


test_dir = Path(__file__).absolute().parent / "test_data"


def pdb_file_example(test_dir) -> Path:
    return test_dir / "5cq2__A2_Q96J02--5cq2__C1_Q9H3M7.pdb"


def test_safe_load_sequence(pdb_5cq2):
    seq = pdb_utils.safe_load_sequence(seq_path=None, pdb_path=str(pdb_5cq2))
    assert isinstance(seq, str)


def test_backbone_atom_tensor():
    atom_tys = ("N", "CA", "C", "O")
    bb_tensor = backbone_atom_tensor(atom_tys)
    assert bb_tensor.shape == Size([4])
    assert bb_tensor.tolist() == [0, 1, 2, 3]


def test_is_homodimer(pinder_temp_dir):
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    dimer = PinderSystem(entry=pinder_id)
    homodimer = pdb_utils.is_homodimer(
        dimer.holo_receptor.sequence,
        dimer.holo_ligand.sequence,
        min_seq_id=0.9,
    )
    assert not homodimer


def test_load_fasta_file(fasta_O14727):
    fasta_seq = pdb_utils.load_fasta_file(fasta_O14727)
    assert len(fasta_seq) == 1248


def test_three_to_one(pdb_5cq2, tmp_path):
    pdb_file = Path(shutil.copy(pdb_5cq2, tmp_path))
    model = PDBFile.read(str(pdb_file))
    arr = model.get_structure(model=1, extra_fields=["atom_id"])  # noqa
    non_std = arr[arr.res_name == "NH2"][0].res_name
    assert pdb_utils.three_to_one(non_std) == "X"


@pytest.mark.parametrize(
    "chain_id",
    [
        None,
        "R",
        "L",
    ],
)
def test_extract_pdb_seq_from_pdb_file(
    chain_id: str | None,
    pdb_5cq2,
) -> tuple[list, ...]:
    struct = Structure(pdb_5cq2)
    if chain_id:
        struct.filter("chain_id", [chain_id], copy=False)

    chain_seq = struct.chain_sequence
    chains = list(chain_seq.keys())
    pdb_seqs = ["".join(slist) for slist in chain_seq.values()]
    res_lists = [
        df["res_id"].tolist() for ch, df in struct.dataframe.groupby("chain_id")
    ]
    list_len = 1 if chain_id else 2
    assert len(pdb_seqs) == list_len
    assert len(chains) == list_len
    assert len(res_lists) == list_len
    expected_seq_lens = {
        "L": [13],
        "R": [76],
        None: [13, 76],
    }
    assert sorted([len(ch_seq) for ch_seq in pdb_seqs]) == expected_seq_lens[chain_id]
