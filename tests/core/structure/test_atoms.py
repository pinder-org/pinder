import shutil
from pathlib import Path
import pytest
import numpy as np
import biotite.structure as struc
from biotite.structure.atoms import AtomArray

from pinder.core.structure import atoms
from pinder.core.structure.models import BackboneDefinition


def test_pdb_loader(pdb_5cq2):
    arr = atoms.atom_array_from_pdb_file(pdb_5cq2)

    assert isinstance(arr, AtomArray)
    assert arr.shape == (692,)


def test_atom_masks(example_atoms):
    arr = example_atoms.copy()
    mask = atoms.backbone_mask(arr, BackboneDefinition("dockq"))
    assert mask.sum() == 349

    assert mask.shape == (692,)
    assert set(arr[mask].atom_name) == set(atoms.DOCKQ_BACKBONE_ATOMS)

    assert atoms.apply_mask(arr, mask).shape == arr[mask].shape

    assert set(atoms.filter_atoms(arr, calpha_only=True).atom_name) == {"CA"}


@pytest.mark.parametrize(
    "backbone_only, calpha_only, expected_mask",
    [
        (True, True, 349),
        (True, False, 349),
        (False, True, 88),
    ],
)
def test_get_backbone_atom_masks(
    backbone_only, calpha_only, expected_mask, example_atoms
):
    arr = example_atoms.copy()
    arr_mask, stack_mask = atoms.get_backbone_atom_masks(
        arr, struc.stack([arr]), backbone_only=backbone_only, calpha_only=calpha_only
    )
    assert arr_mask.shape[0] == 692
    assert arr_mask.shape == stack_mask.shape
    assert arr_mask.sum() == stack_mask.sum() == expected_mask


def test_resn2seq(example_atoms):
    assert atoms.resn2seq(example_atoms.res_name[0:5]) == "TTTPP"
    structure, numbering, resn = atoms._get_structure_and_res_info(example_atoms)
    assert isinstance(structure, AtomArray)
    assert set(numbering) == set(example_atoms.res_id)
    assert atoms.resn2seq(resn[0:2]) == "TP"


def test_assign_receptor_ligand(example_atoms):
    chains = set(example_atoms.chain_id)
    rec, lig = atoms.assign_receptor_ligand(example_atoms, chains)
    assert rec == ["R"]
    assert lig == ["L"]


def test_normalize_orientation(example_atoms):
    normalized = atoms.normalize_orientation(example_atoms)
    norm_var = normalized.coord.var(axis=0)
    orig_var = example_atoms.coord.var(axis=0)
    assert norm_var[0] > orig_var[0]
    assert norm_var[1] < orig_var[1]
    assert np.argmin(norm_var) == 2


def test_get_seq_alignments(test_dir):
    pdb_dir = test_dir / "pinder_data/pinder/pdbs"
    holo_R = pdb_dir / "7cma__A1_A0A2X0TC55-R.pdb"
    holo_L = pdb_dir / "7cma__B2_A0A2X0TC55-L.pdb"
    R = atoms.atom_array_from_pdb_file(holo_R)
    L = atoms.atom_array_from_pdb_file(holo_R)
    R_numbering, R_resn = struc.get_residues(R)
    R_seq = atoms.resn2seq(R_resn)
    L_numbering, L_resn = struc.get_residues(L)
    L_seq = atoms.resn2seq(L_resn)
    ident = atoms.get_seq_identity(R_seq, L_seq)
    assert isinstance(ident, float)
    assert ident == pytest.approx(1.0)

    alns = atoms.get_seq_alignments(R_seq, L_seq)
    mismatches, matches = atoms.calc_num_mismatches(alns)
    assert mismatches == 0
    assert matches == len(R_seq)

    R_seq_aln, L_seq_aln, R_numbering, L_numbering = atoms.align_sequences(R_seq, L_seq)
    assert R_seq_aln == L_seq_aln == R_seq == L_seq
    assert R_numbering == list(range(1, len(R_seq) + 1))
    assert L_numbering == list(range(1, len(L_seq) + 1))


def test_buried_sasa(test_dir):
    # Example structure with non-standard residue names that fail ProtOr vdw radii lookup
    pdb_file = test_dir / "7qir__G1_P00766--7qir__D1_P00974.pdb"
    arr = atoms.atom_array_from_pdb_file(pdb_file)
    R = arr[arr.chain_id == "R"]
    L = arr[arr.chain_id == "L"]
    dsasa = atoms.get_buried_sasa(R, L)
    assert isinstance(dsasa, int)
    assert dsasa == 215


def test_rename_chains(tmp_path, test_dir):
    # Example structure with non-standard residue names that fail ProtOr vdw radii lookup
    pdb_file = test_dir / "7qir__G1_P00766--7qir__D1_P00974.pdb"
    pdb_file = Path(shutil.copy(pdb_file, tmp_path))
    arr_ori = atoms.atom_array_from_pdb_file(pdb_file)
    assert set(arr_ori.chain_id) == {"R", "L"}
    atoms.rename_chains(pdb_file, new_chain={"R": "A", "L": "Z"})
    arr_mod = atoms.atom_array_from_pdb_file(pdb_file)
    assert set(arr_mod.chain_id) == {"A", "Z"}
    assert (
        arr_ori[arr_ori.chain_id == "R"][0].res_name
        == arr_mod[arr_mod.chain_id == "A"][0].res_name
    )
