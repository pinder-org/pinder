from numbers import Number
import biotite.structure as struc
import pandas as pd
import pytest
import torch
from pinder.core import (
    PinderSystem,
    get_index,
)
from pinder.core.loader.structure import (
    canonical_atom_type_mask,
    mask_common_uniprot,
    Structure,
)
from pinder.core.structure.atoms import get_seq_aligned_structures

pindex = get_index()


def test_structure_loader(pdb_5cq2):
    RL = Structure(pdb_5cq2)
    assert isinstance(RL, Structure)
    assert RL.chains == ["L", "R"]
    assert RL.atom_array.shape[0] == 692
    assert RL.dataframe.shape == (692, 12)

    R = RL.filter("chain_id", ["R"])
    assert isinstance(R, Structure)
    assert R.chains == ["R"]

    L = RL.filter("chain_id", ["L"])
    RL = R + L

    assert RL.chains == ["L", "R"]
    assert RL.atom_array.shape[0] == 692
    assert RL.dataframe.shape == (692, 12)

    RL_calpha = RL.filter("atom_name", mask=["CA"])
    assert RL_calpha.atom_names == ["CA"]

    RL.filter("atom_name", mask=["CA"], copy=False)
    assert RL.atom_names == ["CA"]


def test_pinder_unbound_difficulty(tmp_path):
    apo_r_expected = {
        "Fnat": 0.768,
        "Fnonnat": 0.361,
        "common_contacts": 52,
        "differing_contacts": 26,
        "bound_contacts": 69,
        "unbound_contacts": 78,
        "fnonnat_R": 0.130,
        "fnonnat_L": 0.0769,
        "fnat_R": 0.8,
        "fnat_L": 0.961,
        "difficulty": "Rigid-body",
        "I-RMSD": 1.16,
        "L-RMSD": 1.20,
        "monomer_name": "apo",
        "unbound_id": "2rsx__A1_O34841--8i2f__B1_P54421-L",
        "unbound_body": "receptor",
        "matched_interface_chains": 2,
    }
    apo_l_expected = {
        "Fnat": 0.710,
        "Fnonnat": 0.289,
        "common_contacts": 49,
        "differing_contacts": 20,
        "bound_contacts": 69,
        "unbound_contacts": 69,
        "fnonnat_R": 0.206,
        "fnonnat_L": 0.076,
        "fnat_R": 0.92,
        "fnat_L": 0.923,
        "difficulty": "Rigid-body",
        "I-RMSD": 0.720,
        "L-RMSD": 0.668,
        "monomer_name": "apo",
        "unbound_id": "8i2f__A1_O34841-R--8i2d__A1_P54421",
        "unbound_body": "ligand",
        "matched_interface_chains": 2,
    }
    dual_expected = {
        "Fnat": 0.608,
        "Fnonnat": 0.471,
        "common_contacts": 42,
        "differing_contacts": 37,
        "bound_contacts": 69,
        "unbound_contacts": 79,
        "fnonnat_R": 0.310,
        "fnonnat_L": 0.166,
        "fnat_R": 0.8,
        "fnat_L": 0.961,
        "difficulty": "Medium",
        "I-RMSD": 1.33,
        "L-RMSD": 0.668,
        "R-RMSD": 1.20,
        "monomer_name": "apo",
        "unbound_id": "2rsx__A1_O34841--8i2d__A1_P54421",
        "unbound_body": "receptor_ligand",
        "matched_interface_chains": 2,
    }
    ps = PinderSystem(
        "8i2f__A1_O34841--8i2f__B1_P54421",
        apo_receptor_pdb_code="2rsx",
        apo_ligand_pdb_code="8i2d",
    )
    apo_r_actual = ps.apo_monomer_difficulty("apo", "receptor")
    apo_l_actual = ps.apo_monomer_difficulty("apo", "ligand")
    for k in apo_r_expected.keys():
        r_exp = apo_r_expected[k]
        l_exp = apo_l_expected[k]
        if isinstance(r_exp, float):
            assert apo_r_actual[k] == pytest.approx(r_exp, rel=1e-1)
            assert apo_l_actual[k] == pytest.approx(l_exp, rel=1e-1)
        else:
            assert apo_r_actual[k] == r_exp
            assert apo_l_actual[k] == l_exp

    apo_rl_actual = ps.unbound_difficulty("apo")
    for k, v in dual_expected.items():
        if isinstance(v, float):
            assert apo_rl_actual[k] == pytest.approx(v, rel=1e-1)
        else:
            assert apo_rl_actual[k] == v

    expected_structures = [
        "holo_receptor",
        "holo_ligand",
        "apo_receptor",
        "apo_ligand",
        "pred_receptor",
        "pred_ligand",
    ]
    for attr in expected_structures:
        assert hasattr(ps, attr)

    # Other edge-case system
    edge_case_expected = {
        "Fnat": 0.923,
        "Fnonnat": 0.357,
        "common_contacts": 36,
        "differing_contacts": 19,
        "bound_contacts": 39,
        "unbound_contacts": 55,
        "fnonnat_R": 0.1,
        "fnonnat_L": 0.0,
        "fnat_R": 1.0,
        "fnat_L": 1.0,
        "difficulty": "Medium",
        "I-RMSD": 1.76,
        "L-RMSD": 1.22,
        "monomer_name": "apo",
        "unbound_id": "6hn3__A1_P36969--5h5q__B1_UNDEFINED-L",
        "unbound_body": "receptor",
        "matched_interface_chains": 2,
    }
    ps = PinderSystem(
        "5h5q__A1_P36969--5h5q__B1_UNDEFINED",
        apo_receptor_pdb_code="6hn3",
    )
    edge_case_actual = ps.apo_monomer_difficulty("apo", "receptor")
    for k, v in edge_case_expected.items():
        if isinstance(v, Number):
            assert edge_case_actual[k] == pytest.approx(v, rel=1e-1)
        else:
            assert edge_case_actual[k] == v

    edge_case_expected_2 = {
        "Fnat": 0.984,
        "Fnonnat": 0.765,
        "common_contacts": 187,
        "differing_contacts": 606,
        "bound_contacts": 190,
        "unbound_contacts": 797,
        "fnonnat_R": 0.510,
        "fnonnat_L": 0.246,
        "fnat_R": 0.985,
        "fnat_L": 1.0,
        "difficulty": "Medium",
        "I-RMSD": 0.650,
        "L-RMSD": 0.385,
        "monomer_name": "apo",
        "unbound_id": "5o8c__A1_P42212--2g3d__A1_P42212-L",
        "unbound_body": "receptor",
        "matched_interface_chains": 2,
    }
    ps = PinderSystem("2g3d__B1_P42212--2g3d__A1_P42212", apo_receptor_pdb_code="5o8c")
    edge_case_actual_2 = ps.apo_monomer_difficulty("apo", "receptor")
    for k, v in edge_case_expected_2.items():
        if isinstance(v, Number):
            assert edge_case_actual_2[k] == pytest.approx(v, rel=1e-1)
        else:
            assert edge_case_actual_2[k] == v


def test_pinder_system(tmp_path):
    assert isinstance(pindex, pd.DataFrame)
    assert pindex.shape == (2319564, 34)

    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    ps = PinderSystem(entry=pinder_id)
    assert isinstance(ps, PinderSystem)
    af_difficulty = ps.unbound_difficulty("predicted")
    expected_difficulty = {
        "Fnat": 0.742,
        "Fnonnat": 0.684,
        "common_contacts": 127,
        "differing_contacts": 276,
        "bound_contacts": 171,
        "unbound_contacts": 403,
        "fnonnat_R": 0.584,
        "fnonnat_L": 0.490,
        "fnat_R": 0.885,
        "fnat_L": 0.912,
        "difficulty": "Medium",
        "I-RMSD": 1.78,
        "L-RMSD": 2.15,
        "R-RMSD": 2.85,
        "monomer_name": "predicted",
        "unbound_id": "af__Q07009--af__Q64537",
        "unbound_body": "receptor_ligand",
        "matched_interface_chains": 2,
    }
    for k, v in expected_difficulty.items():
        if isinstance(v, Number):
            assert af_difficulty[k] == pytest.approx(v, rel=1e-1)
        else:
            assert af_difficulty[k] == v

    expected_structures = [
        "holo_receptor",
        "holo_ligand",
        "apo_receptor",
        "apo_ligand",
        "pred_receptor",
        "pred_ligand",
    ]
    for attr in expected_structures:
        assert hasattr(ps, attr)

    pred_complex = ps.create_pred_complex(remove_differing_atoms=False)
    assert isinstance(pred_complex, Structure)
    assert pred_complex.dataframe.shape == (6452, 12)

    pred_complex = ps.create_pred_complex(remove_differing_atoms=True)
    assert isinstance(pred_complex, Structure)
    assert pred_complex.dataframe.shape == (6385, 12)


def test_structure_sequence_alignment(tmp_path):
    assert isinstance(pindex, pd.DataFrame)
    assert pindex.shape == (2319564, 34)

    pinder_id = "5e6u__A1_P20701--5e6u__B1_P05107"
    ps = PinderSystem(entry=pinder_id, apo_receptor_pdb_code="7kc6")

    holo_R = ps.holo_receptor
    pred_R = ps.pred_receptor
    holo, pred = get_seq_aligned_structures(holo_R.atom_array, pred_R.atom_array)
    holo_res, holo_resn = struc.get_residues(holo)
    pred_res, pred_resn = struc.get_residues(pred)
    assert len(pred_res) == len(holo_res)
    assert len(pred_resn) == len(holo_resn)
    assert len(pred_res) == 583
    holo_calpha = holo[holo.atom_name == "CA"].shape[0]
    pred_calpha = pred[pred.atom_name == "CA"].shape[0]
    assert holo_calpha == pred_calpha


@pytest.mark.parametrize(
    "pinder_id, remove_differing_atoms, renumber_residues, equal_shapes, equal_sequence, equal_resid, expected_R_delta, expected_L_delta",
    [
        ("7b80__A1_G3I8R9--7b80__B1_Q9BVA6", True, True, True, True, True, 109, 74),
        ("7b80__A1_G3I8R9--7b80__B1_Q9BVA6", True, False, True, True, False, 109, 74),
        ("7b80__A1_G3I8R9--7b80__B1_Q9BVA6", False, False, False, True, False, 16, 20),
        ("7b80__A1_G3I8R9--7b80__B1_Q9BVA6", False, True, False, True, True, 16, 20),
        ("4je4__A1_Q06124--4je4__B1_P02751", True, True, True, True, True, 68, 189),
        ("4je4__A1_Q06124--4je4__B1_P02751", True, False, True, True, False, 68, 189),
        ("4je4__A1_Q06124--4je4__B1_P02751", False, False, False, True, False, 59, 22),
        ("4je4__A1_Q06124--4je4__B1_P02751", False, True, False, True, True, 59, 22),
    ],
)
def test_structure_align_common_sequence(
    pinder_id,
    remove_differing_atoms,
    renumber_residues,
    equal_shapes,
    equal_sequence,
    equal_resid,
    expected_R_delta,
    expected_L_delta,
):
    ps = PinderSystem(pinder_id)
    apo_L, apo_R = ps.apo_ligand, ps.apo_receptor
    pred_L, pred_R = ps.pred_ligand, ps.pred_receptor
    holo_L, holo_R = ps.holo_ligand, ps.holo_receptor
    holo_R_atoms, holo_L_atoms = holo_R.atom_array.shape[0], holo_L.atom_array.shape[0]
    apo_R, holo_R = apo_R.align_common_sequence(
        holo_R,
        remove_differing_atoms=remove_differing_atoms,
        renumber_residues=renumber_residues,
    )
    apo_L, holo_L = apo_L.align_common_sequence(
        holo_L,
        remove_differing_atoms=remove_differing_atoms,
        renumber_residues=renumber_residues,
    )
    pred_R, holo_R = pred_R.align_common_sequence(
        holo_R,
        remove_differing_atoms=remove_differing_atoms,
        renumber_residues=renumber_residues,
    )
    pred_L, holo_L = pred_L.align_common_sequence(
        holo_L,
        remove_differing_atoms=remove_differing_atoms,
        renumber_residues=renumber_residues,
    )
    struc_pairs = [
        (apo_R, holo_R, holo_R_atoms, expected_R_delta),
        (apo_L, holo_L, holo_L_atoms, expected_L_delta),
        (pred_R, holo_R, holo_R_atoms, expected_R_delta),
        (pred_L, holo_L, holo_L_atoms, expected_L_delta),
    ]
    for target, holo, ori_holo_at, expected_delta in struc_pairs:
        target_at_shape = target.atom_array.shape[0]
        holo_at_shape = holo.atom_array.shape[0]
        actual_atom_delta = ori_holo_at - holo_at_shape
        assert actual_atom_delta == expected_delta
        if equal_shapes:
            assert (
                target_at_shape == holo_at_shape,
                f"Target {target.pinder_id} atom shape differs from holo {holo.pinder_id}! "
                f"{target_at_shape} != {holo_at_shape}!",
            )
        if equal_sequence:
            target_seq = target.sequence
            holo_seq = holo.sequence

            assert (
                target_seq == holo_seq,
                f"Target {target.pinder_id} sequence differs from holo {holo.pinder_id}! "
                f"{target_seq} != {holo_seq}!",
            )
        if equal_resid:
            target_res = target.residues
            holo_res = holo.residues
            assert (
                target_res == holo_res,
                f"Target {target.pinder_id} reside numbering differs from holo {holo.pinder_id}! "
                f"{target_res} != {holo_res}!",
            )


@pytest.mark.parametrize(
    "atom_tys,mask_shape,mask_dim1,mask_dim2",
    [
        (("CA"), torch.Size([21, 2]), 21, 0),
        (("CA", "CB"), torch.Size([21, 2]), 21, 19),
    ],
)
def test_canonical_atom_type_mask(atom_tys, mask_shape, mask_dim1, mask_dim2):
    at_mask = canonical_atom_type_mask(atom_tys)
    assert at_mask.shape == mask_shape
    assert at_mask[:, 0].sum().tolist() == mask_dim1
    assert at_mask[:, 1].sum().tolist() == mask_dim2


@pytest.mark.parametrize(
    "pinder_id, monomer1_attr, monomer2_attr, monomer1_shape, monomer2_shape",
    [
        (
            "7b80__A1_G3I8R9--7b80__B1_Q9BVA6",
            "holo_receptor",
            "apo_receptor",
            4009,
            3942,
        ),
        (
            "7b80__A1_G3I8R9--7b80__B1_Q9BVA6",
            "holo_receptor",
            "pred_receptor",
            4019,
            4036,
        ),
        ("7b80__A1_G3I8R9--7b80__B1_Q9BVA6", "holo_receptor", "apo_ligand", 4025, 2693),
        ("4je4__A1_Q06124--4je4__B1_P02751", "holo_receptor", "apo_receptor", 752, 752),
        (
            "4je4__A1_Q06124--4je4__B1_P02751",
            "holo_receptor",
            "pred_receptor",
            811,
            811,
        ),
        ("4je4__A1_Q06124--4je4__B1_P02751", "holo_receptor", "apo_ligand", 820, 2042),
    ],
)
def test_mask_common_uniprot(
    pinder_id, monomer1_attr, monomer2_attr, monomer1_shape, monomer2_shape
):
    ps = PinderSystem(pinder_id)
    monomer1 = getattr(ps, monomer1_attr)
    monomer2 = getattr(ps, monomer2_attr)
    masked1, masked2 = mask_common_uniprot(monomer1, monomer2)
    assert masked1.atom_array.shape[0] == monomer1_shape
    assert masked2.atom_array.shape[0] == monomer2_shape
