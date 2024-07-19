import pytest
from pinder.core.loader.structure import Structure
from pinder.core import PinderSystem


@pytest.mark.parametrize(
    ["pdb_id_1", "chain_id_1", "pdb_id_2", "chain_id_2"],
    [
        ("1iwl", "A", "6fhm", "A"),  # apo/holo pairing with conformational outliers
        ("6z3w", "A", "6z3w", "A"),  # UNK residues that trigger fallback
        ("1vyo", "A", "6j6j", "A"),  # Different but homologous sequences
    ],
)
def test_superimpose_chain(
    pdb_id_1,
    chain_id_1,
    pdb_id_2,
    chain_id_2,
    superimpose_directory,
):
    """
    Check if :func:`superimpose_chain()` can handle different scenarios.
    In all cases the superimposed structure should have the original number of atoms
    and a low RMSD to the fixed structure.
    """
    chain_1 = Structure(superimpose_directory / f"{pdb_id_1}.pdb")
    chain_1.atom_array = chain_1.atom_array[chain_1.atom_array.chain_id == chain_id_1]
    chain_2 = Structure(superimpose_directory / f"{pdb_id_2}.pdb")
    chain_2.atom_array = chain_2.atom_array[chain_2.atom_array.chain_id == chain_id_2]

    super_chain_1, raw_rmsd, refined_rmsd = chain_1.superimpose(chain_2)
    assert isinstance(super_chain_1, Structure)
    assert super_chain_1.atom_array.shape == chain_1.atom_array.shape
    assert (raw_rmsd == 0 and refined_rmsd == 0) or raw_rmsd > refined_rmsd
    assert refined_rmsd < 2.0


@pytest.mark.parametrize(
    ["remove_differing_atoms", "remove_differing_annotations", "expected_elem"],
    [
        (
            True,
            False,
            {"C", "N", "O", "S"},
        ),  # remove atoms not in common, but keep original element/ins_code/b_factor annotations
        (
            True,
            True,
            {""},
        ),  # remove atoms not in common and strip element/ins_code/b_factor annotations
        (
            False,
            True,
            {"C", "N", "O", "S"},
        ),  # no-op, only align the structures but dont crop
    ],
)
def test_create_masked_bound_unbound_complexes(
    remove_differing_atoms,
    remove_differing_annotations,
    expected_elem,
):
    """
    Check if PinderSystem.create_masked_bound_unbound_complexes can return apo, predicted and holo dimer structures
    superimposed to each other with and without masking the structures to be of the same shape.

    For many ML applications, you may need the bound (ground-truth) structure to have the same dimensions as any
    alternative conformations of the proteins. Given most unbound structures will have different resolved residues
    and differences in their atom annotations, we may need to strip residues from the holo structure to achieve this.

    This test also ensures that you can return cropped structures while preserving the `element`, `ins_code` and `b_factor`
    annotation categories from the original structure (in the corresponding cropped atoms). If the annotations are preserved,
    you may not get the desired result from :func:`biotite.structure.filter_intersection`, but the AtomArray's will stay
    intact and can be written to valid PDB/MMCIF files.
    """
    pinder_id = "2gct__C1_P00766--2gct__A1_P00766"
    ps = PinderSystem(pinder_id)
    holo_R_shape = ps.aligned_holo_R.atom_array.shape[0]
    holo_L_shape = ps.aligned_holo_L.atom_array.shape[0]
    apo_R_shape = ps.apo_receptor.atom_array.shape[0]
    apo_L_shape = ps.apo_ligand.atom_array.shape[0]
    pred_R_shape = ps.pred_receptor.atom_array.shape[0]
    pred_L_shape = ps.pred_ligand.atom_array.shape[0]
    holo, apo, pred = ps.create_masked_bound_unbound_complexes(
        monomer_types=["apo", "predicted"],
        remove_differing_atoms=remove_differing_atoms,
        remove_differing_annotations=remove_differing_annotations,
    )
    holo_shape = holo.atom_array.shape[0]
    apo_shape = apo.atom_array.shape[0]
    pred_shape = pred.atom_array.shape[0]
    if remove_differing_atoms:
        assert (
            holo_shape == apo_shape == pred_shape
        ), f"Expected holo-apo-pred shapes to be identical after masking, got {holo_shape}-{apo_shape}-{pred_shape}"
    pred_unmasked = pred_R_shape + pred_L_shape
    holo_unmasked = holo_R_shape + holo_L_shape
    apo_unmasked = apo_R_shape + apo_L_shape
    assert apo_unmasked >= apo_shape
    assert holo_unmasked >= holo_shape
    assert pred_unmasked >= pred_shape
    # Ensure element annotation is preserved if remove_differing_annotations is False
    holo_elements = set(holo.atom_array.element)
    assert (
        holo_elements == expected_elem
    ), f"Expected element annotation category to be preserved! After masking elements are: {holo_elements}, expected {expected_elem}"
