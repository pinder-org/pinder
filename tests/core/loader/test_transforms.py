import pytest
from pinder.core import PinderSystem
from pinder.core.utils import constants as pc
from pinder.core.loader.transforms import (
    SelectAtomTypes,
    SuperposeToReference,
    TransformBase,
    RandomLigandTransform,
)
from pinder.core.loader.structure import Structure
from biotite.structure import AtomArray
import numpy as np


@pytest.fixture
def mock_structure():
    # Create a mock structure with ligand and receptor chains
    ligand_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    receptor_coords = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]])

    ligand_atom_array = AtomArray(3)
    ligand_atom_array.coord = ligand_coords
    ligand_atom_array.chain_id = np.array(["L", "L", "L"])

    receptor_atom_array = AtomArray(3)
    receptor_atom_array.coord = receptor_coords
    receptor_atom_array.chain_id = np.array(["R", "R", "R"])

    structure: Structure = Structure(
        "", atom_array=ligand_atom_array + receptor_atom_array
    )
    return structure


def test_random_ligand_transform(mock_structure):
    original_ligand_coords = mock_structure.atom_array[
        mock_structure.atom_array.chain_id == "L"
    ].coord.copy()
    original_receptor_coords = mock_structure.atom_array[
        mock_structure.atom_array.chain_id == "R"
    ].coord.copy()
    transform = RandomLigandTransform(max_translation=5.0)
    transformed_structure = transform.transform(mock_structure)

    # Check if the structure still has both ligand and receptor chains
    assert set(transformed_structure.atom_array.chain_id) == {"L", "R"}

    # Check if the ligand coordinates have been transformed
    ligand_coords = transformed_structure.atom_array[
        transformed_structure.atom_array.chain_id == "L"
    ].coord
    assert not np.array_equal(original_ligand_coords, ligand_coords)
    assert ligand_coords.shape == (
        3,
        3,
    )  # Ensure the number of ligand atoms remains the same

    # Check if the receptor coordinates remain unchanged
    receptor_coords = transformed_structure.atom_array[
        transformed_structure.atom_array.chain_id == "R"
    ].coord
    assert np.array_equal(original_receptor_coords, receptor_coords)


def test_transform_abc():
    s = PinderSystem("8i2f__A1_O34841--8i2f__B1_P54421")
    with pytest.raises(NotImplementedError):
        TransformBase().transform(s)


@pytest.mark.parametrize(
    "pinder_id",
    [
        "8i2f__A1_O34841--8i2f__B1_P54421",
        "2oxz__A1_P39900--2oxz__B1_UNDEFINED",
    ],
)
def test_transform_superpose(pinder_id):
    s = PinderSystem(pinder_id)
    t = SuperposeToReference()
    ppi = t.transform(s)
    assert isinstance(ppi, PinderSystem)
    assert hasattr(ppi, "apo_receptor")


@pytest.mark.parametrize(
    "pinder_id, atom_types",
    [
        ("8i2f__A1_O34841--8i2f__B1_P54421", ["CA"]),
        ("8i2f__A1_O34841--8i2f__B1_P54421", ["CA", "N", "C", "O"]),
        ("8i2f__A1_O34841--8i2f__B1_P54421", ["foo"]),
        ("2oxz__A1_P39900--2oxz__B1_UNDEFINED", ["CA"]),
        ("2oxz__A1_P39900--2oxz__B1_UNDEFINED", ["CA", "N", "C", "O"]),
        ("2oxz__A1_P39900--2oxz__B1_UNDEFINED", ["foo"]),
    ],
)
def test_select_atom_types_structure_transform(pinder_id, atom_types):
    valid_atom_names = set(pc.ALL_ATOMS)
    expected_atom_names = set(atom_types).intersection(valid_atom_names)
    s = PinderSystem(pinder_id)
    t = SelectAtomTypes(atom_types=atom_types)
    native = s.native
    assert len(native.atom_names) > len(expected_atom_names)
    native = t.transform(native)
    assert set(native.atom_names) == set(expected_atom_names)
