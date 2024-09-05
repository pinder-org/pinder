import pytest
from pinder.core import PinderSystem
from pinder.core.utils import constants as pc
from pinder.core.loader.transforms import (
    SelectAtomTypes,
    SuperposeToReference,
    TransformBase,
)


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
