import pytest


def test_transform_abc():
    from pinder.core import PinderSystem
    from pinder.core.loader.transforms import TransformBase

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
    from pinder.core import PinderSystem
    from pinder.core.loader.transforms import SuperposeToReference

    s = PinderSystem(pinder_id)
    t = SuperposeToReference()
    ppi = t.transform(s)
    assert isinstance(ppi, PinderSystem)
    assert hasattr(ppi, "apo_receptor")


def test_transforms():
    from pinder.core.loader import transforms as t

    for transform in [
        "AddNoise",
        "AddEdges",
        "Noise",
        "CenterSystems",
        "SampleContact",
        "MarkContacts",
        "CheckLength",
        "CheckLengthPrody",
        "CenterOnReceptor",
        "RandomLigandPosition",
        "SetTime",
        "RandomSystemRotation",
        "GetContacts",
        "SampleContacts",
    ]:
        getattr(t, transform)()
