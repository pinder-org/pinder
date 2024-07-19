from pinder.core.utils.dataclass import (
    atom_array_summary_markdown_repr,
    stringify_dataclass,
)


def test_stringify_dataclass(pinder_temp_dir):
    from pinder.core import PinderSystem

    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    system = PinderSystem(pinder_id)
    struct = system.holo_receptor
    assert isinstance(stringify_dataclass(struct), str)


def test_markdown_repr(pinder_temp_dir):
    from pinder.core import PinderSystem

    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    system = PinderSystem(pinder_id)
    struct = system.holo_receptor
    markdown = atom_array_summary_markdown_repr(struct.atom_array)
    assert isinstance(markdown, str)
