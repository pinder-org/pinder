from pinder.core.index.id import Dimer, Monomer, Protein


def test_create_dimer():
    assert str(
        Dimer.from_string("6q0r__B_Q66K64-6q0r__C_Q66K64--6q0r__D_Q14498")
    ) == str(
        Dimer(
            Monomer(
                [
                    Protein(source="6q0r", uniprot="Q66K64", chain="B"),
                    Protein(source="6q0r", uniprot="Q66K64", chain="C"),
                ]
            ),
            Monomer([Protein(source="6q0r", uniprot="Q14498", chain="D")]),
        )
    )


def test_access_dimer_elements():
    filename = "6q0r__B_Q66K64-6q0r__C_Q66K64--6q0r__D_Q14498.pdb"
    parsed_dimer = Dimer.from_string(filename)
    assert parsed_dimer.monomer1.proteins[0].uniprot == "Q66K64"
    assert parsed_dimer.monomer2.proteins[0].uniprot == "Q14498"
    assert parsed_dimer.monomer1.proteins[0].source == "6q0r"
    assert parsed_dimer.monomer2.proteins[0].source == "6q0r"
    assert str(filename) == str(parsed_dimer) + ".pdb"


def test_protein_range():
    filename = "6q0r__B_Q66K64--af2__Q14498_1_100.pdb"
    parsed_dimer = Dimer.from_string(filename)
    assert parsed_dimer.monomer1.proteins[0].uniprot == "Q66K64"
    assert parsed_dimer.monomer1.proteins[0].source == "6q0r"
    assert parsed_dimer.monomer1.proteins[0].chain == "B"

    assert parsed_dimer.monomer2.proteins[0].source == "af2"
    assert parsed_dimer.monomer2.proteins[0].uniprot == "Q14498"
    assert parsed_dimer.monomer2.proteins[0].from_residue == 1
    assert parsed_dimer.monomer2.proteins[0].to_residue == 100

    assert str(filename) == str(parsed_dimer) + ".pdb"


def test_monomer():
    for fname in [
        "6q0r__B_Q66K64.pdb",
        "6q0r__B_Q66K64-L.pdb",
        "af2__Q14498_1_100.pdb",
        "af2__Q14498_1_100-R.pdb",
    ]:
        parsed_monomer = Monomer.from_string(fname)
        assert fname == str(parsed_monomer) + ".pdb"
