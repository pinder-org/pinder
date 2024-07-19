import pytest
import json
import shutil
import pickle
from pathlib import Path

from pinder.core.structure import atoms
from pinder.data.alignment_utils import (
    DomainInfo,
    Domain,
    Alignment,
    Interface,
    safe_read_contact_json,
    load_contact_info,
    ContactConfig,
    get_foldseek_contacts,
    get_foldseek_dimer_contacts,
    get_foldseek_numbering,
    collect_contact_jsons,
    populate_foldseek_contacts,
)
from pinder.data.foldseek_utils import FOLDSEEK_FIELDS


@pytest.fixture
def foldseek_contact_setup_dir(tmp_path, test_dir):
    data_dir = tmp_path / "test_data"
    print(f"{test_dir=}")
    dimer_pdb = list((test_dir / "foldseek_contacts" / "raw").glob("*.pdb"))[0]
    config = ContactConfig()
    contact_info = {
        "interface_id1": "PDBID1",
        "interface_id2": "PDBID2",
        "R_length": 3,
        "L_length": 6,
        "R_residues": "1,2,3",
        "L_residues": "5,6,7,8,2,3",
    }
    config_hash = "dummy_hash"
    contact_info_path = (
        data_dir / "foldseek_contacts" / config_hash / (dimer_pdb.stem + ".json")
    )
    contact_info_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(contact_info, contact_info_path.open("w"))
    tmp_dimer_pdb = data_dir / dimer_pdb.name
    shutil.copy(dimer_pdb, tmp_dimer_pdb)
    expected_interface_pkl = {
        ("PDBID1", "PDBID2"): Interface(
            pdbid1="PDBID1",
            pdbid2="PDBID2",
            indices1={1, 2, 3},
            indices2={2, 3, 5, 6, 7, 8},
            alignments1=set(),
            alignments2=set(),
        )
    }
    yield data_dir, tmp_dimer_pdb, config, config_hash, expected_interface_pkl


def test_domain_from_domain_info() -> None:
    domain_info = DomainInfo("PDBID", "A", "DOMAINID", "TNAME")
    domain = Domain.from_domain_info(domain_info, 0, 1)
    assert domain.pdb_id == "PDBID"
    assert domain.chain == "A"
    assert domain.domain_id == "DOMAINID"
    assert domain.t_name == "TNAME"
    assert domain.pdb_from == 0
    assert domain.pdb_to == 1


@pytest.mark.parametrize(
    "line_dict",
    [
        {
            "pdbid1": "PDBID1",
            "pdbid2": "PDBID2",
            "alntmscore": 0.5,
            "qstart": 0,
            "qend": 10,
            "qlen": 11,
            "tstart": 5,
            "tend": 8,
            "tlen": 4,
            "alnlen": 15,
        },
        {
            "pdbid1": "PDBID20",
            "pdbid2": "PDBID55",
            "alntmscore": "nan",
            "qstart": 0,
            "qend": 10,
            "qlen": 11,
            "tstart": 5,
            "tend": 8,
            "tlen": 4,
            "alnlen": 15,
        },
    ],
)
def test_alignment_from_line(line_dict) -> None:
    line = "\t".join(str(value) for value in line_dict.values())
    alignment = Alignment.from_line(line)

    for key, expected_value in line_dict.items():
        if (key, expected_value) == ("alntmscore", "nan"):
            expected_value = 0.5
        attr_value = getattr(alignment, key, None)
        assert (
            attr_value == expected_value
        ), f"Expected {key} to be {expected_value}, got {attr_value}"


@pytest.mark.parametrize(
    "line_dict",
    [
        {
            "pdbid1": "PDBID1",
            "pdbid2": "PDBID2",
            "alntmscore": 0.75,
            "qstart": 0,
            "qend": 10,
            "qlen": 11,
            "tstart": 5,
            "tend": 8,
            "tlen": 4,
            "alnlen": 15,
        },
        {
            "pdbid1": "PDBID20",
            "pdbid2": "PDBID55",
            "alntmscore": "nan",
            "qstart": 0,
            "qend": 10,
            "qlen": 11,
            "tstart": 5,
            "tend": 8,
            "tlen": 4,
            "alnlen": 15,
        },
    ],
)
def test_alignment_from_foldseek_line(line_dict) -> None:
    line = "\t".join(str(value) for value in line_dict.values())
    alignment = Alignment.from_foldseek_line(line)
    for key, expected_value in line_dict.items():
        if (key, expected_value) == ("alntmscore", "nan"):
            expected_value = 0.5
        attr_value = getattr(alignment, key, None)
        assert (
            attr_value == expected_value
        ), f"Expected {key} to be {expected_value}, got {attr_value}"


@pytest.mark.parametrize(
    "ecod_info_pair",
    [
        (
            Domain("PDBID1", "A", "DOMAINID1", "TNAME1", 0, 1),
            Domain("PDBID2", "B", "DOMAINID2", "TNAME2", 2, 3),
        )
    ],
)
def test_alignment_from_ecod_info(ecod_info_pair) -> None:
    alignment = Alignment.from_ecod_info(ecod_info_pair)
    assert alignment.pdbid1 == ecod_info_pair[0].pdb_id
    assert alignment.pdbid2 == ecod_info_pair[1].pdb_id
    assert alignment.qstart == ecod_info_pair[0].pdb_from
    assert alignment.qend == ecod_info_pair[0].pdb_to
    assert alignment.tstart == ecod_info_pair[1].pdb_from
    assert alignment.tend == ecod_info_pair[1].pdb_to


def test_interface_from_line():
    # Example line format based on the from_line method's expected input
    line = "PDBID1\tPDBID2\t1,2,3\t4,5,6"
    expected_indices1 = {1, 2, 3}
    expected_indices2 = {4, 5, 6}

    interface = Interface.from_line(line)

    assert interface.pdbid1 == "PDBID1"
    assert interface.pdbid2 == "PDBID2"
    assert interface.indices1 == expected_indices1
    assert interface.indices2 == expected_indices2
    assert interface.alignments1 == set()  # Assuming alignments start empty
    assert interface.alignments2 == set()


@pytest.mark.parametrize(
    "contact_info, expected_interface",
    [
        (
            {
                "interface_id1": "PDBID1",
                "interface_id2": "PDBID2",
                "R_residues": "1,2,3",
                "L_residues": "4,5,6",
            },
            Interface(
                pdbid1="PDBID1",
                pdbid2="PDBID2",
                indices1={1, 2, 3},
                indices2={4, 5, 6},
                alignments1=set(),
                alignments2=set(),
            ),
        ),
        # Add more test cases as needed
    ],
)
def test_from_contact_info(contact_info, expected_interface):
    interface = Interface.from_contact_info(contact_info)
    assert interface.pdbid1 == expected_interface.pdbid1
    assert interface.pdbid2 == expected_interface.pdbid2
    assert interface.indices1 == expected_interface.indices1
    assert interface.indices2 == expected_interface.indices2
    assert interface.alignments1 == expected_interface.alignments1
    assert interface.alignments2 == expected_interface.alignments2


def test_flip_interface():
    original = Interface(
        pdbid1="PDBID1",
        pdbid2="PDBID2",
        indices1={1, 2, 3},
        indices2={4, 5, 6},
        alignments1={("A", 0.5)},
        alignments2={("B", 0.8)},
    )

    flipped = original.flip_interface()

    assert flipped.pdbid1 == original.pdbid2
    assert flipped.pdbid2 == original.pdbid1
    assert flipped.indices1 == original.indices2
    assert flipped.indices2 == original.indices1
    assert flipped.alignments1 == original.alignments2
    assert flipped.alignments2 == original.alignments1


def test_safe_read_contact_json_success(tmp_path):
    contact_json = tmp_path / "contact.json"
    known_content = {"key": "value"}
    contact_json.write_text(json.dumps(known_content))

    result = safe_read_contact_json(contact_json)
    assert result == known_content


def test_safe_read_contact_json_missing_file(tmp_path):
    contact_json = tmp_path / "contact.json"
    result = safe_read_contact_json(contact_json)
    assert result == None


def test_safe_read_contact_json_corrupt_file(tmp_path):
    contact_json = tmp_path / "contact.json"
    contact_json.write_text("{corrupt")
    result = safe_read_contact_json(contact_json)
    assert result == None


def test_load_contact_info_success(tmp_path) -> None:
    contact_json = tmp_path / "contact.json"
    known_content = {"contact": "info", "more": "data"}
    contact_json.write_text(json.dumps(known_content))
    pdb_file = tmp_path / "mock.pdb"
    pdb_file.touch()
    config = ContactConfig()
    result = load_contact_info(pdb_file, contact_json, config)
    assert result == known_content


def test_get_foldseek_contact(test_dir):
    dimer_pdb_path = (
        test_dir
        / "foldseek_contacts"
        / "raw"
        / "6wwe__B1_A0A287AZ37--6wwe__C1_L0N7N1.pdb"
    )
    config = ContactConfig()
    expected_result = {
        "id": "6wwe__B1_A0A287AZ37--6wwe__C1_L0N7N1",
        "interface_id1": "6wwe__B1_A0A287AZ37-R",
        "interface_id2": "6wwe__C1_L0N7N1-L",
        "R_residues": "156,157,160,190,193,194,195,260,261,262,263,404,405,406,407,408,409,410,411,412,413,414,416,417,418,419,420,421,422,423,424,425,430,431",
        "L_residues": "143,165,166,167,168,169,170,172,173,265,290,291,292,293,294,295,296,297,298,299,300,347,348,349,350,351,352",
        "radius": 10.0,
        "R_length": 34,
        "L_length": 27,
        "heavy_only": True,
        "backbone_only": True,
        "only_unique_resi": True,
    }
    result = get_foldseek_dimer_contacts(dimer_pdb_path, config)
    assert result == expected_result


def test_collect_contact_jsons(foldseek_contact_setup_dir):
    (
        data_dir,
        dimer_pdb,
        config,
        config_hash,
        expected_interface_pkl,
    ) = foldseek_contact_setup_dir
    collect_contact_jsons(data_dir, [dimer_pdb], config, config_hash, use_cache=False)

    interface_pkl_path = (
        data_dir.parent / "foldseek_contacts" / config_hash / "interfaces.pkl"
    )
    assert interface_pkl_path.exists()
    interface_pkl = pickle.load(open(interface_pkl_path, "rb"))
    assert interface_pkl == expected_interface_pkl


def test_populate_foldseek_contacts(tmp_path, test_dir):
    dimer_pdb_path = test_dir / "foldseek_contacts" / "raw"
    dimer_pdb_path = Path(
        shutil.copytree(dimer_pdb_path, tmp_path / "foldseek_contacts" / "raw")
    )
    dimer_pdbs = list(dimer_pdb_path.glob("*.pdb"))
    config = ContactConfig()
    populate_foldseek_contacts(dimer_pdbs, config, parallel=False)


@pytest.mark.parametrize(
    "pinder_id, radius, return_calpha_only, expected_R_contacts, expected_L_contacts",
    [
        (
            "7qir__G1_P00766--7qir__D1_P00974",
            10.0,
            True,
            {1, 2, 3, 4, 5, 44},
            {58, 24, 25, 26, 27, 28, 29, 57},
        ),
        (
            "7qir__G1_P00766--7qir__D1_P00974",
            10.0,
            False,
            {1, 2, 3, 4, 5, 44},
            {58, 24, 25, 26, 27, 28, 29, 57},
        ),
        (
            "7qir__G1_P00766--7qir__D1_P00974",
            5.0,
            True,
            {1, 2},
            {26, 27},
        ),
    ],
)
def test_get_foldseek_contacts(
    pinder_id, radius, return_calpha_only, expected_R_contacts, expected_L_contacts
):
    from pinder.core import PinderSystem

    ps = PinderSystem(pinder_id)
    R, L = get_foldseek_contacts(
        ps,
        radius=radius,
        return_calpha_only=return_calpha_only,
    )
    assert set(R) == expected_R_contacts
    assert set(L) == expected_L_contacts


case_expected = {
    "no_calpha_in_interface": {
        "R_res": "156,157,160,190,193,194,195,260,261,262,263,404,405,406,407,408,409,410,411,412,413,414,416,417,418,419,420,421,422,423,424,425,430,431",
        "L_res": "143,165,166,167,168,169,170,172,265,290,292,293,294,295,296,297,298,299,300,347,348,349,350,351,352",
    },
    "raw": {
        "R_res": "156,157,160,190,193,194,195,260,261,262,263,404,405,406,407,408,409,410,411,412,413,414,416,417,418,419,420,421,422,423,424,425,430,431",
        "L_res": "143,165,166,167,168,169,170,172,173,265,290,291,292,293,294,295,296,297,298,299,300,347,348,349,350,351,352",
    },
    "residue_gap_L302": {
        "R_res": "156,157,160,190,193,194,195,260,261,262,263,404,405,406,407,408,409,410,411,412,413,414,416,417,418,419,420,421,422,423,424,425,430,431",
        "L_res": "143,165,166,167,168,169,170,172,173,265,290,291,292,293,294,295,296,297,298,299,346,347,348,349,350,351",
    },
    "no_carbon": {
        "R_res": "156,157,160,190,193,194,195,261,262,263,404,405,406,407,408,409,410,411,412,413,414,416,417,418,419,420,421,422,423,424,425,430,431",
        "L_res": "143,165,166,167,168,169,170,172,265,290,292,293,294,295,296,297,298,299,300,347,348,349,350,351,352",
    },
}


@pytest.mark.parametrize(
    "test_case,expected_L_bounds,expected_R_bounds",
    [
        ("raw", (1, 359), (1, 431)),
        ("no_calpha_in_interface", (1, 359), (1, 431)),
        ("residue_gap_L302", (1, 358), (1, 431)),
        ("no_carbon", (1, 359), (1, 431)),
    ],
)
def test_get_foldseek_numbering(
    test_case, expected_L_bounds, expected_R_bounds, test_dir
):
    test_case_dir = test_dir / "foldseek_contacts" / test_case
    pdb_file = list(test_case_dir.glob("*.pdb"))[0]
    structure = atoms.atom_array_from_pdb_file(pdb_file)
    arr_ori = structure.copy()
    R_arr = arr_ori[(arr_ori.chain_id == "R")].copy()
    L_arr = arr_ori[(arr_ori.chain_id == "L")].copy()
    R_map = get_foldseek_numbering(R_arr)
    L_map = get_foldseek_numbering(L_arr)
    L_bounds = (min(L_map.values()), max(L_map.values()))
    R_bounds = (min(R_map.values()), max(R_map.values()))
    assert L_bounds == expected_L_bounds
    assert R_bounds == expected_R_bounds


@pytest.mark.parametrize(
    "test_case",
    [
        "raw",
        "no_calpha_in_interface",
        "residue_gap_L302",
        "no_carbon",
    ],
)
def test_get_foldseek_dimer_contacts(test_case, test_dir):
    test_case_dir = test_dir / "foldseek_contacts" / test_case
    pdb_file = list(test_case_dir.glob("*.pdb"))[0]
    foldseek_contacts = get_foldseek_dimer_contacts(pdb_file)
    assert foldseek_contacts["R_residues"] == case_expected[test_case]["R_res"]
    assert foldseek_contacts["L_residues"] == case_expected[test_case]["L_res"]
