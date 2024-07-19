import json
import pandas as pd
import pytest
import shutil

from pinder.core import get_metadata, PinderSystem
from pinder.core.index.utils import MetadataEntry
from pinder.core.structure import atoms
from pinder.data import get_annotations
from pinder.data.annotation.interface_gaps import annotate_interface_gaps
from pinder.data.annotation.contact_classification import (
    get_crystal_contact_classification,
)
from pinder.data.annotation.detached_components import get_num_connected_components
from pinder.data.annotation.planarity import get_planarity
from pinder.data.annotation.elongation import get_max_var, calculate_elongation


metadata = get_metadata()


@pytest.mark.parametrize("pdb_id, expecting_file", [("7cm8", True), ("vant", False)])
def test_annotate_pisalite(
    tmp_path, pisa_interface_json, pisa_assembly_json, pdb_id, expecting_file
):
    path_to_pdb = tmp_path / pdb_id
    path_to_pdb.mkdir(exist_ok=True, parents=True)
    get_annotations.annotate_pisalite(path_to_pdb)
    assembly_file = path_to_pdb / f"{pdb_id}-pisa-lite-assembly.json"
    interface_file = path_to_pdb / f"{pdb_id}-pisa-lite-interfaces.json"
    assert assembly_file.is_file() == expecting_file
    assert interface_file.is_file() == expecting_file
    if expecting_file:
        with open(assembly_file) as f:
            assembly = json.load(f)

        with open(interface_file) as f:
            interfaces = json.load(f)
        for k, v in pisa_interface_json.items():
            assert interfaces[k] == v

        for k, v in pisa_assembly_json.items():
            assert assembly[k] == v


def test_interface_gap_annotation(pinder_method_test_dir):
    annotation_pymol = pd.DataFrame(
        [
            {
                "pdb_id": "2e31__A1_Q80UW2--2e31__B1_P63208",
                "chain1": "L",
                "chain2": "R",
                "interface_atom_gaps_4A": 1,
                "missing_interface_residues_4A": 0,
                "interface_atom_gaps_8A": 15,
                "missing_interface_residues_8A": 0,
            }
        ]
    )
    id = "2e31__A1_Q80UW2--2e31__B1_P63208"
    pdb_file = pinder_method_test_dir / f"geodock/{id}/{id}.pdb"
    annotation_biotite = annotate_interface_gaps(pdb_file)

    assert annotation_biotite.equals(annotation_pymol)

    entry = MetadataEntry(
        **metadata.query(f'id == "{id}"').to_dict(orient="records")[0]
    )
    annotation_dict = annotation_biotite.to_dict(orient="records")[0]
    for k, v in annotation_dict.items():
        if hasattr(entry, k):
            assert getattr(entry, k) == v


@pytest.mark.skipif(
    shutil.which("prodigy_cryst") is None,
    reason="could not find prodigy_cryst, install from https://github.com/yusuf1759/prodigy-cryst",
)
def test_crystal_contact_classification(pdb_5cq2):
    classification = get_crystal_contact_classification(pdb_5cq2)
    expected = [pdb_5cq2, "3", "1", "0", "2", "0", "0", "0", "1.00", "BIO", "0.516"]
    assert classification == expected


def test_num_connected_components(pdb_5cq2):
    args = (pdb_5cq2, 15.0)
    ch1, ch2, pdb_name = get_num_connected_components(args)
    assert ch1 == 1
    assert ch2 == 1
    assert pdb_name == "5cq2__A2_Q96J02--5cq2__C1_Q9H3M7.pdb"


def test_planarity(pdb_5cq2):
    planarity = get_planarity(pdb_5cq2)
    assert planarity == pytest.approx(3.2286155e-06, abs=1e-6)


def test_get_max_var(pdb_5cq2):
    arr = atoms.atom_array_from_pdb_file(pdb_5cq2)
    max_var = get_max_var(arr.coord)
    assert max_var == pytest.approx(0.7071316, abs=1e-6)


def test_calculate_elongation(test_dir):
    pdb_file = (
        test_dir
        / "pinder_data/nextgen_rcsb/cm/pdb_00007cm8/7cm8__A1_P45040--7cm8__A2_P45040.pdb"
    )
    elongation = calculate_elongation(pdb_file)
    expected = (
        "7cm8_A1_7cm8_A2.pdb",
        0.5344022764647451,
        0.5344022764647451,
        309,
        309,
        5,
        "A1",
        "A2",
        "L",
        "R",
    )
    for i, elem in enumerate(elongation):
        if isinstance(elem, float):
            assert elem == pytest.approx(expected[i], abs=1e-6)
        else:
            assert elem == elongation[i]


def test_annotate_complex(test_dir):
    pdb_file = (
        test_dir
        / "pinder_data/nextgen_rcsb/cm/pdb_00007cm8/7cm8__A1_P45040--7cm8__A2_P45040.pdb"
    )
    args = (pdb_file, 15.0)
    output_tsv = pdb_file.parent / f"{pdb_file.stem}.tsv"
    get_annotations.annotate_complex(args)
    assert output_tsv.is_file()
    results_df = pd.read_csv(output_tsv, sep="\t")
    expected = {
        "path": "7cm8__A1_P45040--7cm8__A2_P45040.pdb",
        "intermolecular_contacts": 149,
        "charged_charged_contacts": 12,
        "charged_polar_contacts": 10,
        "charged_apolar_contacts": 46,
        "polar_polar_contacts": 0,
        "apolar_polar_contacts": 22,
        "apolar_apolar_contacts": 59,
        "link_density": 0.06,
        "label": "BIO",
        "probability": 1.0,
        "number_of_components_1": 1,
        "number_of_components_2": 1,
        "planarity": 4.322492,
        "max_var_1": 0.5344022764647451,
        "max_var_2": 0.5344022764647451,
        "length1": 309,
        "length2": 309,
        "num_atom_types": 5,
        "chain_id1": "A1",
        "chain_id2": "A2",
        "chain1_id": "R",
        "chain2_id": "L",
        "interface_atom_gaps_4A": 0,
        "missing_interface_residues_4A": 0,
        "interface_atom_gaps_8A": 8,
        "missing_interface_residues_8A": 0,
    }
    record = results_df.to_dict(orient="records")[0]
    for k, v in expected.items():
        if isinstance(v, float):
            assert record[k] == pytest.approx(v, abs=1e-6)
        else:
            assert record[k] == v


@pytest.mark.parametrize(
    "two_char_code, parallel",
    [
        (None, False),
        ("cm", False),
    ],
)
def test_get_annotations(two_char_code, parallel, pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    get_annotations.get_annotations(
        data_dir, two_char_code=two_char_code, parallel=parallel
    )
    pdb_file = data_dir / "cm/pdb_00007cm8/7cm8__A1_P45040--7cm8__A2_P45040.pdb"
    output_tsv = pdb_file.parent / f"{pdb_file.stem}.tsv"
    assert output_tsv.is_file()


def test_collect_metadata(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/cm"
    pdb_entries = [
        pdb_dir / "pdb_00007cm8",
        pdb_dir / "pdb_00007cma",
    ]
    df_metadata = get_annotations.collect_metadata(pdb_entries, include_pisa=True)
    assert df_metadata.shape == (2, 12)


def test_collect_interacting_chains(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/cm"
    pdb_entries = [
        pdb_dir / "pdb_00007cm8",
        pdb_dir / "pdb_00007cma",
    ]
    df_interacting = get_annotations.collect_interacting_chains(pdb_entries)
    assert df_interacting.shape == (2, 12)


def test_collect_annotations(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/cm"
    pdb_entries = [
        pdb_dir / "pdb_00007cm8",
        pdb_dir / "pdb_00007cma",
    ]
    df_annotations = get_annotations.collect_annotations(pdb_entries)
    assert df_annotations.shape == (1, 28)


def test_pisa_json_to_dataframe(pinder_data_cp):
    pisa_json_path = pinder_data_cp / "pisa/7cm8/7cm8-pisa-lite-assembly.json"
    df_pisa = get_annotations.pisa_json_to_dataframe(pisa_json_path)
    assert df_pisa.shape == (1, 19)


def test_collect(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb"
    pinder_dir = pinder_data_cp / "pinder_collect_out"
    pinder_dir.mkdir(exist_ok=True)
    get_annotations.collect(pdb_dir, pinder_dir)
    assert (pinder_dir / "structural_metadata.parquet").is_file()
    assert (pinder_dir / "interfaces.parquet").is_file()
    assert (pinder_dir / "interface_annotations.parquet").is_file()
