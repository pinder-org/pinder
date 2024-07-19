import pytest
import shutil
from pathlib import Path

import pandas as pd

from pinder.core.structure import atoms
from pinder.data import get_data, rcsb_rsync
from pinder.data.foldseek_utils import create_fasta_from_systems


def test_download_rcsb_files(tmp_path):
    rcsb_rsync.download_rscb_files(tmp_path / "data", "1a")
    data_dir = tmp_path / "data"
    assert data_dir.is_dir()
    cif = data_dir / "1a/pdb_000011as/pdb_000011as_xyz-enrich.cif.gz"
    assert cif.is_file()


def test_get_rsync_two_char_pdb_entries():
    entries = rcsb_rsync.get_rsync_two_char_pdb_entries(two_char_code="cm")
    assert isinstance(entries, list)
    assert len(entries) >= 245


def test_get_all_rsync_entries():
    entries = rcsb_rsync.get_all_rsync_entries(two_char_codes=["cm", "fa"])
    assert isinstance(entries, list)
    assert len(entries) >= (245 + 250)


def test_get_two_char_codes_not_downloaded(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    missing = rcsb_rsync.get_two_char_codes_not_downloaded(
        data_dir, two_char_codes=["cm", "fa"]
    )
    assert len(missing) == 2


def test_get_interacting_chains(pdb_5cq2):
    arr = atoms.atom_array_from_pdb_file(pdb_5cq2)
    entities = pd.DataFrame(
        [
            {"chain": "L", "asym_id": "L", "entry_id": "5cq2", "pdb_strand_id": "L"},
            {"chain": "R", "asym_id": "R", "entry_id": "5cq2", "pdb_strand_id": "R"},
        ]
    )
    # Original config
    chains = get_data.get_interacting_chains(
        arr, backbone_only=False, contact_threshold=5.0, entities=entities
    )
    chain_dict = chains.to_dict(orient="records")[0]
    expected = {
        "chain_1": "R",
        "asym_id_1": "R",
        "chain_2": "L",
        "asym_id_2": "L",
        "pdb_strand_id_1": "R",
        "pdb_strand_id_2": "L",
        "entry_id": "5cq2",
        # "n_atom_pairs": 29,
        "n_residue_pairs": 3,
        "n_residues": 4,
        "buried_sasa": 169,
        "chain_1_residues": "31",
        "chain_2_residues": "4,5,6",
    }
    for k, v in chain_dict.items():
        assert expected[k] == v

    chains = get_data.get_interacting_chains(
        arr, contact_threshold=10.0, entities=entities
    )
    chain_dict = chains.to_dict(orient="records")[0]
    expected = {
        "chain_1": "R",
        "asym_id_1": "R",
        "chain_2": "L",
        "asym_id_2": "L",
        "pdb_strand_id_1": "R",
        "pdb_strand_id_2": "L",
        "entry_id": "5cq2",
        "n_residue_pairs": 31,
        "n_residues": 15,
        "buried_sasa": 169,
        "chain_1_residues": "27,28,29,30,31,32,33",
        "chain_2_residues": "2,3,4,5,6,7,8,9",
    }
    for k, v in chain_dict.items():
        assert expected[k] == v


@pytest.mark.parametrize(
    "test_mmcif,radius,backbone_only,expect_equal",
    [
        ("cm/pdb_00007cm8/7cm8-assembly.cif", 5.0, False, True),
        ("ww/pdb_00004wwi/4wwi-assembly.cif", 5.0, False, True),
        ("ww/pdb_00006wwe/6wwe-assembly.cif", 5.0, False, True),
        ("cm/pdb_00007cm8/7cm8-assembly.cif", 5.0, True, False),
        ("cm/pdb_00007cm8/7cm8-assembly.cif", 10.0, False, False),
        ("cm/pdb_00007cm8/7cm8-assembly.cif", 10.0, True, False),
    ],
)
def test_interacting_chains_method_regression(
    test_mmcif, radius, backbone_only, expect_equal, pinder_data_cp
):
    from biotite.structure.io.pdbx import PDBxFile, get_structure
    import biotite.structure as struc

    next_gen = pinder_data_cp / "nextgen_rcsb"
    cif_file = next_gen / test_mmcif
    gz_cif = cif_file.parent / f"{cif_file.parent.stem}_xyz-enrich.cif.gz"

    pdbx_file = get_data.read_mmcif_file(gz_cif)
    pdb_id = gz_cif.stem.split("_")[1][-4:].lower()
    bio_asm, entity_map = get_data.generate_bio_assembly(gz_cif)
    bio_asm = bio_asm[struc.filter_amino_acids(bio_asm)].copy()
    entities = get_data.get_entities(pdbx_file, entry_id=pdb_id)
    structure_chains = get_data.get_structure_chains(bio_asm)
    chain_mapping = {}
    unique_uniprot_acc = set()
    chains = []
    for entity_id in entities.entity_id.unique():
        entity_mapping = get_data.sequence_mapping(
            pdbx_file, entry_id=pdb_id, entity_id=entity_id
        )
        # print("Entity mapping", entity_mapping.shape)
        # print("Asyms", entities.query("entity_id == @entity_id").asym_id.unique())
        for asym_id in entities.query("entity_id == @entity_id").asym_id.unique():
            mapping = entity_mapping.query("asym_id == @asym_id")
            uniprot = get_data.infer_uniprot_from_mapping(mapping)
            unique_uniprot_acc.add(uniprot)
            for chain in structure_chains:
                if chain.startswith(asym_id):
                    # entity_mapping["monomer_id"] = str(monomer)
                    # entity_mapping["chain_file"] = side
                    chain_mapping[chain] = mapping.copy()
                    chain_mapping[chain].loc[:, "chain"] = chain
                    chains.append(
                        {
                            "asym_id": asym_id,
                            "chain": chain,
                            "uniprot": uniprot,
                            "length_resolved": chain_mapping[chain].resolved.sum(),
                        }
                    )

    entities = pd.merge(entities, pd.DataFrame(chains), on="asym_id", how="right")

    interacting_v1 = get_data.get_interacting_chains(
        bio_asm,
        entities=entities,
        contact_threshold=5.0,
        backbone_only=False,
    )
    interacting_v2 = get_data.get_interacting_chains(
        bio_asm,
        entities=entities,
        contact_threshold=radius,
        backbone_only=backbone_only,
    )
    assert interacting_v1.equals(interacting_v2) == expect_equal


def test_get_metadata(pinder_data_cp):
    cif_filename = "pdb_00007cm8_xyz-enrich.cif.gz"
    mmcif_file = pinder_data_cp / "nextgen_rcsb/cm/pdb_00007cm8" / cif_filename
    pdbx_file = get_data.read_mmcif_file(mmcif_file)
    metadata = get_data.get_metadata(pdbx_file)
    assert isinstance(metadata, pd.DataFrame)
    assert metadata.shape == (1, 12)
    record = metadata.to_dict(orient="records")[0]
    expected = {
        "entry_id": "7CM8",
        "method": "X-RAY DIFFRACTION",
        "date": "2020-07-25",
        "release_date": "2020-08-19",
        "resolution": "1.90",
        "assembly": "1",
        "assembly_details": "author_and_software_defined_assembly",
        "oligomeric_details": "dimeric",
        "oligomeric_count": "2",
        "biol_details": None,
        "complex_type": None,
        "status": None,
    }
    for k, v in expected.items():
        assert record[k] == v


@pytest.mark.parametrize(
    "pdb_id, expected_result",
    [
        (
            "7cm8",
            {
                "entry_id": "7cm8",
                "method": "X-RAY DIFFRACTION",
                "date": "2020-07-25",
                "release_date": "2020-08-19",
                "resolution": 1.9,
                "assembly": 1,
                "assembly_details": "author_and_software_defined_assembly",
                "oligomeric_details": "dimeric",
                "oligomeric_count": 2,
                "biol_details": None,
                "complex_type": "homomer",
                "status": "complete",
            },
        ),
        (
            "vant",
            None,
        ),
    ],
)
def test_process_mmcif(pinder_data_cp, pdb_id, expected_result):
    cif_filename = "pdb_0000" + pdb_id + "_xyz-enrich.cif.gz"
    two_letter_code = pdb_id[1:3]
    mmcif_file = (
        pinder_data_cp
        / "nextgen_rcsb"
        / two_letter_code
        / ("pdb_0000" + pdb_id)
        / cif_filename
    )
    metadata_file = mmcif_file.parent / f"{pdb_id}-metadata.tsv"
    if expected_result is not None:
        get_data.process_mmcif(mmcif_file)
        assert metadata_file.is_file()
        metadata = pd.read_csv(metadata_file, sep="\t")

        metadata = metadata.replace({float("NaN"): None})
        metadata_dict = metadata.to_dict(orient="records")[0]
        for k, v in expected_result.items():
            assert metadata_dict[k] == v
    else:
        with pytest.raises(Exception):
            get_data.process_mmcif(mmcif_file)


def test_check_assembly(pinder_data_cp):
    import numpy as np
    from biotite.structure.io import pdbx
    from biotite.structure import filter_amino_acids

    cif_filename = "pdb_00007cm8_xyz-enrich.cif.gz"
    mmcif_file = pinder_data_cp / "nextgen_rcsb/cm/pdb_00007cm8" / cif_filename
    get_data.process_mmcif(mmcif_file)
    assembly_cif_path = mmcif_file.parent / "7cm8-assembly.cif"
    pdbx_file = atoms.biotite_pdbxfile().read(assembly_cif_path)
    model = pdbx.get_structure(pdbx_file, model=1, use_author_fields=False)
    model = model[filter_amino_acids(model)]
    assert sorted(np.unique(model.chain_id)) == ["A1", "A2"]
    atoms.cif_to_pdb(
        assembly_cif_path, assembly_cif_path.parent / "test-A1.pdb", chains={"A1": "A"}
    )
    pdb_model = atoms.atom_array_from_pdb_file(assembly_cif_path.parent / "test-A1.pdb")
    pdb_model = pdb_model[filter_amino_acids(pdb_model)]
    assert list(np.unique(pdb_model.chain_id)) == ["A"]
    atoms.cif_to_pdb(
        assembly_cif_path, assembly_cif_path.parent / "test-A2.pdb", chains={"A1": "A"}
    )
    pdb_model = atoms.atom_array_from_pdb_file(assembly_cif_path.parent / "test-A2.pdb")
    pdb_model = pdb_model[filter_amino_acids(pdb_model)]
    assert list(np.unique(pdb_model.chain_id)) == ["A"]


def test_create_fasta_from_systems(tmp_path):
    from pinder.core import PinderSystem

    output_file = tmp_path / "fasta_output.fasta"
    pinder_ids = [
        "1df0__A1_Q07009--1df0__B1_Q64537",
        "5e6u__A1_P20701--5e6u__B1_P05107",
        "5h5q__A1_P36969--5h5q__B1_UNDEFINED",
        "8i2f__A1_O34841--8i2f__B1_P54421",
    ]
    systems = [PinderSystem(pid) for pid in pinder_ids]
    create_fasta_from_systems(systems, output_file)
    assert output_file.is_file()


@pytest.mark.parametrize(
    "test_mmcif,n_asm_chains,n_asym_ids,entity_ids",
    [
        # Example where asym has 15 chains but assembly has 7
        ("bd/pdb_00007bdu/pdb_00007bdu_xyz-enrich.cif.gz", 7, 7, {"1", "2", "3"}),
        ("km/pdb_00007kmx/pdb_00007kmx_xyz-enrich.cif.gz", 840, 14, {"1", "2"}),
        (
            "a7/pdb_00002a79/pdb_00002a79_xyz-enrich.cif.gz",
            34,
            13,
            {"1", "2", "3", "4", "5", "6", "7"},
        ),
        (
            "rw/pdb_00006rw4/pdb_00006rw4_xyz-enrich.cif.gz",
            125,
            125,
            {
                "20",
                "29",
                "18",
                "24",
                "13",
                "4",
                "12",
                "1",
                "6",
                "2",
                "15",
                "16",
                "34",
                "42",
                "35",
                "36",
                "38",
                "3",
                "7",
                "23",
                "37",
                "21",
                "9",
                "14",
                "10",
                "19",
                "5",
                "8",
                "40",
                "22",
                "30",
                "28",
                "32",
                "39",
                "31",
                "41",
                "27",
                "26",
                "25",
                "11",
                "33",
                "17",
            },
        ),
        ("y2/pdb_00002y26/pdb_00002y26_xyz-enrich.cif.gz", 120, 40, {"1", "2"}),
    ],
)
def test_generate_bio_assembly(
    test_mmcif, n_asm_chains, n_asym_ids, entity_ids, pinder_data_cp
):
    next_gen = pinder_data_cp / "nextgen_rcsb"
    gz_cif = next_gen / test_mmcif
    bio_asm, entity_map = get_data.generate_bio_assembly(gz_cif)
    asm_ch = set(bio_asm.chain_id)
    map_ch = set(entity_map.chain_id)
    asym_id = set(entity_map.asym_id)
    assert len(asm_ch.intersection(map_ch)) == len(asm_ch)
    assert len(asym_id) == n_asym_ids
    assert len(asm_ch) == n_asm_chains
    asm_entities = set(entity_map.entity_id)
    assert asm_entities == entity_ids


def test_process_monomer_mmcif(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb/bo/pdb_00001bo0"
    mmcif_file = data_dir / "pdb_00001bo0_xyz-enrich.cif.gz"
    get_data.process_mmcif(mmcif_file)
