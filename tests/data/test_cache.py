import pytest
from pinder.data.pipeline import cache, tasks


def test_get_uningested_mmcif(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    cif_files = tasks.cif_glob(data_dir)
    expected_complete = {
        "pdb_00006wwe_xyz-enrich.cif.gz",
        "pdb_00004wwi_xyz-enrich.cif.gz",
        "pdb_00001fa8_xyz-enrich.cif.gz",
        "pdb_00007nsg_xyz-enrich.cif.gz",
        "pdb_00007cm8_xyz-enrich.cif.gz",
        "pdb_00007cma_xyz-enrich.cif.gz",
    }
    uningested = cache.get_uningested_mmcif(cif_files)
    ingested = set(cif_files) - set(uningested)
    ingested_names = {f.name for f in ingested}
    assert ingested_names == expected_complete


@pytest.mark.parametrize(
    "step_name, run_specific_step, skip_specific_step, should_skip",
    [
        ("ingest_rcsb_files", "ingest_rcsb_files", "", False),
        ("ingest_rcsb_files", "", "ingest_rcsb_files", True),
        ("get_pisa_annotations", "", "", False),
        ("get_pisa_annotations", "ingest_rcsb_files", "", True),
    ],
)
def test_skip_step(step_name, run_specific_step, skip_specific_step, should_skip):
    would_skip = cache.skip_step(step_name, run_specific_step, skip_specific_step)
    assert would_skip == should_skip


def test_get_pisa_unannotated(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    cif_files = tasks.cif_glob(data_dir)
    pisa_unannotated = cache.get_pisa_unannotated(cif_files)
    annotated_cif = set(cif_files) - set(pisa_unannotated)
    annotated_cif_names = {cif.stem for cif in annotated_cif}
    assert annotated_cif_names == {
        "pdb_00004wwi_xyz-enrich.cif",
        "pdb_00006wwe_xyz-enrich.cif",
    }


def test_get_rcsb_unannotated(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    pdb_ids = tasks.pdb_id_glob(data_dir)
    pinder_dir = pinder_data_cp / "pinder"
    assert set(pdb_ids) == set(cache.get_rcsb_unannotated(pdb_ids, pinder_dir))


def test_get_unannotated_dimer_pdbs(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    dimer_pdbs = tasks.dimer_glob(data_dir)
    expected_annotated = {
        "6wwe__A1_Q2XVP4--6wwe__B1_A0A287AZ37",
        "6wwe__A1_Q2XVP4--6wwe__C1_L0N7N1",
        "6wwe__B1_A0A287AZ37--6wwe__C1_L0N7N1",
        "7cm8__A1_P45040--7cm8__A2_P45040",
    }
    unannotated = {p.stem for p in cache.get_unannotated_dimer_pdbs(dimer_pdbs)}
    actual_annotated = {p.stem for p in dimer_pdbs} - unannotated
    assert actual_annotated == expected_annotated


def test_get_dimer_pdbs_missing_foldseek_contacts(pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    dimer_pdbs = tasks.dimer_glob(data_dir)
    missing = cache.get_dimer_pdbs_missing_foldseek_contacts(
        dimer_pdbs, config_hash="foo"
    )
    assert set(dimer_pdbs) == set(missing)
