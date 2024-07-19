import pytest
import pandas as pd
from pinder.data import apo_utils, config, get_apo
from pinder.core.loader.structure import Structure


def test_valid_structure(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/fa/pdb_00008fav"
    apo_id = "8fav__A1_P51449"
    pdb_file = pdb_dir / f"{apo_id}.pdb"
    assert apo_utils.valid_structure(pdb_file)


def test_sufficient_atom_types(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/fa/pdb_00008fav"
    apo_id = "8fav__A1_P51449"
    pdb_file = pdb_dir / f"{apo_id}.pdb"
    apo_struct = Structure(pdb_file)
    min_atom_types = 5000
    sufficient = apo_utils.sufficient_atom_types(apo_struct, min_atom_types)
    assert not sufficient


def test_sufficient_residues(pinder_data_cp):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/fa/pdb_00008fav"
    apo_id = "8fav__A1_P51449"
    pdb_file = pdb_dir / f"{apo_id}.pdb"
    apo_struct = Structure(pdb_file)
    min_residues = 5000
    sufficient = apo_utils.sufficient_residues(apo_struct, min_residues)
    assert not sufficient
    min_residues = 50
    sufficient = apo_utils.sufficient_residues(apo_struct, min_residues)
    assert sufficient


def test_holo_apo_seq_identity(pinder_data_cp):
    two_char_dir = pinder_data_cp / "nextgen_rcsb/fa"
    holo_pdb = two_char_dir / "pdb_00001fa8/1fa8__A1_P0AC81-R.pdb"
    apo_pdb = two_char_dir / "pdb_00008fav/8fav__A1_P51449.pdb"
    holo_seq = Structure(holo_pdb).sequence
    apo_seq = Structure(apo_pdb).sequence
    seq_ident = apo_utils.holo_apo_seq_identity(holo_seq, apo_seq)
    expected = {
        "holo_sequence": holo_seq,
        "apo_sequence": apo_seq,
        "sequence_identity": pytest.approx(0.0234375),
    }
    for k, v in seq_ident.items():
        assert expected[k] == v


@pytest.mark.parametrize(
    "min_atom_types, min_residues, valid_as_apo",
    [
        (3, 5, True),
        (5000, 5, False),
        (3, 5000, False),
    ],
)
def test_validate_apo_monomer(
    min_atom_types, min_residues, valid_as_apo, pinder_data_cp
):
    pdb_dir = pinder_data_cp / "nextgen_rcsb/fa/pdb_00008fav"
    apo_id = "8fav__A1_P51449"
    cfg = config.ApoPairingConfig(
        min_atom_types=min_atom_types, min_residues=min_residues
    )
    apo_info = apo_utils.validate_apo_monomer(
        apo_id=apo_id,
        pdb_dir=pdb_dir,
        config=cfg,
    )
    assert apo_info["valid_as_apo"] == valid_as_apo


def test_get_valid_apo_monomer_ids(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    get_apo.get_valid_apo_monomer_ids(
        pinder_dir=pinder_dir,
        config=config.ApoPairingConfig(),
        max_workers=2,
        use_cache=True,
        remove_chain_copies=True,
        parallel=True,
    )


def test_remove_apo_chain_copies():
    import random

    chains = [
        "A",
        "B",
        "C",
        "ab",
        "BA",
    ]
    monomer_test_df = []
    for ch in chains:
        monomer_test_df.append({"chain": f"{ch}1"})
        monomer_test_df.append({"chain": f"{ch}{random.randint(2, 11)}"})
    monomer_test_df = pd.DataFrame(monomer_test_df)
    dedupe = apo_utils.remove_apo_chain_copies(monomer_test_df)
    assert len(set(dedupe.chain)) == len(set(chains))
    assert (
        len(set(monomer_test_df.chain).intersection(set(dedupe.chain)))
        == dedupe.shape[0]
    )


def test_remove_dimer_chain_copies():
    chain_pairs = [
        ("A1", "A2"),
        ("A1", "A3"),
        ("A2", "A3"),
        ("A1", "B1"),
        ("A1", "B2"),
        ("AB1", "AB2"),
        ("AB2", "AC1"),
    ]
    dimer_test_df = pd.DataFrame(chain_pairs, columns=["chain_R", "chain_L"])
    dimer_test_df.loc[:, "chain_pair"] = (
        dimer_test_df.chain_R + "_" + dimer_test_df.chain_L
    )
    dedupe = apo_utils.remove_dimer_chain_copies(dimer_test_df)
    assert len(set(dedupe.chain_pair)) == 6
    assert len(set(dedupe.chain_R)) == 3
    assert len(set(dedupe.chain_L)) == 6
    assert dedupe.shape == (6, 6)


def test_get_putative_pairings(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    RL = get_apo.get_putative_pairings(
        pinder_dir=pinder_dir,
        use_cache=True,
        remove_chain_copies=False,
    )


def test_get_apo_pairing_metrics_for_id(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    apo_dir = pinder_dir / "apo_metrics"
    RL = get_apo.get_putative_pairings(
        pinder_dir=pinder_dir, use_cache=True, remove_chain_copies=False
    )
    df = RL.query('id == "6wwe__A1_Q2XVP4--6wwe__C1_L0N7N1"').reset_index(drop=True)
    metrics = apo_utils.get_apo_pairing_metrics_for_id(
        df,
        pdb_dir=pinder_dir / "pdbs",
        config=config.ApoPairingConfig(align_method="biotite"),
    )
    expected = {
        "id": "6wwe__A1_Q2XVP4--6wwe__C1_L0N7N1",
        "apo_monomer_id": "4ozq__A1_L0N7N1",
        "holo_R_residues": 441,
        "holo_R_atoms": 3446,
        "holo_L_residues": 359,
        "holo_L_atoms": 2807,
        "receptor_residues": 441,
        "ligand_residues": 683,
        "receptor_atoms": 3446,
        "ligand_atoms": 5070,
        "complex_residues": 693,
        "complex_atoms": 8516,
        "receptor_missing": 0,
        "receptor_native": 60,
        "ligand_missing": 7,
        "ligand_native": 71,
        "Fmiss_R": 0.0,
        "Fmiss_L": pytest.approx(0.09859154929577464),
        "apo_R_code": "holo",
        "apo_L_code": "4ozq__A1_L0N7N1",
        "radius": 10.0,
        "backbone_only": False,
        "heavy_only": False,
        "unbound_body": "L",
        "monomer_name": "apo",
        "Fnat": pytest.approx(0.7634408602150538),
        "Fnonnat": pytest.approx(0.10353535353535354),
        "common_contacts": 355,
        "differing_contacts": 41,
        "bound_contacts": 465,
        "unbound_contacts": 396,
        "fnonnat_R": pytest.approx(0.0392156862745098),
        "fnonnat_L": pytest.approx(0.045454545454545456),
        "fnat_R": pytest.approx(0.8166666666666667),
        "fnat_L": pytest.approx(0.8873239436619719),
        "difficulty": "Rigid-body",
        "I-RMSD": pytest.approx(1.3053585290908813),
        "matched_interface_chains": 2,
        "refine_rmsd": pytest.approx(1.589128017425537),
        "raw_rmsd": pytest.approx(1.589128017425537),
        "refine_aln_ats": 2419,
        "raw_aln_ats": 2419,
        "aln_res": 320,
        "unbound_id": "6wwe__A1_Q2XVP4-R--4ozq__A1_L0N7N1",
        "holo_receptor_interface_res": 60,
        "holo_ligand_interface_res": 71,
        "apo_receptor_interface_res": 51,
        "apo_ligand_interface_res": 66,
        "holo_sequence": "SQVTVAVRVRPFSKREKTEKASQVVFTNGEEITVEHPDMKQVYSFIYDVSFWSFDECHPGYASQTTVYETLAAPLLDRAFEGYNTCLFAYGQTGSGKSYTMMGLNEEPGIIPRFCEDLFAQIAKKQTSEVSYHLEMSFFEVYNEKIHDLLVCKGENGQRKQPLRAREHPVSGPYVEGLSMNVVSSYSDIQSWLELGNKQRATAATGMNDKSSRSHSVFTLVMTQTKTEVVEGEEHDHRITSRINLVDLAGSERCSTAHSSGQRLKEGVSINKSLLTLGKVISALSEQANGKRVFIPYRESTLTWLLKESLGGNSKTAMIATVSPAASNIEETLSTLRYATQARLIVNIAKVNEDMNAKL",
        "apo_sequence": "WINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDAALAAAQTNAAAENSQVTVAVRVRPFSKREKTEKASQVVFTNGEEITVEHPDMKQVYSFIYDVSFWSFDECHPGYASQTTVYETLAAPLLDRAFEGYNTCLFAYGQTGSGKSYTMMGLNEEPGIIPRFCEDLFAQIAKKQTSEVSYHLEMSFFEVYNEKIHDLLVKQPLRAREHPVSGPYVEGLSMNVVSSYSDIQSWLELGNKQRATAKSSRSHSVFTLVMTQTKTEHDHRITSRINLVDLAGSERCSTAGQRLKEGVSINKSLLTLGKVISALSEQANGKRVFIPYRESTLTWLLKESLGGNSKTAMIATVSPAASNIEETLSTLRYATQAR",
        "sequence_identity": pytest.approx(0.8913649025069638),
    }
    assert metrics.shape == (1, 51)
    actual_dict = metrics.to_dict(orient="records")[0]
    for k, actual in actual_dict.items():
        assert (
            expected[k] == actual,
            f"Expected {k} to be {expected[k]}, got {actual}",
        )


def test_get_apo_pairing_metrics(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    apo_dir = pinder_dir / "apo_metrics"
    RL = get_apo.get_putative_pairings(
        pinder_dir=pinder_dir, use_cache=True, remove_chain_copies=False
    )
    get_apo.get_apo_pairing_metrics(
        pinder_dir=pinder_dir,
        putative_pairs=RL,
        config=config.ApoPairingConfig(),
        output_parquet=apo_dir / "pairing_metrics" / "metrics_all.parquet",
        max_workers=2,
        parallel=True,
    )


def test_collate_apo_metrics(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    apo_dir = pinder_dir / "apo_metrics"
    metric_dir = apo_dir / "pairing_metrics"
    missing_metric_dir = apo_dir / "no_metrics"
    output_pqt = apo_dir / "two_sided_apo_monomer_metrics.parquet"
    get_apo.collate_apo_metrics(
        metric_dir=missing_metric_dir, output_parquet=output_pqt
    )
    assert not output_pqt.is_file()

    get_apo.collate_apo_metrics(metric_dir=metric_dir, output_parquet=output_pqt)
    assert output_pqt.is_file()


def test_calculate_frac_monomer_dimer_overlap(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    apo_dir = pinder_dir / "apo_metrics"
    output_pqt = apo_dir / "two_sided_apo_monomer_metrics.parquet"
    pairings = pd.read_parquet(output_pqt)
    df = pairings.query('id == "7nsg__A1_P43005--7nsg__B1_P43005"').reset_index(
        drop=True
    )
    metrics = apo_utils.calculate_frac_monomer_dimer_overlap(
        df, pdb_dir=pinder_dir / "pdbs"
    )
    assert metrics.shape == (8, 5)
    assert metrics.frac_monomer_dimer_uniprot.max() == pytest.approx(0.4883721)
    assert metrics.frac_monomer_dimer.max() == pytest.approx(0.503488)
    assert metrics.query(
        'apo_monomer_id == "8cv3__A1_P43005"'
    ).frac_monomer_dimer.values[0] == pytest.approx(0.48372093)


@pytest.mark.parametrize(
    "align_method",
    [
        # TODO: Pymol is never called in CI, as it is not installed.
        # Remove it entirely or install it in CI and adapt expected values in asserts
        # "pymol",
        "biotite",
    ],
)
def test_hybrid_align(align_method, stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    pdb_dir = pinder_dir / "pdbs"
    RL = get_apo.get_putative_pairings(
        pinder_dir=pinder_dir, use_cache=True, remove_chain_copies=False
    )
    cfg = config.ApoPairingConfig(align_method=align_method)
    holo_pdb = "6wwe__C1_L0N7N1-L.pdb"
    apo_id = "4ozq__A1_L0N7N1"
    df = RL.query(
        f"holo_L_pdb == '{holo_pdb}' and apo_monomer_id == '{apo_id}'"
    ).reset_index(drop=True)
    holo_L_pdb = df.holo_L_pdb.values[0]
    holo_L = Structure(pdb_dir / f"{holo_L_pdb}")
    apo_L_pdb = pdb_dir / f"{apo_id}.pdb"
    L_chain = holo_L.chains[0]
    apo_eval_chain = L_chain
    apo_L = Structure(apo_L_pdb)
    apo_L.atom_array.chain_id[apo_L.atom_array.chain_id == cfg.apo_chain] = (
        apo_eval_chain
    )
    # Align apo_mono to holo_ref
    apo_mono, aln_metrics = apo_utils.hybrid_align(
        apo_L, holo_L, align_method=cfg.align_method
    )
    assert isinstance(apo_mono, Structure)
    assert apo_mono.atom_array.shape == apo_L.atom_array.shape
    required_keys = {
        "refine_rmsd",
        "raw_rmsd",
        "refine_aln_ats",
        "raw_aln_ats",
        "aln_res",
    }
    assert (
        set(aln_metrics.keys()) == required_keys
    ), f"Expected alignment metrics {required_keys}, got {aln_metrics.keys()}"
    assert aln_metrics["raw_aln_ats"] == 2419
    assert aln_metrics["aln_res"] == 320


def test_add_all_apo_pairings_to_index(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    get_apo.add_all_apo_pairings_to_index(
        pinder_dir,
        config=config.ApoPairingConfig(),
        parallel=True,
        max_workers=2,
    )


def test_canonical_apo(stateful_data_cp):
    pinder_dir = stateful_data_cp / "apo_holo"
    index = pd.read_parquet(pinder_dir / "index_with_apo.parquet")
    # Track expected apo-holo pairings and canonical apo assignment from test sample
    # based on what was assigned in earlier versions of the dataset.
    expected_canon = {
        # The apo scores for these two are very close.
        # Depending on whether the pymol vs. biotite alignment method is used,
        # the score for 1s82 vs. 1s85 can be higher or lower.
        # 1s85 is selected if pymol alignment method is used, 1s82 if biotite.
        "1ldt__A1_P00761--1ldt__B1_P80424": {"R": {"1s84"}, "L": "2kmo"},
        "7nsg__A1_P43005--7nsg__C1_P43005": {"R": "6x2z", "L": "8cuj"},
        "7nsg__B1_P43005--7nsg__C1_P43005": {"R": "8cuj", "L": "6x2z"},
        "7nsg__A1_P43005--7nsg__B1_P43005": {"R": "8cuj", "L": "6x2z"},
        "4wwi__D1_P01860--4wwi__A1_P38507": {"L": "4zmd"},
        "6wwe__A1_Q2XVP4--6wwe__C1_L0N7N1": {"L": "4ozq"},
        "6wwe__B1_A0A287AZ37--6wwe__C1_L0N7N1": {"L": "4ozq"},
        "6wwe__A1_Q2XVP4--6wwe__B1_A0A287AZ37": {},
    }
    expected_pairs = {
        "1ldt__A1_P00761--1ldt__B1_P80424": {
            "R": {"1s84", "1s81", "1fni", "1s83"},
            "L": {"2kmq", "2kmp", "2kmo", "2kmr"},
        },
        "7nsg__A1_P43005--7nsg__C1_P43005": {
            "R": {"8cv3", "6x2z", "8cud", "8cuj"},
            "L": {"8cv3", "6x2z", "8cud", "8cuj"},
        },
        "7nsg__B1_P43005--7nsg__C1_P43005": {
            "R": {"8cv3", "8cuj", "8cud", "6x2z"},
            "L": {"8cv3", "8cuj", "8cud", "6x2z"},
        },
        "7nsg__A1_P43005--7nsg__B1_P43005": {
            "R": {"8cv3", "8cuj", "8cud", "6x2z"},
            "L": {"8cv3", "8cuj", "8cud", "6x2z"},
        },
        "4wwi__D1_P01860--4wwi__A1_P38507": {
            "L": {"1edi", "5cbo", "1edk", "4npf", "1edl", "1edj", "2spz", "4zmd"},
        },
        "6wwe__A1_Q2XVP4--6wwe__C1_L0N7N1": {"L": {"4ozq"}},
        "6wwe__B1_A0A287AZ37--6wwe__C1_L0N7N1": {"L": {"4ozq"}},
        "6wwe__A1_Q2XVP4--6wwe__B1_A0A287AZ37": {},
    }
    for pinder_id in expected_canon:
        R_canon = expected_canon[pinder_id].get("R")
        L_canon = expected_canon[pinder_id].get("L")
        R_pairs = expected_pairs[pinder_id].get("R")
        L_pairs = expected_pairs[pinder_id].get("L")
        entries = index.query(f'id == "{pinder_id}"').to_dict(orient="records")
        for entry in entries:
            apo_R_code = entry["apo_R_pdb"].split("__")[0]
            apo_L_code = entry["apo_L_pdb"].split("__")[0]
            apo_R_codes = {p.split("__")[0] for p in entry["apo_R_pdbs"].split(";")}
            apo_L_codes = {p.split("__")[0] for p in entry["apo_L_pdbs"].split(";")}
            if R_canon or apo_R_code != "":
                fail_msg = f"Expected canonical apo R for {pinder_id} was {R_canon}, got {apo_R_code}"
                if isinstance(R_canon, set):
                    match = apo_R_code in R_canon
                else:
                    match = apo_R_code == R_canon
                assert match, fail_msg
            if L_canon or apo_L_code != "":
                fail_msg = f"Expected canonical apo L for {pinder_id} was {L_canon}, got {apo_L_code}"
                if isinstance(L_canon, set):
                    match = apo_L_code in L_canon
                else:
                    match = apo_L_code == L_canon
                assert match, fail_msg
            if R_pairs or apo_R_codes != {""}:
                missing = R_pairs - apo_R_codes or 0
                extra = apo_R_codes - R_pairs or 0
                assert (
                    apo_R_codes == R_pairs
                ), f"Got {extra} extra, {missing} missing, apo R pairs for {pinder_id}"
            if L_pairs or apo_L_codes != {""}:
                if pinder_id == "4wwi__D1_P01860--4wwi__A1_P38507":
                    # This is a case where biotite alignment fails.
                    # The listed codes above are all valid and get marked as such if pymol alignment is used.
                    # Otherwise, the following are missing.
                    method_diff_codes = {"1edl", "1edk", "5cbo", "1edi", "1edj"}
                    L_pairs = L_pairs - method_diff_codes
                    apo_L_codes = apo_L_codes - method_diff_codes
                missing = L_pairs - apo_L_codes or 0
                extra = apo_L_codes - L_pairs or 0
                assert (
                    apo_L_codes == L_pairs
                ), f"Got {extra} extra, {missing} missing, apo L pairs for {pinder_id}"
