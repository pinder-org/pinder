from __future__ import annotations
import logging
import os
import re
import shutil
from itertools import repeat
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from google.auth.exceptions import DefaultCredentialsError

from pinder.core.index.id import Dimer, Monomer, Protein
from pinder.core.index.utils import MetadataEntry, IndexEntry, fix_index, fix_metadata
from pinder.core.structure.atoms import cif_to_pdb
from pinder.core.loader.structure import (
    Structure,
    find_potential_interchain_bonded_atoms,
)
from pinder.core.utils import setup_logger
from pinder.core.utils.cloud import Gsutil
from pinder.core.utils.paths import empty_file, parallel_copy_files
from pinder.core.utils.process import process_map, process_starmap
from pinder.data.csv_utils import parallel_read_csvs, read_csv_non_default_na
from pinder.data.annotation.sabdab import add_sabdab_annotations
from pinder.data.annotation.contact_classification import detect_disulfide_bonds
from pinder.data.config import TransientInterfaceConfig
from pinder.data.pipeline.constants import UNIPROT_UNDEFINED


log = setup_logger(__name__, log_level=logging.DEBUG)


def _path_to_pdb_chain_chain(x: str) -> str:
    dimer = Dimer.from_string(x)
    monomer1 = dimer.monomer1
    monomer2 = dimer.monomer2
    pdb = x.split("_")[0]
    return f"{pdb}_{monomer1.proteins[0].chain}_{monomer2.proteins[0].chain}"


def primary_dimer_index_from_dimers(dimers: list[Dimer]) -> pd.DataFrame:
    index_dicts = []
    for dimer in dimers:
        prot_R = dimer.monomer1.proteins[0]
        prot_L = dimer.monomer2.proteins[0]
        ch_R = prot_R.chain
        ch_L = prot_L.chain
        pdb_id = prot_R.source
        pdb_ch_ch = pdb_id + "_" + ch_R + "_" + ch_L
        entry = {
            "id": str(dimer),
            "pdb_id": pdb_id,
            "chain_R": ch_R,
            "uniprot_R": prot_R.uniprot,
            "chain_L": ch_L,
            "uniprot_L": prot_L.uniprot,
            "holo_R_pdb": f"{Monomer(dimer.monomer1.proteins, side='R')}.pdb",
            "holo_L_pdb": f"{Monomer(dimer.monomer2.proteins, side='L')}.pdb",
            "holo_R": True,
            "holo_L": True,
            "pdb_chain_chain": pdb_ch_ch,
        }
        index_dicts.append(entry)
    df_index = pd.DataFrame(index_dicts)
    return df_index


def merge_index_and_entities(index_df: pd.DataFrame, pinder_dir: Path) -> pd.DataFrame:
    # Add length_resolved_1 and length_resolved_2 which is only in entities tsv
    entity_pqt = pinder_dir / "entity_metadata.parquet"
    if not entity_pqt.is_file():
        log.warning(f"Entity metadata does not exist! {entity_pqt}")
        return index_df

    entity_df = pd.read_parquet(entity_pqt)
    index_df = (
        index_df.merge(
            entity_df[["entry_id", "chain", "length_resolved"]],
            left_on=["entry_id", "chain_R"],
            right_on=["entry_id", "chain"],
            how="left",
        )
        .drop("chain", axis=1)
        .rename({"length_resolved": "length_resolved_1"}, axis=1)
    )
    index_df = (
        index_df.merge(
            entity_df[["entry_id", "chain", "length_resolved"]],
            left_on=["entry_id", "chain_L"],
            right_on=["entry_id", "chain"],
            how="left",
        )
        .drop("chain", axis=1)
        .rename({"length_resolved": "length_resolved_2"}, axis=1)
    )
    return index_df


def validate_schemas(df_index: pd.DataFrame, df_metadata: pd.DataFrame) -> None:
    rec_index = df_index.to_dict(orient="records")
    for rec in tqdm(rec_index):
        _ = IndexEntry(**rec)
    del rec_index
    log.info("Index schema validated")

    rec_metadata = df_metadata.to_dict(orient="records")
    for rec in tqdm(rec_metadata):
        try:
            _ = MetadataEntry(**rec)
        except Exception as e:
            log.error(f"Error validating metadata record: {e}")
            log.error(f"Record: {rec}")
    del rec_metadata
    log.info("Metadata schema validated")


def summarize_putative_apo_pred_counts(pinder_dir: Path) -> None:
    pdb_dtypes = {"pdb_id": "str", "entry_id": "str"}
    monomer_ids = pd.read_parquet(pinder_dir / "monomer_ids.parquet")
    monomer_ids.loc[:, "dimer_monomer"] = monomer_ids.id.str.endswith(
        "-L"
    ) | monomer_ids.id.str.endswith("-R")
    apo_counts = {
        rec["uniprot"]: rec["size"]
        for rec in (
            monomer_ids.query("~dimer_monomer")
            .groupby("uniprot", as_index=False)
            .size()
            .to_dict(orient="records")
        )
    }
    index = read_csv_non_default_na(pinder_dir / "index.1.csv.gz", dtype=pdb_dtypes)
    dimer_df = index[["id", "pdb_id", "uniprot_R", "uniprot_L"]].copy()
    dimer_df.loc[:, "putative_apo_R_count"] = dimer_df.uniprot_R.apply(
        lambda uni: apo_counts.get(uni, 0)
    )
    dimer_df.loc[:, "putative_apo_L_count"] = dimer_df.uniprot_L.apply(
        lambda uni: apo_counts.get(uni, 0)
    )
    pdb_dir = pinder_dir / "pdbs"
    af_unis = {p.stem.split("__")[1] for p in pdb_dir.glob("af_*.pdb")}
    dimer_df.loc[:, "pred_R_count"] = dimer_df.uniprot_R.apply(
        lambda uni: int(uni in af_unis)
    )
    dimer_df.loc[:, "pred_L_count"] = dimer_df.uniprot_L.apply(
        lambda uni: int(uni in af_unis)
    )
    dimer_df.to_parquet(pinder_dir / "apo_pred_counts_by_uniprot.parquet", index=False)


def merge_metadata(
    dimers: list[Dimer],
    pinder_dir: Path,
) -> None:
    """Merge metadata and annotations from dimers into a single csv file

    Parameters
    ----------
    dimers : list[Dimer]
        List of Dimer objects to be processed.
    pinder_dir : Path
        Path to the directory where the output csv file will be saved.

    Returns
    -------
    None
    """

    df_index = primary_dimer_index_from_dimers(dimers)

    # basic metadata for each PDB entity
    df_1 = pd.read_parquet(pinder_dir / "structural_metadata.parquet")
    # interfaces == interacting chains
    df_2 = pd.read_parquet(pinder_dir / "interfaces.parquet")
    df_2["pdb_chain_chain"] = df_2.entry_id + "_" + df_2.chain_1 + "_" + df_2.chain_2
    df_2.drop_duplicates(subset=["pdb_chain_chain"], inplace=True)
    df_2.reset_index(drop=True, inplace=True)

    # interface annotations for each dimer
    df_3 = pd.read_parquet(pinder_dir / "interface_annotations.parquet")
    df_3["pdb_chain_chain"] = df_3.path.apply(_path_to_pdb_chain_chain)
    df_3.reset_index(drop=True, inplace=True)

    df_merged = df_index.merge(df_1, left_on="pdb_id", right_on="entry_id", how="inner")
    # Add length_resolved_1 and length_resolved_2 which is only in entities tsv
    df_merged_2 = merge_index_and_entities(df_merged, pinder_dir)
    df_merged_3 = df_merged_2.merge(df_2, on="pdb_chain_chain", how="inner")
    df_merged_4 = df_merged_3.merge(df_3, on="pdb_chain_chain", how="inner")

    # Get the fields from the pydantic models
    index_fields = [field for field in IndexEntry.__annotations__.keys()]
    metadata_fields = [field for field in MetadataEntry.__annotations__.keys()]

    # Create dataframes with matching columns
    supp_idx_cols = ["pdb_chain_chain"]
    df_index = df_merged_4[
        list(set(df_merged_4.columns) & set(index_fields).union(set(supp_idx_cols)))
    ].copy()

    # Supplementary columns that are used in intermediate processing
    # but dropped from final metadata
    supp_meta_cols = ["pdb_chain_chain", "chain_1_residues", "chain_2_residues"]
    df_metadata = df_merged_4[
        list(set(df_merged_4.columns) & set(metadata_fields).union(set(supp_meta_cols)))
    ].copy()

    df_index = fix_index(df_index, index_fields)
    df_metadata = fix_metadata(df_metadata, metadata_fields)
    # Reorder columns
    df_index = df_index[index_fields + supp_idx_cols]
    df_metadata = df_metadata[metadata_fields + supp_meta_cols]

    # Validate schemas
    validate_schemas(df_index[index_fields], df_metadata[metadata_fields])

    # Save to CSV
    df_index.to_csv(pinder_dir / "index.1.csv.gz", index=False)
    df_metadata.to_csv(pinder_dir / "metadata.1.csv.gz", index=False)


def cast_resi_to_valid_str(resi: str | int | float) -> str:
    try:
        # Raises TypeError for strings like '100B'
        is_na = pd.isna(resi)
    except TypeError:
        is_na = False
    if is_na:
        resi_str = ""
    else:
        # First try to cast to float. If successful, then we can go back to int and then str
        # If it wasn't, it means we have other characters like 101A
        # In that case, we can just treat it as its original string
        try:
            resi_str = str(int(float(resi)))
        except ValueError:
            resi_str = str(resi)
    return resi_str


def load_mapping_chains(pqt_file: Path) -> dict[str, str | int | None]:
    mapping_dict: dict[str, str | int | None] = {}
    try:
        mapping = pd.read_parquet(pqt_file)
        # Appears that some entries with UNDEFINED uniprot didn't get set
        # correctly upstream
        if "resi_uniprot" not in mapping.columns:
            mapping.loc[:, "resi_uniprot"] = ""
            mapping.to_parquet(pqt_file, index=False)

        mapping_rec = (
            mapping[["asym_id", "entity_id", "pdb_strand_id", "chain"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        if len(mapping_rec) > 1:
            log.warning(f"WARNING: more than one chain found for {pqt_file.name}!")
        mapping_dict = mapping_rec[0]
        # Store comma-separated list of residues in the chain while preserving order and filling
        # NaN / unparseable values with empty str so that mapping order is preserved.
        for resi_col in ["resi", "resi_pdb", "resi_auth", "resi_uniprot"]:
            # Note: this strips any undesired float conversions of residues but keeps alternative
            # numberings from e.g. resi_auth like 101A
            resi_str = ",".join(
                list(map(cast_resi_to_valid_str, list(mapping[resi_col])))
            )
            mapping_dict[resi_col] = resi_str
        mapping_dict["mapping_pqt"] = str(pqt_file)
    except Exception as e:
        log.error(f"WARNING: failed to process mapping {pqt_file}: {e}")
        mapping_dict = {
            "asym_id": None,
            "entity_id": None,
            "pdb_strand_id": None,
            "chain": pqt_file.stem.split("__")[1].split("_")[0],
            "resi": "",
            "resi_pdb": "",
            "resi_auth": "",
            "resi_uniprot": "",
            "mapping_pqt": str(pqt_file),
        }
    return mapping_dict


def collate_chain_info(
    pinder_dir: Path,
    max_workers: int | None = None,
    parallel: bool = True,
) -> None:
    index = read_csv_non_default_na(
        pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
    )
    # Construct unique set of PDBs so mappings only read once
    holo_R = set(index.holo_R_pdb)
    holo_L = set(index.holo_L_pdb)
    holo_R_index = []
    for pdb_file in holo_R:
        r_name = pdb_file.split(".pdb")[0]
        pqt = pinder_dir / f"mappings/{r_name}.parquet"
        holo_R_index.append({"holo_R_pdb": pdb_file, "mapping_pqt": pqt})
    holo_L_index = []
    for pdb_file in holo_L:
        l_name = pdb_file.split(".pdb")[0]
        pqt = pinder_dir / f"mappings/{l_name}.parquet"
        holo_L_index.append({"holo_L_pdb": pdb_file, "mapping_pqt": pqt})
    holo_L_pqts = pd.DataFrame(holo_L_index)
    holo_R_pqts = pd.DataFrame(holo_R_index)
    mapping_pqts = set(holo_L_pqts.mapping_pqt).union(set(holo_R_pqts.mapping_pqt))
    chain_info = process_map(
        load_mapping_chains, mapping_pqts, parallel=parallel, max_workers=max_workers
    )

    # Tracks mapping CSV file which we will merge with the file index dfs above
    chain_info = pd.DataFrame(chain_info)
    holo_R_pqts.mapping_pqt = holo_R_pqts.mapping_pqt.astype(str)
    holo_L_pqts.mapping_pqt = holo_L_pqts.mapping_pqt.astype(str)
    R_chains = pd.merge(holo_R_pqts, chain_info, how="left")
    L_chains = pd.merge(holo_L_pqts, chain_info, how="left")
    R_chains = pd.merge(index[["id", "holo_R_pdb"]], R_chains, how="left")
    L_chains = pd.merge(index[["id", "holo_L_pdb"]], L_chains, how="left")
    R_chains.drop("mapping_pqt", axis=1, inplace=True)
    L_chains.drop("mapping_pqt", axis=1, inplace=True)
    rename_cols = [
        "asym_id",
        "entity_id",
        "pdb_strand_id",
        "chain",
        "resi",
        "resi_pdb",
        "resi_auth",
        "resi_uniprot",
    ]
    R_chains.rename({c: f"{c}_R" for c in rename_cols}, axis=1, inplace=True)
    L_chains.rename({c: f"{c}_L" for c in rename_cols}, axis=1, inplace=True)
    chain_meta = pd.merge(R_chains, L_chains, how="left")
    col_order = ["id"]
    for c in rename_cols:
        col_order.append(f"{c}_R")
        col_order.append(f"{c}_L")
    chain_meta = chain_meta[col_order].copy()
    chain_meta.to_parquet(pinder_dir / "chain_metadata.parquet", index=False)


def collate_entity_pqts(
    entry_dirs: list[Path],
    pinder_dir: Path,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    output_pqt = pinder_dir / "entity_metadata.parquet"
    if output_pqt.is_file() and use_cache:
        return

    entity_files = get_matching_entry_files(
        entry_dirs, "*-entities.parquet", max_workers=max_workers, parallel=parallel
    )
    entity_dfs = parallel_read_csvs(
        entity_files, parallel=parallel, max_workers=max_workers
    )
    if entity_dfs:
        entity_df = pd.concat(entity_dfs, ignore_index=True)
        entity_df.to_parquet(output_pqt, index=False)


def populate_predicted(
    monomer_ids: list[str],
    pinder_path: Path,
    alphafold_path: str = "gs://public-datasets-deepmind-alphafold-v4",
    google_cloud_project: str = "",
    use_cache: bool = True,
) -> list[Monomer]:
    """Populate AlphaFold2 (aka "predicted") monomer structures

    Parameters
    ----------
    monomer_ids : list[str]
        The list of monomer IDs for which to populate the predicted structure.
    pinder_path : Path
        The path to the Pinder dataset.
    alphafold_path : str, optional
        The path to the AlphaFold dataset, by default "gs://public-datasets-deepmind-alphafold-v4".
    google_cloud_project : str, optional
        The Google Cloud project to use, by default "".
    use_cache : bool
        Whether to skip populating predicted PDBs if they already exist at
        the destination paths.

    Returns
    -------
    list[Monomer]
        The list of predicted Monomer instances which were successfully downloaded.
    """

    uniprot_ids = set()
    for monomer_id in monomer_ids:
        monomer = Monomer.from_string(monomer_id)
        uniprot = monomer.proteins[0].uniprot
        if uniprot == UNIPROT_UNDEFINED:
            continue
        uniprot_ids.add(uniprot)

    from google.cloud.storage.client import Client

    # client = Client.create_anonymous_client()
    # anonymous client does not work because we have to pay to access the bucket
    client = Client(project=google_cloud_project)
    bucket_name = alphafold_path.split("gs://")[1].split("/")[0]
    bucket = client.bucket(bucket_name)
    log.info(f"Searching alphafold bucket for {len(uniprot_ids)} UniProt IDs")
    valid_uniprots = []
    local_paths = []
    remote_paths = []
    af_dest_pdbs = []
    for uniprot in tqdm(uniprot_ids):
        af_monomer = Monomer([Protein(source="af", uniprot=uniprot)])
        af_dest_pdb = pinder_path / "pdbs" / f"{af_monomer}.pdb"
        af_dest_pdbs.append(af_dest_pdb)
        if use_cache and not empty_file(af_dest_pdb):
            continue

        # Check for uniprot isoforms
        if "-" in uniprot:
            canonical_uniprot = uniprot.split("-")[0]
        else:
            canonical_uniprot = uniprot

        blob_name = f"AF-{canonical_uniprot}-F1-model_v4.cif"
        blob = bucket.blob(blob_name)
        if blob.exists():
            valid_uniprots.append(uniprot)
            af_source_cif = f"{alphafold_path}/{blob_name}"
            af_dest_cif = pinder_path / "pdbs" / f"{af_monomer}.cif"
            af_dest_pdb = pinder_path / "pdbs" / f"{af_monomer}.pdb"
            remote_paths.append(af_source_cif)
            local_paths.append(af_dest_cif)

    gs = Gsutil(client=client)
    if len(local_paths):
        try:
            gs.cp_paired_paths(remote_paths, local_paths)
        except Exception as e:
            log.error(
                f"Encountered unexpected exception when downloading AF cif files: {e}"
            )

        af_monomers = []
        for af_dest_cif in local_paths:
            if af_dest_cif.is_file():
                try:
                    # alphafold uses chain A for all monomers
                    af_dest_pdb = af_dest_cif.parent / f"{af_dest_cif.stem}.pdb"
                    cif_to_pdb(af_dest_cif, af_dest_pdb)
                    af_dest_cif.unlink()
                    af_monomer = Monomer.from_string(Path(af_dest_pdb).stem)
                    af_monomers.append(af_monomer)
                except Exception as e:
                    log.error(f"Failed to convert {af_dest_cif} to PDB! {e}")

    af_monomers = []
    for af_dest_pdb in af_dest_pdbs:
        if not af_dest_pdb.is_file():
            continue
        af_monomer = Monomer.from_string(Path(af_dest_pdb).stem)
        af_monomers.append(af_monomer)

    log.info(f"Successfullly populated {len(af_monomers)} predicted monomer PDBs")
    return af_monomers


def get_pdb_entry_dirs(data_dir: Path) -> list[Path]:
    entry_dirs = []
    for two_char_dir in data_dir.iterdir():
        if two_char_dir.name.startswith("."):
            continue
        for entry in tqdm(two_char_dir.iterdir()):
            if not entry.name.startswith("pdb_"):
                continue
            entry_id = entry.name.split("_")[1][-4:].lower()
            entry_dirs.append(entry)
    return entry_dirs


def get_monomers_from_mapping_pqts(
    mapping_pqts: list[Path], pinder_dir: Path
) -> list[Monomer]:
    monomers = []
    for fname in mapping_pqts:
        try:
            mono = Monomer.from_string(fname.stem)
        except Exception as e:
            log.debug(f"Failed to parse {fname.name} into Monomer")
            continue

        monomer_dest_pdb = pinder_dir / "pdbs" / f"{mono}.pdb"
        if monomer_dest_pdb.is_file():
            monomers.append(mono)
    return monomers


def get_dimers_from_dimer_pdbs(
    dimer_pdbs: list[Path],
    pinder_dir: Path,
    validate_files: bool = True,
) -> tuple[list[Dimer], list[Monomer]]:
    pdb_sink = pinder_dir / "pdbs"
    pqt_sink = pinder_dir / "mappings"
    dimers = []
    monomers = []
    for fname in dimer_pdbs:
        pdb_id = fname.stem.split("__")[0]
        dimer = Dimer.from_string(fname.stem)
        prot_R = dimer.monomer1.proteins[0]
        prot_L = dimer.monomer2.proteins[0]
        mono_R = Monomer(
            [Protein(source=pdb_id, chain=prot_R.chain, uniprot=prot_R.uniprot)],
            side="R",
        )
        mono_L = Monomer(
            [Protein(source=pdb_id, chain=prot_L.chain, uniprot=prot_L.uniprot)],
            side="L",
        )
        mono_R_pdb = fname.parent / f"{mono_R}.pdb"
        mono_L_pdb = fname.parent / f"{mono_L}.pdb"
        mono_R_pqt = fname.parent / f"{mono_R}.parquet"
        mono_L_pqt = fname.parent / f"{mono_L}.parquet"
        required_files = [
            pdb_sink / fname.name,
            pdb_sink / mono_R_pdb.name,
            pdb_sink / mono_L_pdb.name,
            pqt_sink / mono_R_pqt.name,
            pqt_sink / mono_L_pqt.name,
        ]
        if validate_files:
            valid_dimer = all(f.is_file() for f in required_files)
            if not valid_dimer:
                continue
        dimers.append(dimer)
        monomers.append(mono_R)
        monomers.append(mono_L)
    return dimers, monomers


def get_af_monomers_from_monomer_ids(
    monomer_ids: list[str],
    pinder_dir: Path,
) -> list[Monomer]:
    uniprot_ids = set()
    for monomer_id in monomer_ids:
        monomer = Monomer.from_string(monomer_id)
        uniprot = monomer.proteins[0].uniprot
        if uniprot == UNIPROT_UNDEFINED:
            continue
        uniprot_ids.add(uniprot)

    af_monomers = []
    for uniprot in tqdm(uniprot_ids):
        af_monomer = Monomer([Protein(source="af", uniprot=uniprot)])
        af_dest_pdb = pinder_dir / "pdbs" / f"{af_monomer}.pdb"
        if af_dest_pdb.is_file():
            af_monomers.append(af_monomer)
    return af_monomers


def populate_entries(
    data_dir: Path,
    pinder_dir: Path,
    alphafold_path: str = "gs://public-datasets-deepmind-alphafold-v4",
    google_cloud_project: str = "",
    entry_dirs: list[Path] | None = None,
    use_cache: bool = True,
    use_af_cache: bool = True,
    populate_alphafold: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    """Index PINDER dimers

    Parameters:
    data_dir (Path): The directory where the data is stored.
    pinder_dir (Path): The directory where the PINDER data will be stored.
    alphafold_path (str, optional): The path to the AlphaFold models. Defaults to "gs://public-datasets-deepmind-alphafold-v4".
    google_cloud_project (str, optional): The name of the Google Cloud project that you have access to. Defaults to "".
    entry_dirs (list[Path], optional): Optional subset of PDB entry directories to populate. Will populate all if not provided.
    use_cache (bool): Whether to skip populating entries if they are already populated.
    use_af_cache (bool): Whether to skip populating AF2 entries if they are already populated.
    populate_alphafold (bool): Whether to populate AF2 entries after RCSB-derived PDBs.
    parallel (bool): Whether to populate entries in parallel. Note: this part requires more memory than other steps.
    max_workers (int, optional): Limit number of parallel processes spawned to `max_workers`.
    """

    if not entry_dirs:
        entry_dirs = get_pdb_entry_dirs(data_dir)

    mapping_files = get_matching_entry_files(
        entry_dirs,
        "*__*.parquet",
        max_workers=max_workers,
        parallel=parallel,
    )
    pdb_files = get_matching_entry_files(
        entry_dirs,
        "*.pdb",
        max_workers=max_workers,
        parallel=parallel,
    )

    monomer_mappings, dimer_mappings = split_monomer_dimer_mapping_pqts(mapping_files)
    monomer_pdbs, dimer_pdbs = split_monomer_dimer_pdbs(pdb_files)

    monomer_index = get_monomer_index_from_files(monomer_mappings, monomer_pdbs)
    monomer_pqt_files = list(set(monomer_index.monomer_pqt))
    monomer_pdb_files = list(set(monomer_index.monomer_pdb))

    pqt_dir = pinder_dir / "mappings"
    pdb_dir = pinder_dir / "pdbs"
    for d in [pqt_dir, pdb_dir]:
        if not d.is_dir():
            d.mkdir(parents=True)

    src_files = monomer_pqt_files + monomer_pdb_files
    dest_pqts = [pqt_dir / f.name for f in monomer_pqt_files]
    dest_pdbs = [pdb_dir / f.name for f in monomer_pdb_files]
    dest_files = dest_pqts + dest_pdbs
    log.info(f"Populating {len(monomer_pdb_files)} monomer entries!")
    parallel_copy_files(
        src_files=src_files,
        dest_files=dest_files,
        use_cache=use_cache,
        max_workers=max_workers,
        parallel=parallel,
    )

    dimer_index = get_dimer_index_from_files(
        data_dir, pinder_dir, dimer_mappings, dimer_pdbs
    )
    dimer_pdb_files = list(dimer_index.dimer_pdb)
    pdbs = list(dimer_index.dimer_pdb)
    pdbs.extend(list(dimer_index.R_pdb))
    pdbs.extend(list(dimer_index.L_pdb))

    pqts = list(dimer_index.R_map)
    pqts.extend(list(dimer_index.L_map))

    pqts = list(set(pqts))
    pdbs = list(set(pdbs))

    src_files = pqts + pdbs
    dest_pqts = [pqt_dir / f.name for f in pqts]
    dest_pdbs = [pdb_dir / f.name for f in pdbs]
    dest_files = dest_pqts + dest_pdbs
    log.info(f"Populating {len(pqts)} dimer mapping parquets + {len(pdbs)} PDBs")
    parallel_copy_files(
        src_files=src_files,
        dest_files=dest_files,
        use_cache=use_cache,
        max_workers=max_workers,
        parallel=parallel,
    )
    monomers = get_monomers_from_mapping_pqts(monomer_mappings, pinder_dir)
    log.info("Getting Dimers from dimer PDBs")
    dimers, dimer_monomers = get_dimers_from_dimer_pdbs(
        dimer_pdb_files, pinder_dir, validate_files=False
    )
    monomers.extend(dimer_monomers)

    if populate_alphafold:
        log.info(f"Constructing list of {len(monomers)} Monomer IDs")
        monomer_ids = list(sorted(set([str(monomer) for monomer in monomers])))
        # populate predicted monomers
        log.info("Populating predicted monomers")
        try:
            af_monomers = populate_predicted(
                monomer_ids,
                pinder_dir,
                alphafold_path,
                google_cloud_project,
                use_cache=use_af_cache,
            )
        except ValueError as e:
            log.error(f"Unable to populate predicted: {e}")
        except DefaultCredentialsError as e:
            log.error(f"Not authenticated: {e}")


def populate_predicted_from_monomers(
    data_dir: Path,
    pinder_dir: Path,
    alphafold_path: str = "gs://public-datasets-deepmind-alphafold-v4",
    google_cloud_project: str = "",
    entry_dirs: list[Path] | None = None,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    """Populate predicted monomers after monomers and dimers have been populated.

    Parameters:
    data_dir (Path): The directory where the data is stored.
    pinder_dir (Path): The directory where the PINDER data will be stored.
    alphafold_path (str, optional): The path to the AlphaFold models. Defaults to "gs://public-datasets-deepmind-alphafold-v4".
    google_cloud_project (str, optional): The name of the Google Cloud project that you have access to. Defaults to "".
    entry_dirs (list[Path], optional): Optional subset of PDB entry directories to populate. Will populate all if not provided.
    use_cache (bool): Whether to skip populating entries if they are already populated.
    """

    if not entry_dirs:
        entry_dirs = get_pdb_entry_dirs(data_dir)

    mapping_files = get_matching_entry_files(
        entry_dirs,
        "*__*.parquet",
        max_workers=max_workers,
        parallel=parallel,
    )
    pdb_files = get_matching_entry_files(
        entry_dirs,
        "*.pdb",
        max_workers=max_workers,
        parallel=parallel,
    )

    monomer_mappings, dimer_mappings = split_monomer_dimer_mapping_pqts(mapping_files)
    monomer_pdbs, dimer_pdbs = split_monomer_dimer_pdbs(pdb_files)

    dimer_index = get_dimer_index_from_files(
        data_dir, pinder_dir, dimer_mappings, dimer_pdbs
    )
    dimer_pdb_files = list(dimer_index.dimer_pdb)
    monomers = get_monomers_from_mapping_pqts(monomer_mappings, pinder_dir)
    log.info("Getting Dimers from dimer PDBs")
    dimers, dimer_monomers = get_dimers_from_dimer_pdbs(
        dimer_pdb_files, pinder_dir, validate_files=False
    )
    monomers.extend(dimer_monomers)

    log.info("Extracting monomer IDs...")
    monomer_ids = list(sorted(set([str(monomer) for monomer in monomers])))
    # populate predicted monomers
    log.info("Populating predicted monomers")
    try:
        af_monomers = populate_predicted(
            monomer_ids,
            pinder_dir,
            alphafold_path,
            google_cloud_project,
            use_cache=use_cache,
        )
    except ValueError as e:
        log.error(f"Unable to populate predicted: {e}")
    except DefaultCredentialsError as e:
        log.error(f"Not authenticated: {e}")


def get_dimers_from_interface_annotations(
    data_dir: Path,
    pinder_dir: Path,
) -> pd.DataFrame:
    annot = pd.read_parquet(pinder_dir / "interface_annotations.parquet")
    annot["dimer"] = annot.path.apply(lambda x: Dimer.from_string(Path(x).stem))
    annot.loc[:, "two_char"] = [entry[1:3] for entry in annot.entry_id]
    annot.loc[:, "entry_dir"] = annot.two_char + "/pdb_0000" + annot.entry_id
    R_map = []
    L_map = []
    R_pdb = []
    L_pdb = []
    dimer_pdb = []
    for entry_dir, dimer, path in zip(annot.entry_dir, annot.dimer, annot.path):
        R_map.append(data_dir / entry_dir / f"{dimer.monomer1}-R.parquet")
        L_map.append(data_dir / entry_dir / f"{dimer.monomer2}-L.parquet")
        R_pdb.append(data_dir / entry_dir / f"{dimer.monomer1}-R.pdb")
        L_pdb.append(data_dir / entry_dir / f"{dimer.monomer2}-L.pdb")
        dimer_pdb.append(data_dir / entry_dir / path)
    annot.loc[:, "R_map"] = R_map
    annot.loc[:, "L_map"] = L_map
    annot.loc[:, "R_pdb"] = R_pdb
    annot.loc[:, "L_pdb"] = L_pdb
    annot.loc[:, "dimer_pdb"] = dimer_pdb
    return annot


def _get_matching_entry_files(
    entry_dir: Path,
    glob_pattern: str,
) -> list[Path]:
    entry_files = list(entry_dir.glob(glob_pattern))
    return entry_files


def get_matching_entry_files(
    entry_dirs: list[Path],
    glob_pattern: str,
    max_workers: int | None = None,
    parallel: bool = True,
) -> list[Path]:
    """Find all files matching a glob pattern in parallel across a list of ingested
    PDB entry directories.

    Parameters:
    entry_dirs (list[Path]): PDB entry directories to search.
    glob_pattern (str): The glob expression to use for matching files.
    max_workers (int, optional): Limit number of parallel processes spawned to `max_workers`.

    """
    entry_file_lists: list[list[Path]] = process_starmap(
        _get_matching_entry_files,
        zip(entry_dirs, repeat(glob_pattern)),
        parallel=parallel,
        max_workers=max_workers,
    )
    entry_files = []
    for file_list in entry_file_lists:
        entry_files.extend(file_list)
    return entry_files


def split_monomer_dimer_mapping_pqts(
    mapping_files: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Split list of mapping parquet files into true monomers and split dimer monomers.

    Parameters:
    mapping_files (list[Path]): List of mapping files with parquet extension.

    """
    monomer_mappings = []
    dimer_mappings = []
    for fname in mapping_files:
        mapping_name = fname.stem
        # Its a split dimer monomer
        if mapping_name.endswith("-R") or mapping_name.endswith("-L"):
            dimer_mappings.append(fname)
        else:
            monomer_mappings.append(fname)
    return monomer_mappings, dimer_mappings


def split_monomer_dimer_pdbs(
    pdb_files: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Split list of PDB files into true monomer PDBs and dimer + split-dimer PDBs.

    Parameters:
    pdb_files (list[Path]): List of pdb files to split into monomers and dimers.

    """
    monomer_pdbs = []
    dimer_pdbs = []
    for fname in pdb_files:
        if fname.stem.endswith("-R") or fname.stem.endswith("-L"):
            dimer_pdbs.append(fname)
        elif "--" in fname.stem:
            dimer_pdbs.append(fname)
        else:
            monomer_pdbs.append(fname)
    return monomer_pdbs, dimer_pdbs


def get_monomer_index_from_files(
    monomer_mappings: list[Path],
    monomer_pdbs: list[Path],
) -> pd.DataFrame:
    """Get index of monomers with valid parquet mapping and PDB file pair on disk.
    The monomer mapping and PDB files do not need to be in a paired order.

    Parameters:
    monomer_mappings (list[Path]): List of mapping files corresponding to true monomers.
    monomer_pdbs (list[Path]): List of PDB files corresponding to true monomers.

    """
    monomer_pdb_df = []
    monomer_pqt_df = []
    for fname in monomer_pdbs:
        monomer_pdb_df.append({"monomer_pdb": fname, "monomer_id": fname.stem})
    for fname in monomer_mappings:
        monomer_pqt_df.append({"monomer_pqt": fname, "monomer_id": fname.stem})
    monomer_pdb_df = pd.DataFrame(monomer_pdb_df)
    monomer_pqt_df = pd.DataFrame(monomer_pqt_df)
    monomer_index = pd.merge(monomer_pdb_df, monomer_pqt_df)
    monomer_index = monomer_index[~monomer_index.monomer_pqt.isna()]
    monomer_index = monomer_index[~monomer_index.monomer_pdb.isna()].reset_index(
        drop=True
    )
    return monomer_index


def get_dimer_index_from_files(
    data_dir: Path,
    pinder_dir: Path,
    dimer_mappings: list[Path],
    dimer_pdbs: list[Path],
) -> pd.DataFrame:
    """Get index of dimer files with valid parquet mappings and PDB file pairs on disk.

    Parameters:
    data_dir (Path): The directory where the data is stored.
    pinder_dir (Path): The directory where the PINDER data will be stored.
    dimer_mappings (list[Path]): List of mapping files corresponding to split-dimer monomers.
    dimer_pdbs (list[Path]): List of PDB files corresponding to dimers and split-dimer monomers.

    """

    annot = get_dimers_from_interface_annotations(data_dir, pinder_dir)

    dimer_R_pdb = []
    dimer_L_pdb = []
    dimer_pdb = []
    for fname in dimer_pdbs:
        if fname.stem.endswith("-R"):
            dimer_R_pdb.append({"R_pdb": fname, "R_pdb_valid": True})
        elif fname.stem.endswith("-L"):
            dimer_L_pdb.append({"L_pdb": fname, "L_pdb_valid": True})
        else:
            dimer_pdb.append({"dimer_pdb": fname, "dimer_pdb_valid": True})
    dimer_R_pdb = pd.DataFrame(dimer_R_pdb)
    dimer_L_pdb = pd.DataFrame(dimer_L_pdb)
    dimer_pdb = pd.DataFrame(dimer_pdb)

    dimer_R_map = []
    dimer_L_map = []
    for fname in dimer_mappings:
        mapping_name = fname.stem
        if mapping_name.endswith("-R"):
            dimer_R_map.append({"R_map": fname, "R_map_valid": True})
        elif mapping_name.endswith("-L"):
            dimer_L_map.append({"L_map": fname, "L_map_valid": True})

    dimer_R_map = pd.DataFrame(dimer_R_map)
    dimer_L_map = pd.DataFrame(dimer_L_map)

    # We only want the collection of dimers with interface annotations
    dimer_index = pd.merge(annot, dimer_pdb, how="left")
    dimer_index = pd.merge(dimer_index, dimer_R_pdb, how="left")
    dimer_index = pd.merge(dimer_index, dimer_L_pdb, how="left")
    dimer_index = pd.merge(dimer_index, dimer_R_map, how="left")
    dimer_index = pd.merge(dimer_index, dimer_L_map, how="left")

    dimer_index.loc[:, "dimer_valid"] = (
        ~dimer_index.R_pdb_valid.isna()
        & ~dimer_index.L_pdb_valid.isna()
        & ~dimer_index.R_map_valid.isna()
        & ~dimer_index.L_map_valid.isna()
    )
    dimer_index = dimer_index.query("dimer_valid").reset_index(drop=True)
    return dimer_index


def get_populated_entries(
    data_dir: Path,
    pinder_dir: Path,
    alphafold_path: str = "gs://public-datasets-deepmind-alphafold-v4",
    google_cloud_project: str = "",
    entry_dirs: list[Path] | None = None,
    transient_interface_config: TransientInterfaceConfig = TransientInterfaceConfig(),
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    """Index PINDER dimers

    Parameters:
    data_dir (Path): The directory where the data is stored.
    pinder_dir (Path): The directory where the PINDER data will be stored.
    alphafold_path (str, optional): The path to the AlphaFold models. Defaults to "gs://public-datasets-deepmind-alphafold-v4".
    google_cloud_project (str, optional): The name of the Google Cloud project that you have access to. Defaults to "".
    entry_dirs (list[Path], optional): Optional subset of PDB entry directories to populate. Will populate all if not provided.
    transient_interface_config (TransientInterfaceConfig): Config object containing parameters used to label potentially transient interfaces.
    use_cache (bool): Whether to skip populating entries if they are already populated.
    """

    pdb_dir = Path(data_dir)
    pinder_dir = Path(pinder_dir)
    log.info(f"Indexing PINDER dimers: from {pdb_dir} to {pinder_dir}")

    if not (pinder_dir / "pdbs").is_dir():
        (pinder_dir / "pdbs").mkdir(parents=True, exist_ok=True)

    if not (pinder_dir / "mappings").is_dir():
        (pinder_dir / "mappings").mkdir(parents=True, exist_ok=True)

    if not entry_dirs:
        entry_dirs = get_pdb_entry_dirs(data_dir)

    log.info("Globbing mapping files...")
    mapping_files = get_matching_entry_files(
        entry_dirs,
        "*__*.parquet",
        max_workers=max_workers,
        parallel=parallel,
    )
    log.info("Globbing PDB files...")
    pdb_files = get_matching_entry_files(
        entry_dirs,
        "*.pdb",
        max_workers=max_workers,
        parallel=parallel,
    )

    monomer_mappings, dimer_mappings = split_monomer_dimer_mapping_pqts(mapping_files)
    monomer_pdbs, dimer_pdbs = split_monomer_dimer_pdbs(pdb_files)

    dimer_index = get_dimer_index_from_files(
        data_dir, pinder_dir, dimer_mappings, dimer_pdbs
    )
    dimer_pdb_files = list(dimer_index.dimer_pdb)
    monomers = get_monomers_from_mapping_pqts(monomer_mappings, pinder_dir)
    log.info("Getting Dimers from dimer PDBs")
    dimers, dimer_monomers = get_dimers_from_dimer_pdbs(
        dimer_pdb_files, pinder_dir, validate_files=False
    )
    monomers.extend(dimer_monomers)

    log.info("Constructing list of Monomer IDs")
    monomer_ids = list(sorted(set([str(monomer) for monomer in monomers])))
    monomer_df = pd.DataFrame({"id": monomer_ids})

    pdb_id = []
    chain = []
    uniprot = []
    for m_id in monomer_ids:
        mono = Monomer.from_string(m_id).proteins[0]
        pdb_id.append(mono.source)
        chain.append(mono.chain)
        uniprot.append(mono.uniprot)
    monomer_df["pdb_id"] = pdb_id
    monomer_df["chain"] = chain
    monomer_df["uniprot"] = uniprot
    log.info("Writing monomer_ids.parquet")
    monomer_df.to_parquet(pinder_dir / "monomer_ids.parquet", index=False)

    log.info("Constructing list of Dimer IDs")
    dimer_ids = list(sorted([str(dimer) for dimer in dimers]))
    dimers_df = pd.DataFrame({"id": dimer_ids})

    pdb_id = []
    chain_R = []
    chain_L = []
    uniprot_R = []
    uniprot_L = []
    for d_id in dimer_ids:
        dimer = Dimer.from_string(d_id)
        m1 = dimer.monomer1
        m2 = dimer.monomer2
        pdb_id.append(m1.proteins[0].source)
        uniprot_R.append(m1.proteins[0].uniprot)
        uniprot_L.append(m2.proteins[0].uniprot)
        chain_R.append(m1.proteins[0].chain)
        chain_L.append(m2.proteins[0].chain)

    dimers_df["pdb_id"] = pdb_id
    dimers_df["chain_R"] = chain_R
    dimers_df["uniprot_R"] = uniprot_R
    dimers_df["chain_L"] = chain_L
    dimers_df["uniprot_L"] = uniprot_L
    dimers_df["pdb_chain_chain"] = (
        dimers_df.pdb_id + "_" + dimers_df.chain_R + "_" + dimers_df.chain_L
    )
    log.info("Writing dimer_ids.parquet")
    dimers_df.to_parquet(pinder_dir / "dimer_ids.parquet", index=False)

    log.info("Getting predicted Monomers from monomer IDs")
    af_monomers = get_af_monomers_from_monomer_ids(monomer_ids, pinder_dir)
    valid_uniprots = {afm.proteins[0].uniprot for afm in af_monomers}
    predicted_list = []
    for monomer_id in monomer_ids:
        monomer = Monomer.from_string(monomer_id)
        uniprot = monomer.proteins[0].uniprot
        if uniprot in valid_uniprots:
            af_monomer = Monomer([Protein(source="af", uniprot=uniprot)])
            predicted_list.append({"id": monomer_id, "predicted_id": str(af_monomer)})
    predicted_df = pd.DataFrame(predicted_list)

    pdb_id = []
    chain = []
    uniprot = []
    for m_id in list(predicted_df.id):
        mono = Monomer.from_string(m_id).proteins[0]
        pdb_id.append(mono.source)
        chain.append(mono.chain)
        uniprot.append(mono.uniprot)
    predicted_df["pdb_id"] = pdb_id
    predicted_df["chain"] = chain
    predicted_df["uniprot"] = uniprot
    log.info("Writing predicted monomer IDs")
    predicted_df.to_parquet(pinder_dir / "monomer_predicted_ids.parquet", index=False)

    # Collate entity parquet files
    collate_entity_pqts(entry_dirs, pinder_dir, use_cache=use_cache, parallel=parallel)

    # merge metadata
    log.info("Merging metadata")
    merge_metadata(dimers, pinder_dir)

    # collect detailed chain metadata from mapping parquets (including pdb_strand_id)
    collate_chain_info(pinder_dir, parallel=parallel)

    # add enzyme classification
    log.info("Adding contains_enzyme boolean flag")
    add_enzyme_classification(pinder_dir, use_cache=use_cache)

    log.info("Mapping sabdab annotations to index")
    add_sabdab_annotations(pinder_dir, use_cache=use_cache)

    add_ecod_to_metadata(pinder_dir, use_cache=use_cache)

    label_potential_transient_interfaces(
        pinder_dir,
        config=transient_interface_config,
        use_cache=use_cache,
        parallel=parallel,
        max_workers=max_workers,
    )
    log.info("Indexing Done")


def index_dimers(
    data_dir: Path | str,
    pinder_dir: Path | str,
    alphafold_path: str = "gs://public-datasets-deepmind-alphafold-v4",
    google_cloud_project: str = "",
    entry_dirs: list[Path] | None = None,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
    transient_interface_config: TransientInterfaceConfig = TransientInterfaceConfig(),
) -> None:
    # Ensure data_dir and pinder_dir are Path objects.
    data_dir = Path(data_dir)
    pinder_dir = Path(pinder_dir)

    populate_entries(
        data_dir,
        pinder_dir,
        alphafold_path,
        google_cloud_project,
        entry_dirs,
        use_cache,
        parallel=parallel,
        max_workers=max_workers,
    )
    populate_predicted_from_monomers(
        data_dir=data_dir,
        pinder_dir=pinder_dir,
        google_cloud_project=google_cloud_project,
        entry_dirs=entry_dirs,
        use_cache=use_cache,
        parallel=parallel,
        max_workers=max_workers,
    )
    get_populated_entries(
        data_dir=data_dir,
        pinder_dir=pinder_dir,
        alphafold_path=alphafold_path,
        google_cloud_project=google_cloud_project,
        entry_dirs=entry_dirs,
        transient_interface_config=transient_interface_config,
        use_cache=use_cache,
        parallel=parallel,
        max_workers=max_workers,
    )


def find_intersection(row: pd.Series) -> int:
    """Find number of pinder dimer interface residues that intersect with an ECOD domain.

    Applied to each row of the per-chain mappings, where each row contains the residues in
    the interface, the residues in the structure in our numbering, PDB numbering, and the
    ECOD domain begin and end residue IDs in PDB numbering.

    Each row contains residues in condensed form, with comma separators. It is required to be
    in the same ordering for each column, such that a mapping can be constructed by splitting on
    commas. E.g., row.resi_pdb = '-1,0,1,2,3' and row.resi = '1,2,3,4,5'.
    """

    inter_res = row["interface"].split(",")
    our_res = row["resi"].split(",")
    pdb_res = row["resi_pdb"].split(",")
    try:
        start, end = int(row["beg_seq_id"]), int(row["end_seq_id"])
    except ValueError as e:
        log.error(
            f"Failed to convert ECOD domain ({row['beg_seq_id']}, {row['end_seq_id']}) to int"
        )
        return 0

    pdb_range = list(range(start, end + 1))
    pdb_map = {o: p for o, p in zip(our_res, pdb_res)}

    inter_pdb_res: set[int] = set()
    for i in inter_res:
        if not pdb_map.get(i):
            continue
        pdb_i = pdb_map[i]
        # First remove any letters that may exist
        pdb_i = re.sub("[A-Za-z]", "", pdb_i)
        # Remove potential float formatting
        pdb_i = cast_resi_to_valid_str(pdb_i)
        # Attempt cast to int, skip if unable
        try:
            int_i = int(pdb_i)
        except ValueError as e:
            log.error(f"Failed to convert {pdb_i} to integer!")
            continue
        inter_pdb_res.add(int_i)
    return len(inter_pdb_res.intersection(set(pdb_range)))


def get_per_chain_ecod_summary(
    ecod_RL: pd.DataFrame,
) -> pd.DataFrame:
    """Find ECOD annotations corresponding to pinder dimer chains.

    Adds comma-separated ECOD domain IDs, names and number of interface residues
    that intersect with the domain annotation for each matched pinder dimer chain.
    """
    ecod_RL.reset_index(drop=True, inplace=True)
    interface_intersect = []
    for i, r in tqdm(ecod_RL.iterrows()):
        interface_intersect.append(find_intersection(r))

    ecod_RL.loc[:, "interface_intersect"] = interface_intersect
    # Might want to save this in full long-form
    log.info("Collating per-chain ECOD domain overlap summary")
    grp_cols = ["id", "body", "chain", "asym_id", "pdb_strand_id", "entity_id"]
    summarized = []
    for (id, body, chain, asym_id, pdb_strand_id, entity_id), df in ecod_RL.groupby(
        grp_cols
    ):
        fids = ",".join(list(df["feature_id"]))
        fnames = ",".join(map(str, list(df["name"])))
        ecod_inter = ",".join(map(str, list(df["interface_intersect"])))
        summarized.append(
            {
                "id": id,
                "body": body,
                "chain": chain,
                "ECOD_ids": fids,
                "ECOD_names": fnames,
                "ECOD_intersection": ecod_inter,
            }
        )
    summary_df = pd.DataFrame(summarized)
    return summary_df


def add_ecod_to_metadata(
    pinder_dir: Path | str,
    use_cache: bool = True,
) -> None:
    """Add ECOD domain overlap with pinder dimers into metadata.

    Reads stage 1 metadata.1.csv.gz and writes a new metadata.2.csv.gz. If the output metadata
    file exists and `use_cache` is True, the step is skipped.
    """
    pinder_dir = Path(pinder_dir)
    if use_cache and (pinder_dir / "metadata.2.csv.gz").is_file():
        log.debug(f"Using cached ECOD annotations, already in metadata...")
        return None

    annotation_fp = pinder_dir / "rcsb_annotations"
    ecod_csv = annotation_fp / "features_ecod.csv.gz"
    if not ecod_csv.is_file():
        raise FileNotFoundError(
            "ECOD annotations not found! Run graphql queries first..."
        )
    ecod = read_csv_non_default_na(ecod_csv, dtype={"pdb_id": "str"})
    chain_meta = pd.read_parquet(pinder_dir / "chain_metadata.parquet")
    metadata = read_csv_non_default_na(
        pinder_dir / "metadata.1.csv.gz", dtype={"entry_id": "str"}
    )
    master_metadata = metadata.copy()
    # Interface metadata
    metadata = metadata[
        ["id", "entry_id", "chain_1_residues", "chain_2_residues"]
    ].copy()
    metadata.rename(
        {
            "chain_1_residues": "interface_R",
            "chain_2_residues": "interface_L",
            "entry_id": "pdb_id",
        },
        axis=1,
        inplace=True,
    )
    chain_meta = pd.merge(chain_meta, metadata, how="left")
    ecod = ecod[ecod.pdb_id.isin(set(chain_meta.pdb_id))].reset_index(drop=True)
    id_cols = ["id", "pdb_id"]
    chain_cols = [
        "asym_id",
        "entity_id",
        "pdb_strand_id",
        "chain",
        "resi",
        "resi_pdb",
        "resi_auth",
        "interface",
    ]
    pinder_cols = id_cols + chain_cols + ["body"]
    chain_R = chain_meta[id_cols + [f"{c}_R" for c in chain_cols]].copy()
    chain_L = chain_meta[id_cols + [f"{c}_L" for c in chain_cols]].copy()
    chain_R.rename({f"{c}_R": c for c in chain_cols}, axis=1, inplace=True)
    chain_L.rename({f"{c}_L": c for c in chain_cols}, axis=1, inplace=True)
    chain_R.loc[:, "body"] = "R"
    chain_L.loc[:, "body"] = "L"
    chain_RL = pd.concat([chain_R, chain_L]).reset_index(drop=True)
    ecod_cols = [
        "pdb_id",
        "feature_id",
        "name",
        "beg_seq_id",
        "end_seq_id",
        "assignment_version",
        "asym_id",
        "auth_asym_id",
        "entity_id",
        "additional_properties",
    ]
    ecod_RL = pd.merge(ecod[ecod_cols], chain_RL[pinder_cols])
    summary_df = get_per_chain_ecod_summary(ecod_RL)
    ecod_rename_cols = [c for c in summary_df.columns if c not in ["id", "body"]]
    summary_R = (
        summary_df.query('body == "R"').drop("body", axis=1).reset_index(drop=True)
    )
    summary_L = (
        summary_df.query('body == "L"').drop("body", axis=1).reset_index(drop=True)
    )
    summary_R.rename({c: f"{c}_R" for c in ecod_rename_cols}, axis=1, inplace=True)
    summary_L.rename({c: f"{c}_L" for c in ecod_rename_cols}, axis=1, inplace=True)
    # Add additional chain metadata columns to master metadata
    add_metadata_cols = [
        c for c in chain_cols if "resi" not in c and "interface" not in c
    ]
    chain_R_index = chain_meta[id_cols + [f"{c}_R" for c in add_metadata_cols]].copy()
    chain_L_index = chain_meta[id_cols + [f"{c}_L" for c in add_metadata_cols]].copy()
    chain_meta_index = pd.merge(metadata, chain_R_index, how="left")
    chain_meta_index = pd.merge(chain_meta_index, chain_L_index, how="left")
    metadata_ecod = pd.merge(chain_meta_index, summary_R, how="left")
    metadata_ecod = pd.merge(metadata_ecod, summary_L, how="left")
    metadata_ecod.to_parquet(pinder_dir / "ecod_metadata.parquet", index=False)

    meta_ecod_cols = [
        "entity_id_R",
        "entity_id_L",
        "pdb_strand_id_R",
        "pdb_strand_id_L",
        "ECOD_ids_R",
        "ECOD_ids_L",
        "ECOD_names_R",
        "ECOD_names_L",
        "ECOD_intersection_R",
        "ECOD_intersection_L",
    ]
    col_order = list(master_metadata.columns) + meta_ecod_cols
    metadata2 = pd.merge(
        # Drop any previously inserted default value columns when running fix_metadata
        master_metadata.drop(columns=meta_ecod_cols, errors="ignore"),
        metadata_ecod[["id"] + meta_ecod_cols],
        how="left",
        on="id",
    )
    metadata2 = metadata2[col_order].copy()
    metadata2.to_csv(pinder_dir / "metadata.2.csv.gz", index=False)
    log.info("Successfully wrote metadata.2.csv.gz with ECOD annotations")


def add_enzyme_classification(
    pinder_dir: Path | str,
    use_cache: bool = True,
) -> None:
    """Add enzyme classification numbers and set contains_enzyme based on RCSB EC annotations.

    Reads stage 1 index.1.csv.gz and writes a new `enzyme_classification_metadata.parquet` and
    sets the `contains_enzyme` column boolean column in the index based on whether either asym_id in
    the pinder dimer has an EC number. If the output metadata parquet file exists and `use_cache` is True,
    the step is skipped.
    """
    pinder_dir = Path(pinder_dir)
    if use_cache and (pinder_dir / "enzyme_classification_metadata.parquet").is_file():
        log.debug(f"Using cached EC annotations, already in metadata...")
        return None

    annotation_fp = pinder_dir / "rcsb_annotations"
    ec_csv = annotation_fp / "enzyme_classification.csv.gz"
    if not ec_csv.is_file():
        raise FileNotFoundError(
            "EC annotations not found! Run graphql queries first..."
        )
    ec = read_csv_non_default_na(ec_csv, dtype={"pdb_id": "str", "entry_id": "str"})
    ec_asym_long = []
    for pdb_id, df in ec.groupby("pdb_id"):
        for i, r in df.iterrows():
            ec_asym_long.extend(
                [
                    {"pdb_id": pdb_id, "ec": r.ec, "asym_id": asym}
                    for asym in r.asym_ids.split(",")
                ]
            )
    long_df = pd.DataFrame(ec_asym_long)
    ec_asym = []
    for (pdb_id, asym_id), df in long_df.groupby(["pdb_id", "asym_id"]):
        ec_ids = ",".join(set(df.ec))
        ec_asym.append({"pdb_id": pdb_id, "asym_id": asym_id, "ec": ec_ids})
    ec_df = pd.DataFrame(ec_asym)

    chain_meta = pd.read_parquet(pinder_dir / "chain_metadata.parquet")
    chain_meta.loc[:, "pdb_id"] = [id.split("__")[0] for id in list(chain_meta.id)]
    chain_R = chain_meta[
        ["id", "pdb_id", "asym_id_R", "pdb_strand_id_R", "chain_R"]
    ].copy()
    chain_L = chain_meta[
        ["id", "pdb_id", "asym_id_L", "pdb_strand_id_L", "chain_L"]
    ].copy()
    # Merge EC numbers with chain metadata
    EC_R = pd.merge(
        chain_R,
        ec_df.rename({"asym_id": "asym_id_R", "ec": "EC_R"}, axis=1),
        how="left",
    )
    EC_L = pd.merge(
        chain_L,
        ec_df.rename({"asym_id": "asym_id_L", "ec": "EC_L"}, axis=1),
        how="left",
    )
    RL = pd.merge(EC_R, EC_L, how="left")
    RL.loc[RL.EC_R.isna(), "EC_R"] = ""
    RL.loc[RL.EC_L.isna(), "EC_L"] = ""
    # Contains enzyme if either chain R or chain L have an EC number.
    RL.loc[:, "contains_enzyme"] = (RL.EC_R != "") | (RL.EC_L != "")
    RL.to_parquet(pinder_dir / "enzyme_classification_metadata.parquet", index=False)
    contains_enzyme = {
        r["id"]: r["contains_enzyme"]
        for r in RL[["id", "contains_enzyme"]].to_dict(orient="records")
    }
    index = read_csv_non_default_na(
        pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
    )
    index.loc[:, "contains_enzyme"] = index.id.apply(
        lambda x: contains_enzyme.get(x, False)
    )
    index.to_csv(pinder_dir / "index.1.csv.gz", index=False)
    log.info("Successfully wrote index.1.csv.gz with contains_enzyme boolean")


def add_predicted_monomers_to_index(
    pinder_dir: Path,
    use_cache: bool = True,
) -> None:
    pred_checkpoint = pinder_dir / "index_with_pred.parquet"
    if pred_checkpoint.is_file() and use_cache:
        log.info(f"{pred_checkpoint} exists, skipping...")

    pred_ids = pd.read_parquet(pinder_dir / "monomer_predicted_ids.parquet")
    index = read_csv_non_default_na(
        pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
    )
    col_order = list(index.columns)
    index.drop(["predicted_R_pdb", "predicted_L_pdb"], axis=1, inplace=True)

    # Update predicted_R/L and predicted_R/L_pdb columns in index
    # based on available pairings per R/L holo side
    pred_ids = pred_ids[["predicted_id", "uniprot"]].drop_duplicates(ignore_index=True)
    pred_ids["predicted_id"] = pred_ids["predicted_id"] + ".pdb"
    for side in ["R", "L"]:
        pred_pdb_col = f"predicted_{side}_pdb"
        index = pd.merge(
            index,
            pred_ids.rename(
                {"uniprot": f"uniprot_{side}", "predicted_id": pred_pdb_col}, axis=1
            ),
            how="left",
        )
        index.loc[index[pred_pdb_col].isna(), pred_pdb_col] = ""
        # Set the boolean indicating availability of predicted monomer for this side
        index.loc[:, f"predicted_{side}"] = index[pred_pdb_col] != ""
    index = index[col_order].copy()
    index.to_parquet(pred_checkpoint, index=False)


def get_dimer_interchain_bond_atom_info(
    pdb_file: Path,
    interface_res: dict[str, list[int]],
    config: TransientInterfaceConfig = TransientInterfaceConfig(),
) -> dict[str, str | int]:
    structure = Structure(pdb_file)
    interchain_at = find_potential_interchain_bonded_atoms(
        structure,
        interface_res=interface_res,
        radius=config.radius,
    )
    interchain_res_str = ";".join(
        [
            f"{at.chain_id}.{at.res_name}.{at.res_id}.{at.atom_name}"
            for at in interchain_at
        ]
    )
    disulfide_bond_indices = detect_disulfide_bonds(
        structure=structure.atom_array,
        distance=config.disulfide_bond_distance,
        distance_tol=config.disulfide_bond_distance_tol,
        dihedral=config.disulfide_bond_dihedral,
        dihedral_tol=config.disulfide_bond_dihedral_tol,
    )
    disulfide_bonds = disulfide_bond_indices.shape[0]
    interchain_info: dict[str, str | int] = {
        "id": structure.filepath.stem,
        "n_interchain_bond_atoms": len(interchain_at),
        "interchain_res": interchain_res_str,
        "disulfide_bonds": disulfide_bonds,
    }
    return interchain_info


def label_potential_transient_interfaces(
    pinder_dir: Path,
    config: TransientInterfaceConfig = TransientInterfaceConfig(),
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    output_pqt = pinder_dir / "transient_interface_metadata.parquet"
    if output_pqt.is_file() and use_cache:
        log.info(f"{output_pqt} exists, skipping...")
        return
    meta = read_csv_non_default_na(pinder_dir / "metadata.2.csv.gz")
    pdb_dir = pinder_dir / "pdbs"
    dimer_pdbs = []
    interface_res_list = []
    for pid, ch1_res, ch2_res in zip(
        meta["id"], meta["chain_1_residues"], meta["chain_2_residues"]
    ):
        interface_res = {
            "R": list(map(int, ch1_res.split(","))),
            "L": list(map(int, ch2_res.split(","))),
        }
        dimer_pdbs.append(pdb_dir / f"{pid}.pdb")
        interface_res_list.append(interface_res)

    interchain_bond_info = process_starmap(
        get_dimer_interchain_bond_atom_info,
        zip(dimer_pdbs, interface_res_list, repeat(config)),
        parallel=parallel,
        max_workers=max_workers,
    )
    interchain_df = pd.DataFrame(interchain_bond_info)
    transient_meta = pd.merge(meta[["id", "buried_sasa"]], interchain_df, how="left")
    transient_meta.loc[:, "interchain_bond_or_clash"] = (
        transient_meta.n_interchain_bond_atoms > 0
    )
    transient_meta.loc[:, "potential_transient"] = (
        transient_meta.buried_sasa < config.min_buried_sasa
    )
    transient_meta.loc[:, "potential_disulfide"] = transient_meta.disulfide_bonds > 0
    transient_meta.to_parquet(output_pqt, index=False, engine="pyarrow")
