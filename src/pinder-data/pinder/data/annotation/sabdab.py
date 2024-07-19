from __future__ import annotations
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from pinder.core.utils import setup_logger
from pinder.data.csv_utils import read_csv_non_default_na

log = setup_logger(__name__)


def download_sabdab(
    download_dir: Path | str,
    filename: str = "sabdab_summary_all.tsv",
    url: str = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/",
    overwrite: bool = True,
) -> Path:
    if not isinstance(download_dir, Path):
        download_dir = Path(download_dir)
    if not download_dir.is_dir():
        download_dir.mkdir(parents=True)
    filepath = download_dir / filename
    if not overwrite and filepath.is_file():
        return filepath

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return filepath


def explode_sabdab_per_chain(
    sabdab_df: pd.DataFrame,
) -> pd.DataFrame:
    # Explode sabdab table to be per chain for all columns that map to multiple values within single row
    explode_cols = [
        c
        for c in sabdab_df.columns
        if sabdab_df[c].astype(str).str.contains("|", regex=False).any()
    ]
    explode = []
    for i, r in sabdab_df.iterrows():
        long_vals: dict[str, list[str]] = {}
        for c in explode_cols:
            # Convert NA | NA ... to NaN | NaN
            if isinstance(r[c], str):
                vals = [
                    val.strip() if val.strip() != "NA" else np.nan
                    for val in r[c].split("|")
                ]
            else:
                vals = [r[c]]
            long_vals.setdefault(c, vals)
        # Ensure all exploded values are same length (e.g., antigen_species may not be same length)
        max_items = max(map(len, long_vals.values()))
        for k, v in long_vals.items():
            if len(v) < max_items:
                long_vals[k] = v + [v[0] for i in range(max_items - len(v))]
        keys = list(long_vals.keys())
        for row_vals in zip(*long_vals.values()):
            # Copy full row and replace condensed columns with exploded values
            row_dict = r.to_dict()
            update_vals = {k: v for k, v in zip(keys, row_vals)}
            row_dict.update(update_vals)
            explode.append(row_dict)
    long_sabdab = pd.DataFrame(explode)
    return long_sabdab


def map_to_pinder_chains(
    chains: pd.DataFrame,
    sabdab: pd.DataFrame,
) -> pd.DataFrame:
    # Create fake chain since sabdab sometimes uses one-letter author chain
    # despite multi-letter author chain in RCSB (e.g. H vs. HHH)
    chains.loc[:, "pdb_strand_id_R_alt"] = chains.pdb_strand_id_R.str[0]
    chains.loc[:, "pdb_strand_id_L_alt"] = chains.pdb_strand_id_L.str[0]
    chains.loc[:, "pdb_strand_id_R_lowercase"] = chains.pdb_strand_id_R_alt.str.lower()
    chains.loc[:, "pdb_strand_id_L_lowercase"] = chains.pdb_strand_id_L_alt.str.lower()
    for side in ["R", "L"]:
        # First prioritize merging on un-modified pdb_strand_id columns
        chains = pd.merge(
            chains,
            sabdab.rename(
                {
                    "sabdab_type": f"sabdab_{side}",
                    "pdb_strand_id": f"pdb_strand_id_{side}",
                },
                axis=1,
            ),
            how="left",
        )
        chains[f"sabdab_{side}"] = chains[f"sabdab_{side}"].fillna("")
        # Find systems without a match and attempt to match on first character of pdb_strand_id
        chains_defined = chains[chains[f"sabdab_{side}"] != ""].reset_index(drop=True)
        chains_undefined = chains[chains[f"sabdab_{side}"] == ""].reset_index(drop=True)
        chains_undefined = pd.merge(
            chains_undefined.drop(f"sabdab_{side}", axis=1),
            sabdab.rename(
                {
                    "sabdab_type": f"sabdab_{side}",
                    "pdb_strand_id": f"pdb_strand_id_{side}_alt",
                },
                axis=1,
            ),
            how="left",
        )
        chains_undefined[f"sabdab_{side}"] = chains_undefined[f"sabdab_{side}"].fillna(
            ""
        )
        # Combine the two attempts at matching
        chains = (
            pd.concat([chains_defined, chains_undefined], ignore_index=True)
            .sort_values("id")
            .reset_index(drop=True)
        )
    add_match_list = []
    for sabdab_type in ["Hchain", "Lchain", "antigen_chain"]:
        # Find systems without a match and attempt to match on lower-case version of pdb_strand_id
        missing_type = chains[
            ~chains.sabdab_R.str.contains(sabdab_type, regex=False)
            & ~chains.sabdab_L.str.contains(sabdab_type, regex=False)
        ].reset_index(drop=True)
        for side in ["R", "L"]:
            missing_side = missing_type[
                ~missing_type[f"sabdab_{side}"].str.contains(sabdab_type, regex=False)
            ].reset_index(drop=True)
            missing_side = pd.merge(
                missing_side.drop([f"sabdab_{side}"], axis=1, errors="ignore"),
                sabdab.rename(
                    {
                        "sabdab_type": f"sabdab_{side}",
                        "pdb_strand_id": f"pdb_strand_id_{side}_lowercase",
                    },
                    axis=1,
                ),
                how="left",
            )
            missing_side[f"sabdab_{side}"] = missing_side[f"sabdab_{side}"].fillna("")
            matched = missing_side[missing_side[f"sabdab_{side}"] != ""].reset_index(
                drop=True
            )
            add_match_list.append(matched)
    add_matches = pd.concat(add_match_list, ignore_index=True)
    # Combine all attempts at matching
    combined_matches = pd.concat([chains, add_matches], ignore_index=True)
    group_cols = [
        c for c in list(combined_matches.columns) if not c.startswith("sabdab_")
    ]
    # Condense multiple matches to a single pinder chain (e.g. chain R can be both H and L chain)
    chains = combined_matches.groupby(group_cols, as_index=False).agg(
        {"sabdab_R": lambda x: ";".join(set(x)), "sabdab_L": lambda x: ";".join(set(x))}
    )
    chains.loc[:, "contains_antigen"] = chains.sabdab_R.str.contains(
        "antigen", regex=False
    ) | chains.sabdab_L.str.contains("antigen", regex=False)
    chains.loc[:, "contains_antibody"] = (
        chains.sabdab_R.str.contains("Hchain", regex=False)
        | chains.sabdab_L.str.contains("Hchain", regex=False)
        | chains.sabdab_R.str.contains("Lchain", regex=False)
        | chains.sabdab_L.str.contains("Lchain", regex=False)
    )
    return chains


def map_to_sabdab_db(
    chains_long: pd.DataFrame,
    sabdab_long: pd.DataFrame,
) -> pd.DataFrame:
    # Set a `pinder_monomer` column corresponding to the R/L holo PDB.
    pinder_monomer = []
    for id, body in zip(chains_long["id"], chains_long["body"]):
        id_idx = int(body == "L")
        monomer = id.split("--")[id_idx] + f"-{body}"
        pinder_monomer.append(monomer)

    chains_long.loc[:, "pinder_monomer"] = pinder_monomer

    sabdab_with_pinder = sabdab_long.copy()
    for sabdab_type, chains_df in chains_long.groupby("sabdab_type"):
        chains_df.reset_index(drop=True, inplace=True)
        # Don't merge with NaNs here
        sabdab_type_df = sabdab_with_pinder[
            ~sabdab_with_pinder[sabdab_type].isna()
        ].reset_index(drop=True)
        sabdab_type_na = sabdab_with_pinder[
            sabdab_with_pinder[sabdab_type].isna()
        ].reset_index(drop=True)
        # Create single string of pinder IDs and pinder monomer IDs that match the row instance in sabdab
        chains_condensed = (
            chains_df.groupby(["pdb_id", "pdb_strand_id"], as_index=False)
            .agg(
                {
                    "id": lambda x: ";".join(set(x)),
                    "pinder_monomer": lambda x: ";".join(set(x)),
                }
            )
            .rename(
                {
                    "pdb_strand_id": sabdab_type,
                    "id": f"pinder_{sabdab_type}_ids",
                    "pinder_monomer": f"pinder_{sabdab_type}_monomers",
                },
                axis=1,
            )
        )
        sabdab_type_df = pd.merge(
            sabdab_type_df,
            chains_condensed[
                [
                    "pdb_id",
                    sabdab_type,
                    f"pinder_{sabdab_type}_ids",
                    f"pinder_{sabdab_type}_monomers",
                ]
            ],
            how="left",
        )
        sabdab_with_pinder = (
            pd.concat([sabdab_type_df, sabdab_type_na], ignore_index=True)
            .sort_values("pdb_id")
            .reset_index(drop=True)
        )
        sabdab_with_pinder[f"pinder_{sabdab_type}_ids"] = sabdab_with_pinder[
            f"pinder_{sabdab_type}_ids"
        ].fillna("")
        sabdab_with_pinder[f"pinder_{sabdab_type}_monomers"] = sabdab_with_pinder[
            f"pinder_{sabdab_type}_monomers"
        ].fillna("")
    return sabdab_with_pinder


def add_sabdab_annotations(
    pinder_dir: Path,
    use_cache: bool = True,
) -> None:
    sabdab_checkpoint = pinder_dir / "sabdab_metadata.parquet"
    if sabdab_checkpoint.is_file() and use_cache:
        log.info(f"{sabdab_checkpoint} exists, skipping...")
        return
    sabdab_tsv = download_sabdab(download_dir=pinder_dir / "external_annotations")
    # In this case, we DONT want to read the csv with non-default NA since sabdab uses `NA` to denote `NaN` / not defined.
    # Unfortunately, this means there is no distinction for cases where asym_id/pdb_strand_id = "NA" literal
    df = pd.read_csv(sabdab_tsv, dtype={"pdb": "str"}, sep="\t").rename(
        {"pdb": "pdb_id"}, axis=1
    )
    long_sabdab = explode_sabdab_per_chain(df)
    sabdab_chains = (
        long_sabdab.melt(
            id_vars=["pdb_id"], value_vars=["Lchain", "Hchain", "antigen_chain"]
        )
        .drop_duplicates()
        .rename({"variable": "sabdab_type", "value": "pdb_strand_id"}, axis=1)
    )
    sabdab_chains = (
        sabdab_chains[~sabdab_chains.pdb_strand_id.isna()]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    condensed = (
        sabdab_chains.groupby(["pdb_id", "pdb_strand_id"])["sabdab_type"]
        .apply(";".join)
        .reset_index()
    )
    chain_meta = pd.read_parquet(pinder_dir / "chain_metadata.parquet")
    chain_meta.loc[:, "pdb_id"] = [id.split("__")[0] for id in list(chain_meta.id)]
    meta_cols = ["id", "pdb_id"]
    chain_cols = ["asym_id", "pdb_strand_id"]
    for c in chain_cols:
        meta_cols.extend([f"{c}_R", f"{c}_L"])
    chains = chain_meta[meta_cols].copy()
    chains = map_to_pinder_chains(chains, condensed)
    pindex = read_csv_non_default_na(
        pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
    )
    # Drop the columns if they were previously added
    pindex.drop(
        ["contains_antibody", "contains_antigen"], axis=1, errors="ignore", inplace=True
    )
    pindex = pd.merge(
        pindex, chains[["id", "contains_antibody", "contains_antigen"]], how="left"
    )
    pindex.to_csv(pinder_dir / "index.1.csv.gz", index=False)

    # Now prepare an auxillary metadata table which contains pinder info mapped to the sabdab table schema.
    chains_long = (
        chains[["id", "pdb_id", "pdb_strand_id_R", "pdb_strand_id_L"]]
        .rename({"pdb_strand_id_R": "R", "pdb_strand_id_L": "L"}, axis=1)
        .melt(id_vars=["id", "pdb_id"], value_vars=["R", "L"])
        .rename({"variable": "body", "value": "pdb_strand_id"}, axis=1)
    )
    # Include fake chain names
    chains_alt = (
        chains[["id", "pdb_id", "pdb_strand_id_R_alt", "pdb_strand_id_L_alt"]]
        .rename({"pdb_strand_id_R_alt": "R", "pdb_strand_id_L_alt": "L"}, axis=1)
        .melt(id_vars=["id", "pdb_id"], value_vars=["R", "L"])
        .rename({"variable": "body", "value": "pdb_strand_id"}, axis=1)
    )

    chains_long = (
        pd.concat([chains_long, chains_alt], ignore_index=False)
        .sort_values(["id", "body", "pdb_strand_id"])
        .reset_index(drop=True)
    )

    # Merge long-form chain metadata with long-form pinder-mapped sabdab_type per chain
    chains_long = chains_long.merge(sabdab_chains, how="left")
    # Find chains without a match and attempt to match on lower-case variant
    unmatched = chains_long[chains_long.sabdab_type.isna()].reset_index(drop=True)
    lower_match = unmatched.merge(sabdab_chains, how="left")
    lower_match = lower_match[~lower_match.sabdab_type.isna()].reset_index(drop=True)
    chains_long = chains_long[~chains_long.sabdab_type.isna()].reset_index(drop=True)
    chains_long = pd.concat([chains_long, lower_match], ignore_index=True)

    sabdab_with_pinder = map_to_sabdab_db(chains_long, long_sabdab)
    sabdab_with_pinder.to_parquet(sabdab_checkpoint, index=False)
