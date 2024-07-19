from __future__ import annotations
import math
import os
import random
from pathlib import Path

import pandas as pd

from pinder.core.loader.structure import Structure
from pinder.core.structure.atoms import get_seq_identity, normalize_orientation
from pinder.core.utils import setup_logger
from pinder.core.utils.process import process_map


log = setup_logger(__name__)


def create_transformed_holo_monomer(pdb_paths: tuple[Path, Path]) -> None:
    assert len(pdb_paths) == 2
    input_pdb, output_pdb = pdb_paths
    holo_struct = Structure(input_pdb)
    arr = holo_struct.atom_array.copy()
    normalized = normalize_orientation(arr)
    holo_struct.atom_array = normalized.copy()
    holo_struct.to_pdb(output_pdb)


def create_normalized_test_monomers(
    pinder_dir: Path,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    # Store test set holo PDBs in separate directory for easier distribution and
    # to avoid modifying their filenames when creating normalized orientations
    src_pdb_dir = pinder_dir / "pdbs"
    test_pdb_dir = pinder_dir / "test_set_pdbs"
    test_pdb_dir.mkdir(exist_ok=True, parents=True)
    index = pd.read_parquet(pinder_dir / "index.parquet")
    test_index = index.query('split == "test"').reset_index(drop=True)
    holo_pdb_names = list(set(test_index.holo_R_pdb).union(set(test_index.holo_L_pdb)))
    pdb_path_pairs = [
        (src_pdb_dir / pdb_name, test_pdb_dir / pdb_name) for pdb_name in holo_pdb_names
    ]
    if use_cache:
        pdb_path_pairs = [pair for pair in pdb_path_pairs if not pair[1].is_file()]
    _ = process_map(
        create_transformed_holo_monomer,
        pdb_path_pairs,
        parallel=parallel,
        max_workers=max_workers,
    )


def get_stratified_sample(
    df: pd.DataFrame,
    feat: str | list[str],
    n_samples: int = 20,
    n_bins: int = 5,
    random_state: int = random.randint(0, 1000),
) -> pd.DataFrame:
    """Gets a stratified sample

    Outputs a stratified sample of the provided dataframe based on criteria set
    for binning distribution. Max binning dimension supported is 2D.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sample from.
    feat : Union[str, List]
        The feature to segment bins from (1-D or 2-D)
    n_samples : int
        The number of samples to sample from df with the same distribution probability
    n_bins : int
        Number of bin segments. Note: Exponential for 2D. eg 25 for 5 bins in 2D.
    random_state : int, optional
        random seed number

    Returns
    -------
    pd.DataFrame
        DataFrame containing sampled data from the original DataFrame

    """
    # Argument exceptions
    if isinstance(feat, list) & len(feat) > 2:
        raise ValueError(
            f"Too many features were provided! Max two allowed, got <{feat}>"
        )
    if n_bins > df.shape[0]:
        raise ValueError(
            "You must provide a bin size less than the number of rows! "
            f"n_bins = {n_bins}, df.shape[0] = {df.shape[0]}"
        )
    elif n_bins == 1:
        raise ValueError(f"You must provide a bin greater than 1, got <{n_bins}>")
    # If number of samples is greater than the original df size, gracefully return.
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].astype(str)
    if n_samples > df.shape[0]:
        return df
    dummy_df = df.copy()
    if isinstance(feat, list) & len(feat) > 1:
        col_a = f"{feat[0]}_bin"
        col_b = f"{feat[1]}_bin"
        dummy_df[col_a] = pd.cut(dummy_df[feat[0]], n_bins)
        dummy_df[col_b] = pd.cut(dummy_df[feat[1]], n_bins)
        group_df = dummy_df.groupby([col_a, col_b], observed=True)
        feats = [col_a, col_b]
    else:
        if isinstance(feat, list):
            feat = feat[0]
        col = f"{feat}_bin"
        dummy_df[col] = pd.cut(dummy_df[feat], n_bins)
        group_df = dummy_df.groupby(col, observed=True)
        feats = [col]
    frequency = group_df.ngroup().value_counts(normalize=True, sort=False)
    frequency.index = group_df.groups.keys()
    weights_df = (
        group_df.size()
        .to_frame("count")
        .assign(weights=frequency)
        .fillna(0)
        .reset_index()
    )
    weights_df = pd.merge(
        dummy_df, weights_df, how="left", left_on=feats, right_on=feats
    )
    sample = weights_df.sample(n_samples, weights="weights", random_state=random_state)
    return sample[df.columns].reset_index(drop=True)


def assign_pinder_s_subset(
    pinder_dir: Path,
    index: pd.DataFrame,
    max_size: int = 250,
    min_frac_heterodimers: float = 0.75,
    heterodimer_seq_identity_threshold: float = 0.8,
) -> pd.DataFrame:
    pdb_dir = pinder_dir / "pdbs"
    pinder_s = index.query('split == "test"').reset_index(drop=True)
    pinder_s.loc[:, "both_apo"] = pinder_s.apo_R & pinder_s.apo_L
    pinder_s.loc[:, "both_pred"] = pinder_s.predicted_R & pinder_s.predicted_L
    pinder_s.loc[:, "seq_R"] = [
        Structure(pdb_dir / f"{holo_R}").sequence
        for holo_R in list(pinder_s.holo_R_pdb)
    ]
    pinder_s.loc[:, "seq_L"] = [
        Structure(pdb_dir / f"{holo_L}").sequence
        for holo_L in list(pinder_s.holo_L_pdb)
    ]
    pinder_s.loc[:, "holo_seq_identity"] = [
        get_seq_identity(seq_R, seq_L)
        for seq_R, seq_L in zip(pinder_s["seq_R"], pinder_s["seq_L"])
    ]
    # use sequence identity < 0.8 as definition of heterodimer vs. uniprot (which may be undefined)
    pinder_s.loc[:, "heterodimer"] = (
        pinder_s.holo_seq_identity < heterodimer_seq_identity_threshold
    )

    metadata = pd.read_parquet(pinder_dir / "metadata.parquet")

    # keep all heterodimers with dual apo (and dual AF2)
    # apo_hetero = set(pinder_s.query('both_apo and heterodimer').id)
    apo_pred_hetero = pinder_s.query("both_apo and both_pred and heterodimer")
    target_ids = set(apo_pred_hetero.id)
    used_uniprots = set(apo_pred_hetero.uniprot_R).union(set(apo_pred_hetero.uniprot_L))
    remaining = pinder_s[~pinder_s.id.isin(target_ids)].reset_index(drop=True)
    remaining = pd.merge(
        remaining,
        metadata[["id", "probability", "buried_sasa", "ECOD_names_R", "ECOD_names_L"]],
    )
    remaining["ECOD_names_R"] = remaining["ECOD_names_R"].astype("str")
    remaining["ECOD_names_L"] = remaining["ECOD_names_L"].astype("str")
    remaining.loc[:, "ECOD_names_R"] = remaining.ECOD_names_R.apply(
        lambda x: ",".join(sorted(set(x.split(","))))
    )
    remaining.loc[:, "ECOD_names_L"] = remaining.ECOD_names_L.apply(
        lambda x: ",".join(sorted(set(x.split(","))))
    )
    # Create unique set of pfam IDs contained within a dimer for diversity sampling
    remaining.loc[:, "pfam_pairs"] = [
        ",".join(sorted(set(pfam_r.split(",")).union(set(pfam_l.split(",")))))
        for pfam_r, pfam_l in zip(remaining["ECOD_names_R"], remaining["ECOD_names_L"])
    ]
    # Float16 not supported by pd.cut
    remaining["buried_sasa"] = remaining["buried_sasa"].astype("float")
    # desired homodimer/heterodimer split
    n_hetero_samples = math.ceil(max_size * min_frac_heterodimers) - len(target_ids)
    n_homo_samples = max_size - n_hetero_samples - len(target_ids)
    # Split remaining into heterodimers with AF2 monomers available and homodimers with apo + AF2 monomers available.
    pred_hetero = remaining.query("both_pred and heterodimer").reset_index(drop=True)
    apo_homo = remaining.query("both_pred and both_apo and ~heterodimer").reset_index(
        drop=True
    )
    # Random sample one system from each set of unique pfam_pairs
    for sample_group, n_samples in [
        (pred_hetero, n_hetero_samples),
        (apo_homo, n_homo_samples),
    ]:
        pfam_samples = []
        for pf_p, df in sample_group.groupby("pfam_pairs", as_index=False):
            df = df[
                (~df.uniprot_R.isin(used_uniprots))
                & (~df.uniprot_L.isin(used_uniprots))
            ].reset_index(drop=True)
            if not df.shape[0]:
                log.warning(
                    f"Not able to select a sample from PFAM {pf_p}, all uniprots already sampled"
                )
                continue
            df = df.sample(1).reset_index(drop=True)
            used_uniprots.add(df.uniprot_R.values[0])
            used_uniprots.add(df.uniprot_L.values[0])
            pfam_samples.append(df)

        if not len(pfam_samples):
            log.warning(
                "Unable to sample any PFAM pairs from this group, all uniprots sampled"
            )
            continue

        sampled = pd.concat(pfam_samples, ignore_index=True)
        # Get desired number of samples from binned buried_sasa for representative sampling of interface size
        n_bins = min([5, sampled.shape[0]])
        if n_bins == 1:
            sasa_sampled = sampled.copy()
        else:
            sasa_sampled = get_stratified_sample(
                sampled,
                "buried_sasa",
                n_samples=n_samples,
                n_bins=n_bins,
            )
        target_ids = target_ids.union(set(sasa_sampled.id))

    index.loc[:, "pinder_s"] = index.id.isin(target_ids)
    return index


def assign_test_subsets(pinder_dir: Path, use_cache: bool = True) -> None:
    index = pd.read_parquet(pinder_dir / "index.parquet", engine="pyarrow")
    if use_cache and index.pinder_xl.sum() > 0:
        return
    index.loc[index.split == "test", "pinder_xl"] = True
    index = assign_pinder_s_subset(pinder_dir, index)
    index.to_parquet(pinder_dir / "index.parquet", index=False, engine="pyarrow")


def curate_test_split(
    pinder_dir: Path,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    log.info("Curating normalized holo monomers for test set...")
    create_normalized_test_monomers(pinder_dir, use_cache, parallel, max_workers)
    log.info("Assigning pinder_xl and pinder_s test subsets...")
    assign_test_subsets(pinder_dir, use_cache)


def extract_sequence(pdb_file: Path) -> dict[str, str]:
    try:
        seq = Structure(pdb_file).sequence
    except Exception as e:
        log.error(f"Failed to get sequence for {pdb_file.name}")
        seq = ""
    seq_info: dict[str, str] = {"pdb": pdb_file.name, "sequence": seq}
    return seq_info


def construct_sequence_database(
    pinder_dir: Path,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    seq_db_chkpt = pinder_dir / "sequence_database.parquet"
    if use_cache and seq_db_chkpt.is_file():
        log.info(f"{seq_db_chkpt} exists, skipping...")
        return
    pdb_dir = pinder_dir / "pdbs"
    pdb_files = [pdb_dir / p for p in os.listdir(pdb_dir)]
    seq_info = process_map(
        extract_sequence,
        pdb_files,
        parallel=parallel,
        max_workers=max_workers,
    )
    seq_db = pd.DataFrame(seq_info)
    seq_db.loc[seq_db.pdb.str.contains("--", regex=False), "pdb_kind"] = "dimer"
    seq_db.loc[seq_db.pdb.str.endswith("-R.pdb"), "pdb_kind"] = "receptor"
    seq_db.loc[seq_db.pdb.str.endswith("-L.pdb"), "pdb_kind"] = "ligand"
    seq_db.loc[seq_db.pdb.str.startswith("af__"), "pdb_kind"] = "predicted"
    seq_db.loc[seq_db.pdb_kind.isna(), "pdb_kind"] = "monomer"
    seq_db["pdb_kind"] = seq_db["pdb_kind"].astype("category")
    seq_db["sequence"] = seq_db["sequence"].astype("category")
    seq_db.to_parquet(seq_db_chkpt, index=False, engine="pyarrow")
