from __future__ import annotations
import logging
from itertools import repeat
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pinder.core.utils import setup_logger
from pinder.core.utils.process import process_starmap
from pinder.data.config import ApoPairingConfig, MMSeqsConfig
from pinder.data.foldseek_utils import (
    alignment_to_parquet,
    parallel_extract_fasta,
    run_mmseqs,
)
from pinder.data.apo_utils import (
    calculate_frac_monomer_dimer_overlap,
    get_apo_pairing_metrics_for_id,
    remove_apo_chain_copies,
    remove_dimer_chain_copies,
    validate_apo_monomer,
)

UNIPROT_UNDEFINED = "UNDEFINED"


log = setup_logger(__name__, log_level=logging.WARNING)


def get_valid_apo_monomer_ids(
    pinder_dir: Path,
    config: ApoPairingConfig = ApoPairingConfig(),
    max_workers: int | None = None,
    use_cache: bool = True,
    remove_chain_copies: bool = True,
    parallel: bool = True,
) -> None:
    """Validates and stores a list of valid apo monomer IDs based on specific criteria defined in the configuration.

    This function processes monomer IDs to determine which qualify as valid apo monomers based on atom types
    and residue counts. Results are saved to a Parquet file, and processing is skipped if the file already exists
    and caching is enabled.

    Args:
        pinder_dir (Path): The directory that contains monomer data and where the output will be stored.
        config (ApoPairingConfig, optional): Configuration containing the validation thresholds.
        max_workers (int | None, optional): The maximum number of worker processes for parallel computation.
        use_cache (bool, optional): If True, skips processing if the output file already exists.
        remove_chain_copies (bool, optional): If True, removes duplicate chain entries before processing.
    """
    output_pqt = pinder_dir / "putative_apo_monomer_ids.parquet"
    if output_pqt.is_file() and use_cache:
        log.info(f"Skipping apo monomer ID validation, {output_pqt} exists...")
        return
    monomer_ids = pd.read_parquet(pinder_dir / "monomer_ids.parquet")
    monomer_ids.loc[:, "is_dimer"] = monomer_ids.id.str.endswith(
        "-R"
    ) | monomer_ids.id.str.endswith("-L")
    monomer_ids_apo = monomer_ids.query("~is_dimer").reset_index(drop=True)
    if remove_chain_copies:
        monomer_ids_apo = remove_apo_chain_copies(monomer_ids_apo)
    apo_ids = list(monomer_ids_apo.id)
    pdb_dir = pinder_dir / "pdbs"

    apo_info = process_starmap(
        validate_apo_monomer,
        zip(apo_ids, repeat(pdb_dir), repeat(config)),
        parallel=parallel,
        max_workers=max_workers,
    )
    apo_info = pd.DataFrame(apo_info)
    monomer_ids_apo = pd.merge(monomer_ids_apo, apo_info)
    monomer_ids_apo.to_parquet(output_pqt, index=False)


def get_putative_pairings(
    pinder_dir: Path,
    use_cache: bool = True,
    remove_chain_copies: bool = False,
) -> pd.DataFrame:
    """Generates a DataFrame of putative apo-holo pairings from validated apo monomer IDs.

    This function loads validated apo monomer IDs and pairs them with corresponding holo structures.
    The pairing is done based solely on Uniprot ID of the holo and apo monomer, respectively.
    Results are stored in a Parquet file and cached if enabled.

    Args:
        pinder_dir (Path): Directory containing the validated apo monomer IDs and holo structures.
        use_cache (bool, optional): If True, returns cached pairings from a Parquet file if available.
        remove_chain_copies (bool, optional): If True, removes duplicate chain entries before pairing.

    Returns:
        pd.DataFrame: A DataFrame containing putative apo-holo pairings.
    """
    output_pqt = pinder_dir / "putative_two_sided_apo_pairings.parquet"
    if output_pqt.is_file() and use_cache:
        log.info(f"Skipping putative two-sided pairing, {output_pqt} exists...")
        putative_pairings = pd.read_parquet(output_pqt)
        return putative_pairings
    putative_apo = pd.read_parquet(pinder_dir / "putative_apo_monomer_ids.parquet")
    putative_apo = putative_apo.query("valid_as_apo").reset_index(drop=True)
    # Create copy of apo for R and L side
    R_apo = (
        putative_apo.copy()
        .query(f'uniprot != "{UNIPROT_UNDEFINED}"')
        .rename(
            {
                "uniprot": "uniprot_R",
                "id": "apo_monomer_id",
                "pdb_id": "apo_pdb_id",
                "chain": "apo_chain",
            },
            axis=1,
        )
        .drop("is_dimer", axis=1)
    )
    R_apo.loc[:, "body"] = "R"
    L_apo = (
        putative_apo.copy()
        .query(f'uniprot != "{UNIPROT_UNDEFINED}"')
        .rename(
            {
                "uniprot": "uniprot_L",
                "id": "apo_monomer_id",
                "pdb_id": "apo_pdb_id",
                "chain": "apo_chain",
            },
            axis=1,
        )
        .drop("is_dimer", axis=1)
    )
    L_apo.loc[:, "body"] = "L"

    # Only get first instance of a chain to avoid counting copies
    pindex = pd.read_parquet(pinder_dir / "index_with_pred.parquet")
    if remove_chain_copies:
        pindex = remove_dimer_chain_copies(pindex)

    # Create full set of holo-R and holo-L pdbs to evaluate with apo
    holo_R = pindex[
        ["id", "pdb_id", "holo_R_pdb", "holo_L_pdb", "uniprot_R", "uniprot_L"]
    ].copy()
    holo_L = pindex[
        ["id", "pdb_id", "holo_R_pdb", "holo_L_pdb", "uniprot_R", "uniprot_L"]
    ].copy()
    R = pd.merge(holo_R, R_apo)
    L = pd.merge(holo_L, L_apo)
    putative_pairings = (
        pd.concat([R, L], ignore_index=True).sort_values("id").reset_index(drop=True)
    )
    putative_pairings.to_parquet(output_pqt, index=False)
    return putative_pairings


def get_apo_pairing_metrics(
    pinder_dir: Path,
    putative_pairs: list[str] | pd.DataFrame,
    config: ApoPairingConfig = ApoPairingConfig(),
    max_workers: int | None = None,
    output_parquet: Path | None = None,
    use_cache: bool = True,
    parallel: bool = True,
) -> pd.DataFrame:
    """Retrieves or calculates apo-holo pairing metrics from specified pair identifiers or DataFrame.

    This function processes pairings to calculate various metrics that help assess the suitability of apo-holo
    pairings. If caching is enabled and a valid cache file exists, the function returns the data from the cache.

    Args:
        pinder_dir (Path): Base directory containing the data.
        putative_pairs (list[str] | pd.DataFrame): Either a list of pairing identifiers or a DataFrame with pairings.
        config (ApoPairingConfig): Configuration settings for the pairing analysis.
        max_workers (int | None): Maximum number of worker processes for parallel computation.
        output_parquet (Path | None): Path to a Parquet file where results are stored or retrieved.
        use_cache (bool): Whether to use cached results if available.

    Returns:
        pd.DataFrame: A DataFrame containing metrics for each pair.
    """

    if output_parquet:
        if use_cache and output_parquet.is_file():
            log.debug(
                f"Skipping two-sided apo metric calculation, {output_parquet} exists..."
            )
            metric_df = pd.read_parquet(output_parquet)
            return metric_df
        if not output_parquet.parent.is_dir():
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pdb_dir = pinder_dir / "pdbs"
    if isinstance(putative_pairs, pd.DataFrame):
        putative_pairings = putative_pairs.copy()
    else:
        pairings = get_putative_pairings(
            pinder_dir=pinder_dir, use_cache=True, remove_chain_copies=False
        )
        # Only operate on subset of pairing IDs provided in putative_pairing_ids
        pairings["pairing_id"] = (
            pairings["id"] + ":" + pairings["apo_monomer_id"] + ":" + pairings["body"]
        )
        putative_pairings = pairings[
            pairings["pairing_id"].isin(putative_pairs)
        ].reset_index(drop=True)
        putative_pairings.drop("pairing_id", axis=1, inplace=True)

    dfs = [df for id, df in putative_pairings.groupby("id")]
    metric_dfs = process_starmap(
        get_apo_pairing_metrics_for_id,
        zip(dfs, repeat(pdb_dir), repeat(config)),
        parallel=parallel,
        max_workers=max_workers,
    )
    metric_dfs = [df for df in metric_dfs if isinstance(df, pd.DataFrame)]
    if len(metric_dfs):
        metric_df = pd.concat(metric_dfs).reset_index(drop=True)
    else:
        log.warning("Two-sided apo metrics returned zero valid dataframes!")
        metric_df = pd.DataFrame()
    if output_parquet:
        metric_df.to_parquet(output_parquet, index=False)
    return metric_df


def select_potential_apo(
    pinder_dir: Path,
    config: ApoPairingConfig = ApoPairingConfig(),
    parallel: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Selects potential apo structures based on various metrics from a metrics DataFrame stored in a Parquet file.
    Additional metrics are calculated and hard filters are applied based on config.

    Args:
        pinder_dir (Path): The directory where the data is stored.
        config (ApoPairingConfig): Configuration used to determine selection criteria.

    Returns:
        pd.DataFrame: A DataFrame of selected potential apo structures based on defined criteria.
    """
    metric_dir = pinder_dir / "apo_metrics"
    metric_pqt = metric_dir / "two_sided_apo_monomer_metrics.parquet"
    assert metric_pqt.is_file()
    two_sided = pd.read_parquet(metric_pqt)
    two_sided.loc[:, "pdb_id"] = [pid.split("__")[0] for pid in list(two_sided.id)]
    # We may want to normalize these values somehow by clashes vs. contacts.
    # The general idea is to find structures where the apo monomer contains the whole holo dimer
    # or large superset of both dimer monomers.
    # However, we don't want to penalize / invalidate an apo monomer
    # if it simply has more residues resolved (e.g. akin to using AF2 structure).
    two_sided.loc[:, "apo_holo_ligand_interface_frac"] = (
        two_sided.apo_ligand_interface_res / two_sided.holo_ligand_interface_res
    )
    two_sided.loc[:, "apo_holo_receptor_interface_frac"] = (
        two_sided.apo_receptor_interface_res / two_sided.holo_receptor_interface_res
    )
    two_sided.loc[:, "pairing_id"] = (
        two_sided.id + "___" + two_sided.apo_monomer_id + "___" + two_sided.unbound_body
    )
    two_sided.loc[:, "max_miss"] = two_sided[["Fmiss_R", "Fmiss_L"]].max(axis=1)
    # We may want to revisit dimer entries which no longer have a potential apo after applying this filter
    # to see if any can be rescued with alternative metrics. For now, this favors higher-confidence pairings.
    two_sided.loc[:, "invalid_coverage"] = (
        (
            two_sided.apo_holo_receptor_interface_frac
            >= config.invalid_coverage_upper_bound
        )
        | (
            two_sided.apo_holo_ligand_interface_frac
            >= config.invalid_coverage_upper_bound
        )
        | (
            two_sided.apo_holo_receptor_interface_frac
            < config.invalid_coverage_lower_bound
        )
        | (
            two_sided.apo_holo_ligand_interface_frac
            < config.invalid_coverage_lower_bound
        )
    )
    two_sided.loc[:, "holo_res"] = [len(s) for s in two_sided.holo_sequence]
    two_sided.loc[:, "apo_res"] = [len(s) for s in two_sided.apo_sequence]
    two_sided.loc[:, "frac_aligned"] = [
        aln_res / min([holo_res, apo_res])
        for aln_res, holo_res, apo_res in zip(
            two_sided["aln_res"], two_sided["holo_res"], two_sided["apo_res"]
        )
    ]
    potential_apo = (
        two_sided.query(
            f"Fmiss_R <= {config.max_interface_miss_frac} and Fmiss_L <= {config.max_interface_miss_frac}"
        )
        .query(f"sequence_identity >= {config.min_seq_identity}")
        .query(f"refine_rmsd < {config.max_refine_rmsd}")
        .query(f"frac_aligned >= {config.min_aligned_apo_res_frac}")
        .query("~invalid_coverage")
        .sort_values(["id", "max_miss"], ascending=False)
        .reset_index(drop=True)
    )
    # Identify proteins which spontaneously hydrolyze/fragment into separate "homodimer" chains
    # We don't want our apo pair to contain the R+L chain in a single chain.
    potential_apo.loc[:, "uniprot_R"] = [
        id.split("--")[0].split("_")[-1] for id in list(potential_apo.id)
    ]
    potential_apo.loc[:, "uniprot_L"] = [
        id.split("--")[1].split("_")[-1] for id in list(potential_apo.id)
    ]
    potential_apo.loc[:, "homodimer"] = (
        potential_apo.uniprot_R == potential_apo.uniprot_L
    )
    heterodimer = potential_apo.query("~homodimer").reset_index(drop=True)
    homodimer = potential_apo.query("homodimer").reset_index(drop=True)
    homodimer_eval = homodimer[["id", "apo_monomer_id", "unbound_body"]].copy()
    pdb_dir = pinder_dir / "pdbs"
    dfs = [df for id, df in homodimer_eval.groupby("id")]
    metric_dfs = process_starmap(
        calculate_frac_monomer_dimer_overlap,
        zip(dfs, repeat(pdb_dir), repeat(config)),
        parallel=parallel,
        max_workers=max_workers,
    )
    metric_dfs = [df for df in metric_dfs if isinstance(df, pd.DataFrame)]
    if len(metric_dfs):
        metric_df = pd.concat(metric_dfs, ignore_index=True)
        homodimer = pd.merge(homodimer, metric_df, how="left")
    else:
        homodimer.loc[:, "frac_monomer_dimer"] = 0.0
    potential_apo = pd.concat(
        [
            homodimer,
            heterodimer,
        ],
        ignore_index=True,
    )
    potential_apo["frac_monomer_dimer"] = potential_apo["frac_monomer_dimer"].fillna(
        0.0
    )
    potential_apo.drop(["homodimer", "uniprot_R", "uniprot_L"], axis=1, inplace=True)
    # Cutoff not applied - has false negatives
    # potential_apo = (
    #     potential_apo
    #     .query(f'frac_monomer_dimer < {config.max_frac_monomer_dimer_sequence}')
    #     .reset_index(drop=True)
    # )
    return potential_apo


def get_apo_monomer_weighted_score(
    apo_data: pd.DataFrame,
    config: ApoPairingConfig = ApoPairingConfig(),
    scale_type: str = "standard",
) -> pd.DataFrame:
    """Uses suite of apo-holo difficulty assessment metrics to compute a weighted score
    that is used to rank and select a single receptor and single ligand monomer for a given pinder dimer
    entry when apo structures are available.

    Args:
        apo_data (pd.DataFrame): Data containing metrics for each apo monomer.
        config (ApoPairingConfig): Configuration containing the metrics and their weights.
        scale_type (str): Type of scaling to apply, 'standard' for Z-score or 'minmax' for Min-Max scaling.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'apo_score' containing the computed scores.
    """
    if scale_type == "standard":
        scaler = StandardScaler()
    elif scale_type == "minmax":
        scaler = MinMaxScaler()
    scaler = StandardScaler()
    weight = 1 / len(config.scaled_score_metrics)
    metric_weights: dict[str, float] = {
        metric: weight for metric in config.scaled_score_metrics
    }
    scaled = pd.DataFrame(
        scaler.fit_transform(apo_data[metric_weights.keys()]),
        columns=list(metric_weights.keys()),
    )
    # Invert scaling on i-RMSD. Higher = worse
    # The score will be higher = better
    inverse_metrics = ["raw_rmsd", "refine_rmsd", "I-RMSD", "Fnonnat"]
    for metric in inverse_metrics:
        if metric in config.scaled_score_metrics:
            scaled.loc[:, metric] = scaled[metric] * -1
    apo_data["apo_score"] = scaled.dot(
        pd.DataFrame(
            metric_weights.values(),
            index=metric_weights.keys(),
        )
    )
    return apo_data


def add_weighted_apo_score(
    potential_apo: pd.DataFrame,
    config: ApoPairingConfig = ApoPairingConfig(),
) -> pd.DataFrame:
    """Adds a weighted score to each potential apo structure to facilitate the selection of the most suitable structure.

    Args:
        potential_apo (pd.DataFrame): DataFrame containing the potential apo structures.
        config (ApoPairingConfig): Configuration settings for the scoring system.

    Returns:
        pd.DataFrame: The DataFrame with an additional column representing the weighted score of each entry.
    """
    scored = (
        potential_apo.groupby(["id", "unbound_body"], as_index=False)
        .apply(
            lambda x: get_apo_monomer_weighted_score(
                x.reset_index(drop=True), config=config
            )
        )
        .reset_index(drop=True)
        .sort_values("apo_score", ascending=False)
        .reset_index(drop=True)
    )
    return scored


def run_monomer_dimer_mmseqs(
    pinder_dir: Path,
    potential_apo: pd.DataFrame,
    use_cache: bool = True,
) -> None:
    """Executes MMseqs2 to compare sequence similarities between monomers and dimers and caches the results.
    This method acts as a second layer of validation on the original pairing algorithm. The alignment file
    can be used to calculate an alternative metric akin to calculate_frac_monomer_dimer_overlap.
    The usage of the mmseqs outputs are currently experimental and not used in the pairing or final selection.

    Args:
        pinder_dir (Path): Directory where data is stored and from which MMseqs2 is run.
        potential_apo (pd.DataFrame): DataFrame containing potential apo structures to compare.
        use_cache (bool): If True, uses cached results if available.

    Returns:
        None: Results are saved to files and not directly returned.
    """
    apo_dir = pinder_dir / "apo_metrics"
    mmseqs_dir = apo_dir / "mmseqs"
    mmseqs_dir.mkdir(exist_ok=True, parents=True)
    mmseqs_chkpt = mmseqs_dir / "potential_apo_mmseqs.parquet"
    if mmseqs_chkpt.is_file() and use_cache:
        log.info(f"{mmseqs_chkpt} exists, skipping...")
    dimer_pdbs = [pinder_dir / "pdbs" / f"{id}.pdb" for id in set(potential_apo.id)]
    apo_pdbs = [
        pinder_dir / "pdbs" / f"{id}.pdb" for id in set(potential_apo.apo_monomer_id)
    ]
    apo_seqs = parallel_extract_fasta(pdb_files=apo_pdbs)
    apo_fasta = mmseqs_dir / "apo_db.fasta"
    with open(apo_fasta, "w") as f:
        f.write("\n".join(apo_seqs))
    dimer_seqs = parallel_extract_fasta(pdb_files=dimer_pdbs)
    dimer_fasta = mmseqs_dir / "dimer_db.fasta"
    with open(dimer_fasta, "w") as f:
        f.write("\n".join(dimer_seqs))
    run_mmseqs(dimer_fasta, mmseqs_dir, apo_fasta, use_cache=use_cache)
    run_mmseqs(
        apo_fasta,
        mmseqs_dir,
        dimer_fasta,
        use_cache=use_cache,
        config=MMSeqsConfig(alignment_filename="reversed_alignment.txt"),
    )
    alignment_to_parquet(
        alignment_file=mmseqs_dir / "alignment.txt",
        alignment_type="mmseqs",
        use_cache=use_cache,
        remove_original=False,
    )
    alignment_to_parquet(
        alignment_file=mmseqs_dir / "reversed_alignment.txt",
        alignment_type="mmseqs",
        use_cache=use_cache,
        remove_original=False,
    )
    mmseqs1 = pd.read_parquet(mmseqs_dir / "alignment.parquet")
    mmseqs2 = pd.read_parquet(mmseqs_dir / "reversed_alignment.parquet")
    mmseqs1.rename(
        {
            "query": "id",
            "target": "apo_monomer_id",
            "qlen": "dimer_len",
            "tlen": "apo_len",
            "pident": "dimer_apo_pident",
            "alnlen": "dimer_apo_alnlen",
        },
        axis=1,
        inplace=True,
    )
    mmseqs2.rename(
        {
            "query": "apo_monomer_id",
            "target": "id",
            "qlen": "apo_len",
            "tlen": "dimer_len",
            "pident": "apo_dimer_pident",
            "alnlen": "apo_dimer_alnlen",
        },
        axis=1,
        inplace=True,
    )
    mmseqs2.loc[:, "pairing_id"] = mmseqs2.id + ":" + mmseqs2.apo_monomer_id
    mmseqs1.loc[:, "pairing_id"] = mmseqs1.id + ":" + mmseqs1.apo_monomer_id
    mmseqs1.drop(["qstart", "qend", "tstart", "tend"], axis=1, inplace=True)
    mmseqs2.drop(["qstart", "qend", "tstart", "tend"], axis=1, inplace=True)
    mmseqs = pd.merge(mmseqs1, mmseqs2, how="outer")
    mmseqs.loc[:, "mmseqs_apo_dimer_frac"] = mmseqs.apo_dimer_alnlen / mmseqs.dimer_len
    mmseqs.loc[:, "mmseqs_dimer_apo_frac"] = mmseqs.dimer_apo_alnlen / mmseqs.dimer_len
    potential_apo.loc[:, "pairing_id"] = (
        potential_apo.id + ":" + potential_apo.apo_monomer_id
    )
    mmseqs.to_parquet("apo_dimer_mmseqs.parquet", index=False)
    mmseqs = mmseqs[
        mmseqs["pairing_id"].isin(set(potential_apo.pairing_id))
    ].reset_index(drop=True)
    apo_mmseqs = pd.merge(
        potential_apo[
            [
                "id",
                "apo_monomer_id",
                "unbound_body",
                "frac_monomer_dimer",
                "frac_monomer_dimer_uniprot",
                "apo_score",
            ]
        ],
        mmseqs,
        how="left",
    )
    apo_mmseqs["mmseqs_dimer_apo_frac"] = apo_mmseqs["mmseqs_dimer_apo_frac"].fillna(
        0.0
    )
    apo_mmseqs["mmseqs_apo_dimer_frac"] = apo_mmseqs["mmseqs_apo_dimer_frac"].fillna(
        0.0
    )
    apo_mmseqs.to_parquet(mmseqs_chkpt, index=False)


def add_all_apo_pairings_to_index(
    pinder_dir: Path,
    config: ApoPairingConfig = ApoPairingConfig(),
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    """Adds all validated apo pairings to the pinder index.

    The index is updated with boolean columns for apo_R/L to indicate whether an apo monomer exists
    for the pinder dimer entry. A canonical apo_R_pdb and apo_L_pdb is selected to use downstream
    when evaluating methods on the test set. Alternative apo structures are stored in a semi-colon separated
    string in the apo_R_pdbs and apo_L_pdbs column for optional usage during e.g. training.

    Args:
        pinder_dir (Path): Directory containing all necessary datasets and configuration files.
        config (ApoPairingConfig): Configuration settings that dictate the process.
        use_cache (bool): If True, will not reprocess if the result is already calculated and stored.

    Returns:
        None: The results are saved in the index and no value is returned.
    """
    apo_checkpoint = pinder_dir / "index_with_apo.parquet"
    if apo_checkpoint.is_file() and use_cache:
        log.info(f"{apo_checkpoint} exists, skipping...")
        return
    scored_chkpt = pinder_dir / "scored_apo_pairings.parquet"
    if scored_chkpt.is_file() and use_cache:
        potential_apo = pd.read_parquet(scored_chkpt)
    else:
        potential_apo = select_potential_apo(
            pinder_dir, config=config, parallel=parallel, max_workers=max_workers
        )
        potential_apo = add_weighted_apo_score(potential_apo, config=config)
        # Not used
        # run_monomer_dimer_mmseqs(pinder_dir, potential_apo, use_cache=use_cache)
        potential_apo.to_parquet(scored_chkpt, index=False)
    potential_apo.loc[:, "apo_pdb"] = potential_apo.apo_monomer_id + ".pdb"
    # Select canonical apo monomer per R/L holo side based on apo score
    canonical = (
        potential_apo.sort_values("apo_score", ascending=False)
        .drop_duplicates(["id", "unbound_body"], keep="first")
        .reset_index(drop=True)
    )
    # Condense all valid pairings per R/L holo side based on available structures
    # Store in order of descreasing apo score.
    condensed_pairings = (
        potential_apo.sort_values(["id", "unbound_body", "apo_score"], ascending=False)
        .groupby(["id", "unbound_body"])["apo_pdb"]
        .apply(";".join)
        .reset_index()
    )
    condensed_pairings = pd.merge(
        condensed_pairings,
        canonical[["id", "unbound_body", "apo_pdb"]].rename(
            {"apo_pdb": "canonical_pdb"}, axis=1
        ),
        how="left",
    )
    # Update apo_R/L, apo_R/L_pdb, and apo_R/L_pdbs columns in index
    index = pd.read_parquet(pinder_dir / "index_with_pred.parquet")
    col_order = list(index.columns)
    index.drop(
        ["apo_R_pdb", "apo_L_pdb", "apo_R_pdbs", "apo_L_pdbs"],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    for side, df in condensed_pairings.groupby("unbound_body"):
        df = df.drop("unbound_body", axis=1).reset_index(drop=True)
        canon_pdb_col = f"apo_{side}_pdb"
        pdbs_col = f"apo_{side}_pdbs"
        rename_cols = {
            "canonical_pdb": canon_pdb_col,
            "apo_pdb": pdbs_col,
        }
        df.rename(rename_cols, axis=1, inplace=True)
        index = pd.merge(index, df, how="left")
        index.loc[index[pdbs_col].isna(), pdbs_col] = ""
        index.loc[index[canon_pdb_col].isna(), canon_pdb_col] = ""
        index.loc[:, f"apo_{side}"] = index[canon_pdb_col] != ""
    index = index[col_order].copy()
    index.to_parquet(apo_checkpoint, index=False)


def collate_apo_metrics(
    metric_dir: Path,
    output_parquet: Path,
) -> None:
    """Collates individual metric files into a single Parquet file for easier management and access.

    Args:
        metric_dir (Path): Directory containing the individual metric Parquet files.
        output_parquet (Path): Path to the output Parquet file where the collated metrics will be stored.

    Returns:
        None: The results are written directly to a Parquet file.
    """
    metric_pqts = list(metric_dir.glob("*.parquet"))
    metrics = []
    for p in metric_pqts:
        try:
            df = pd.read_parquet(p)
            metrics.append(df)
        except Exception as e:
            log.error(f"Failed to read metric parquet {p}: {e}")

    if len(metrics):
        collated = pd.concat(metrics, ignore_index=True)
        collated.to_parquet(output_parquet, index=False)
