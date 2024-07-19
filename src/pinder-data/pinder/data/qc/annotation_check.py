from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from pinder.core import get_metadata
from pinder.core.index.utils import setup_logger
from pinder.data.qc.utils import load_index, load_metadata
from pinder.data.config import ClusterConfig

log = setup_logger(__name__)


def get_paired_uniprot_intersection(
    split_index: pd.DataFrame, against: str = "test"
) -> tuple[set[tuple[str, str]], float]:
    """
    Get the intersection of the uniprot pairs between the train and test/val splits

    Args:
        split_index: the split index dataframe
        against: the split to compare against, either "test" or "val"
    """
    against = against.lower()
    if against not in ["test", "val"]:
        raise ValueError("against must be either 'test' or 'val'")

    # Using vectorized operations to create sorted pairs
    split_index["uniprot_pairs"] = split_index.apply(
        lambda row: tuple(sorted([row["uniprot_R"], row["uniprot_L"]])), axis=1
    )

    # Creating sets of uniprot pairs for 'train' and the chosen 'against' split
    uniprot_train_pairs = set(
        split_index.loc[split_index["split"] == "train", "uniprot_pairs"].unique()
    )
    uniprot_against_pairs = set(
        split_index.loc[split_index["split"] == against, "uniprot_pairs"].unique()
    )

    # Calculating the intersection of pairs
    problem_uniprots = uniprot_against_pairs.intersection(uniprot_train_pairs)

    # Now compute the number of *elements* with problem uniprots
    problem_elements = split_index[
        (split_index["split"] == against)
        & (split_index["uniprot_pairs"].isin(problem_uniprots))
    ]

    # What if we exclude "UNDEFINED?"
    problem_elements_noUNDEF = split_index[
        (split_index["split"] == against)
        & (split_index["uniprot_pairs"].isin(problem_uniprots))
        & ~(split_index["uniprot_pairs"].isin([("UNDEFINED", "UNDEFINED")]))
    ]

    against_count = len(split_index[split_index["split"] == against])
    log.info(
        (
            f"Problem cases: {len(problem_elements)} out of {against_count} "
            f"({100*len(problem_elements)/against_count:.2f}%) in '{against}' split"
        )
    )
    log.info(
        (
            f"Problem cases (excluding UNDEFINED): {len(problem_elements_noUNDEF)} out of {against_count} "
            f"({100*len(problem_elements_noUNDEF)/against_count:.2f}%) in '{against}' split"
        )
    )

    return (
        problem_elements,
        len(problem_elements_noUNDEF) / against_count
        if against_count
        else float("nan"),
    )


def metadata_to_ecod_pairs(
    meta_data: pd.DataFrame, pindex: pd.DataFrame
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Convert the metadata dataframe to a dataframe with ECOD pairs

    Args:
        meta_data: the metadata dataframe
        pindex: the pindex dataframe

    Returns:
        ecod_RL: the ECOD pairs dataframe
        test_ids: the test ids
        val_ids: the val ids
    """
    # Using vectorized operations for interface lengths
    meta_data["chain_R_interface_len"] = (
        meta_data["chain_1_residues"].str.split(",").str.len()
    )
    meta_data["chain_L_interface_len"] = (
        meta_data["chain_2_residues"].str.split(",").str.len()
    )

    # Extracting unique test and validation IDs
    test_ids = set(pindex.loc[pindex["split"] == "test", "id"].unique())
    val_ids = set(pindex.loc[pindex["split"] == "val", "id"].unique())

    # Prepare right and left side ECOD data
    rename_R = {
        "ECOD_names_R": "ECOD_names",
        "ECOD_intersection_R": "ECOD_intersection",
        "chain_R_interface_len": "interface_len",
    }
    rename_L = {
        "ECOD_names_L": "ECOD_names",
        "ECOD_intersection_L": "ECOD_intersection",
        "chain_L_interface_len": "interface_len",
    }

    ecod_R = meta_data[["id"] + list(rename_R.keys())].rename(columns=rename_R)
    ecod_L = meta_data[["id"] + list(rename_L.keys())].rename(columns=rename_L)

    ecod_R["body"] = "R"
    ecod_L["body"] = "L"

    # Combining right and left side data
    ecod_RL = pd.concat([ecod_R, ecod_L], ignore_index=True)

    return ecod_RL, test_ids, val_ids


def annotate_longest_intersecting_ecod_domain(
    ecod_RL: pd.DataFrame,
    min_intersection: int = 10,
    frac_interface_threshold: float = 0.25,
) -> pd.DataFrame:
    """Annotate the longest intersecting ECOD domain for each chain in the pair

    Args:
        ecod_RL: the ECOD annotated dataframe

    Returns:
        ecod_RL_with_ecod: the ECOD annotated dataframe with the longest intersecting domain
    """
    max_intersect = []
    max_intersect_domain = []
    sufficient_overlap = []
    ecod_labels = []

    for ecod_names, ecod_inter, interface_len in zip(
        ecod_RL["ECOD_names"], ecod_RL["ECOD_intersection"], ecod_RL["interface_len"]
    ):
        if isinstance(ecod_inter, str) and ecod_inter != "":
            intersections = list(map(int, ecod_inter.strip().split(",")))
            largest_domain = ecod_names.split(",")[np.argmax(intersections)]
            largest_intersect = max(intersections)
            sufficient = (
                largest_intersect > min_intersection
                and (largest_intersect / interface_len) > frac_interface_threshold
            )
            if sufficient:
                ecod_label = largest_domain
            else:
                ecod_label = ""
        else:
            largest_intersect = 0
            largest_domain = ""
            sufficient = False
            ecod_label = ""

        max_intersect.append(largest_intersect)
        max_intersect_domain.append(largest_domain)
        sufficient_overlap.append(sufficient)
        ecod_labels.append(ecod_label)

    ecod_RL_with_ecod = ecod_RL.copy()
    ecod_RL_with_ecod.loc[:, "ecod_domain"] = max_intersect_domain
    ecod_RL_with_ecod.loc[:, "ecod_domain_len"] = max_intersect
    ecod_RL_with_ecod.loc[:, "sufficient_overlap"] = sufficient_overlap
    ecod_RL_with_ecod.loc[:, "ecod_label"] = ecod_labels

    ecod_R = (
        ecod_RL_with_ecod.query("body == 'R'")
        .drop(["body", "ECOD_names", "ECOD_intersection"], axis=1)
        .reset_index(drop=True)
    )
    ecod_L = (
        ecod_RL_with_ecod.query("body == 'L'")
        .drop(["body", "ECOD_names", "ECOD_intersection"], axis=1)
        .reset_index(drop=True)
    )

    ecod_R.rename(
        {c: f"{c}_R" for c in ecod_R.columns if c != "id"}, axis=1, inplace=True
    )
    ecod_L.rename(
        {c: f"{c}_L" for c in ecod_L.columns if c != "id"}, axis=1, inplace=True
    )

    return ecod_RL_with_ecod, ecod_R, ecod_L


def merge_annotated_interfaces_with_pindex(
    pindex: pd.DataFrame, ecod_R: pd.DataFrame, ecod_L: pd.DataFrame
) -> pd.DataFrame:
    """Merge the annotated interfaces with the pindex dataframe

    Args:
        pindex: the pindex dataframe
        ecod_R: the ECOD annotated dataframe for chain R
        ecod_L: the ECOD annotated dataframe for chain L

    Returns:
        pindex_with_RL: the pindex dataframe with the ECOD annotations

    """
    pindex_with_RL = pd.merge(pindex, ecod_R, how="left")
    pindex_with_RL = pd.merge(pindex_with_RL, ecod_L, how="left")

    str_cols = ["ecod_domain_R", "ecod_label_R", "ecod_domain_L", "ecod_label_L"]
    int_cols = [
        "interface_len_R",
        "interface_len_L",
        "ecod_domain_len_R",
        "ecod_domain_len_L",
    ]

    for c in str_cols:
        pindex_with_RL.loc[pindex_with_RL[c].isna(), c] = ""
    for c in int_cols:
        pindex_with_RL.loc[pindex_with_RL[c].isna(), c] = 0

    pindex_with_RL.loc[
        pindex_with_RL.sufficient_overlap_R.isna(), "sufficient_overlap_R"
    ] = False
    pindex_with_RL.loc[
        pindex_with_RL.sufficient_overlap_L.isna(), "sufficient_overlap_L"
    ] = False

    pindex_with_RL.loc[:, "ecod_pair"] = [
        "--".join(sorted([r_lab, l_lab]))
        for r_lab, l_lab in zip(
            pindex_with_RL["ecod_label_R"], pindex_with_RL["ecod_label_L"]
        )
    ]
    pindex_with_RL["ecod_overlap_ratio_R"] = (
        pindex_with_RL["ecod_domain_len_R"] / pindex_with_RL["interface_len_R"]
    )
    pindex_with_RL["ecod_overlap_ratio_L"] = (
        pindex_with_RL["ecod_domain_len_L"] / pindex_with_RL["interface_len_L"]
    )
    return pindex_with_RL


def get_ecod_paired_leakage(
    metadata: pd.DataFrame,
    pindex: pd.DataFrame,
    frac_interface_threshold: float = 0.25,
    min_intersection: int = 10,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the ECOD paired leakage between the test and train/val splits

    Args:
        metadata: the metadata dataframe
        pindex: the pindex dataframe
        frac_interface_threshold: the fraction of the interface that must be covered by the ECOD domain
        min_intersection: int
            the minimum length of an interface
        top_n: int
            The number of leaked pairs to include in the output dataframe

    Returns:
        problem_cases_test: the problem cases (potentially leaked ECOD accession pairs) in the test split
        problem_cases_val: the problem cases (potentially leaked ECOD accession pairs) in the val split
    """
    ecod_RL, _, _ = metadata_to_ecod_pairs(metadata, pindex)
    ecod_RL, ecod_R, ecod_L = annotate_longest_intersecting_ecod_domain(
        ecod_RL,
        frac_interface_threshold=frac_interface_threshold,
        min_intersection=min_intersection,
    )

    pindex_with_RL = merge_annotated_interfaces_with_pindex(pindex, ecod_R, ecod_L)

    unknown_domains = ["", "F_UNCLASSIFIED", "UNK_F_TYPE"]
    af2_test_ecod_pairs = set(
        pindex_with_RL.query(
            f"pinder_af2 and ecod_label_R not in {unknown_domains} and ecod_label_L not in {unknown_domains}"
        ).ecod_pair
    )
    af2_train_ecod_pairs = set(
        pindex_with_RL.query(
            f"af2_train and ecod_label_R not in {unknown_domains} and ecod_label_L not in {unknown_domains}"
        ).ecod_pair
    )

    pindex_nonpeptide = pindex_with_RL[
        ~(pindex_with_RL["cluster_id"].str.contains("-1", regex=False))
        & ~(pindex_with_RL["cluster_id"].str.contains("p_p", regex=False))
    ]
    test_ecod_pairs = set(
        pindex_nonpeptide.query(
            f'split == "test" and ecod_label_R not in {unknown_domains} and ecod_label_L not in {unknown_domains}'
        ).ecod_pair
    )
    val_ecod_pairs = set(
        pindex_nonpeptide.query(
            f'split == "val" and ecod_label_R not in {unknown_domains} and ecod_label_L not in {unknown_domains}'
        ).ecod_pair
    )

    train_ecod_pairs = set(
        pindex_nonpeptide.query(
            f'split == "train" and ecod_label_R not in {unknown_domains} and ecod_label_L not in {unknown_domains}'
        ).ecod_pair
    )
    log.info(
        (
            f"Total unique ECOD pairs: Test = {len(test_ecod_pairs)}, "
            f"Val = {len(val_ecod_pairs)}, "
            f"Train = {len(train_ecod_pairs)}, "
            f"AF2-Test = {len(af2_test_ecod_pairs)}, "
            f"AF2-Train = {len(af2_train_ecod_pairs)} "
        )
    )
    problem_cases_test = train_ecod_pairs.intersection(test_ecod_pairs)
    problem_cases_val = train_ecod_pairs.intersection(val_ecod_pairs)
    problem_cases_af2 = af2_train_ecod_pairs.intersection(af2_test_ecod_pairs)

    # What percent of the test set has ecod overlap with train?
    num_test_with_problems = pindex_nonpeptide[
        (pindex_nonpeptide["split"] == "test")
        & (pindex_nonpeptide["ecod_pair"].isin(problem_cases_test))
    ].shape[0]
    num_test_total = pindex_nonpeptide.query('split == "test"').shape[0]
    pct_test_overlap = 100 * num_test_with_problems / num_test_total
    log.info(
        (
            f"Percent of test set with same ECOD pair as train:"
            f" {pct_test_overlap:.2f}% ({num_test_with_problems}/{num_test_total})"
        )
    )

    num_af_with_problems = pindex_with_RL[
        (pindex_with_RL["pinder_af2"])
        & (pindex_with_RL["ecod_pair"].isin(problem_cases_af2))
    ].shape[0]
    num_af_total = pindex_with_RL.query("pinder_af2").shape[0]
    pct_af_overlap = 100 * num_af_with_problems / num_af_total
    log.info(
        (
            f"Percent of PINDER-AF2 set with same ECOD pair as AF2-train:"
            f" {pct_af_overlap:.2f}% ({num_af_with_problems}/{num_af_total})"
        )
    )

    # What percent of the val set has ecod overlap with train?
    num_val_with_problems = pindex_nonpeptide[
        (pindex_nonpeptide["split"] == "val")
        & (pindex_nonpeptide["ecod_pair"].isin(problem_cases_val))
    ].shape[0]
    num_val_total = pindex_nonpeptide.query('split == "val"').shape[0]
    pct_val_overlap = 100 * num_val_with_problems / num_val_total
    log.info(
        (
            f"Percent of val set with same ECOD pair as train:"
            f" {pct_val_overlap:.2f}% ({num_val_with_problems}/{num_val_total})"
        )
    )
    report = pd.DataFrame(
        [
            {
                "Measure": "Leakage",
                "Metric": "ECOD pair",
                "Test": pct_test_overlap,
                "Val": pct_val_overlap,
            },
        ]
    )

    df_problem_cases_af2 = (
        pindex_with_RL[(pindex_with_RL["ecod_pair"].isin(problem_cases_af2))]
        .groupby(["ecod_pair", "pinder_af2", "split"], observed=True)
        .head(top_n if top_n is not None else pindex_with_RL.shape[0])
        .sort_values(by=["ecod_pair", "split"])
    )

    df_problem_cases_test = (
        pindex_nonpeptide[(pindex_nonpeptide["ecod_pair"].isin(problem_cases_test))]
        .groupby(["ecod_pair", "split"], observed=True)
        .head(top_n if top_n is not None else pindex_with_RL.shape[0])
        .sort_values(by=["ecod_pair", "split"])
    )

    df_problem_cases_val = (
        pindex_nonpeptide[(pindex_nonpeptide["ecod_pair"].isin(problem_cases_val))]
        .groupby(["ecod_pair", "split"], observed=True)
        .head(top_n if top_n is not None else pindex_with_RL.shape[0])
        .sort_values(by=["ecod_pair", "split"])
    )

    return (
        df_problem_cases_test,
        df_problem_cases_val,
        df_problem_cases_af2,
        pindex_with_RL,
        report,
    )


def get_binding_leakage(
    annotated_pindex: pd.DataFrame, against: str = "test"
) -> tuple[pd.DataFrame, float]:
    """Obtains the chain level binding site leakage between the test/val and train splits"""
    against = against.lower()
    assert against.lower() in [
        "test",
        "val",
        "pinder_af2",
    ], "against must be either test, val or pinder_af2"
    assert "ecod_label_R" in annotated_pindex.columns, "ecod_label_R not in pindex"
    assert "ecod_label_L" in annotated_pindex.columns, "ecod_label_L not in pindex"

    annotated_pindex_all = annotated_pindex.copy()
    annotated_pindex = annotated_pindex[
        ~(annotated_pindex["cluster_id"].str.contains("-1", regex=False))
        & ~(annotated_pindex["cluster_id"].str.contains("p_p", regex=False))
    ]

    ecod_labels: dict[str, set[str]] = {}
    for split_type in ["train", "val", "test", "pinder_af2", "af2_train"]:
        if "af2" in split_type:
            data = annotated_pindex_all
            mask = data[split_type]
        else:
            data = annotated_pindex
            mask = data["split"] == split_type
        split_data = data[mask]
        ecod_labels[split_type] = set(split_data.ecod_label_R).union(
            set(split_data.ecod_label_L)
        )

    # Report this:
    log.info(
        (
            "Total unique ECOD domains: "
            f"Test = {len(ecod_labels['test'])}, "
            f"Val = {len(ecod_labels['val'])}, "
            f"Train = {len(ecod_labels['train'])}, "
            f"AF2-Test = {len(ecod_labels['pinder_af2'])}, "
            f"AF2-Train = {len(ecod_labels['af2_train'])}"
        )
    )

    # We want to know how many unique ecods we have in our test/val set
    if against in ["test", "val"]:
        pindex_df = annotated_pindex.copy()
        against_mask = pindex_df["split"] == against
        # unique_test_val_ecods
        unique_ecod_labels = (set(ecod_labels["test"]) | set(ecod_labels["val"])) - set(
            ecod_labels["train"]
        )
    else:
        pindex_df = annotated_pindex_all.copy()
        against_mask = pindex_df["pinder_af2"]
        # unique pinder_af2 ecods
        unique_ecod_labels = (set(ecod_labels["pinder_af2"])) - set(
            ecod_labels["af2_train"]
        )

    # How many test set members have no ECOD not in train?
    no_unique_ecod = pindex_df[
        against_mask
        & (
            ~pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            & ~pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
    ].shape[0]
    # How many test set members have exactly one ECOD not in train?
    one_unique_ecod = pindex_df[
        against_mask
        & (
            pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            ^ pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
    ].shape[0]
    # How many test set members have exactly one ECOD not in train?
    both_unique_ecod_df = pindex_df[
        against_mask
        & (
            pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            & pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
    ]
    both_unique_ecod = both_unique_ecod_df.shape[0]
    # What is the size of our set?
    against_size = pindex_df[against_mask].shape[0]

    log.info(f"Size of {against} set: {against_size}")
    log.info(
        (
            f"Percent of {against} set with <0, 1, 2> unique ECOD:\t "
            f"<{100*no_unique_ecod/against_size:.2f}%, "
            f"{100*one_unique_ecod/against_size:.2f}%, "
            f"{100*both_unique_ecod/against_size:.2f}%>"
        )
    )
    pair_ecod_diversity = 100 * both_unique_ecod / against_size
    log.info(
        (
            f"Percent of {against} set with at least 1 unique ECOD:\t "
            f"{100*(one_unique_ecod + both_unique_ecod)/against_size:.2f}%"
        )
    )

    ## Now to look at heterodimers
    # How many heterdimers are there, anyway?
    num_het = pindex_df[
        against_mask & (pindex_df["uniprot_L"] != pindex_df["uniprot_R"])
    ].shape[0]
    # How many test set heterodimers have no ECOD not in train?
    het_no_unique_ecod = pindex_df[
        against_mask
        & (
            ~pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            & ~pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
        & (pindex_df["uniprot_L"] != pindex_df["uniprot_R"])
    ].shape[0]

    # How many test set heterodimers have exactly one ECOD not in train?
    het_one_unique_ecod = pindex_df[
        against_mask
        & (
            pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            ^ pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
        & (pindex_df["uniprot_L"] != pindex_df["uniprot_R"])
    ].shape[0]

    # How many test set heterodimers have exactly one ECOD not in train?
    het_both_unique_ecod = pindex_df[
        against_mask
        & (
            pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            & pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
        & (pindex_df["uniprot_L"] != pindex_df["uniprot_R"])
    ].shape[0]

    log.info(
        (
            f"Percent of {against} set that are heterodimers: \t "
            f"{100*num_het/against_size:.2f}%"
        )
    )
    log.info(
        (
            f"Percent of {against} set that are heterodimers with <0, 1, 2> unique ECOD:\t "
            f"<{100*het_no_unique_ecod/against_size:.2f}%, "
            f"{100*het_one_unique_ecod/against_size:.2f}%, "
            f"{100*het_both_unique_ecod/against_size:.2f}%>"
        )
    )
    log.info(
        (
            f"Percent of {against} set that are heterodimers with at least 1 unique ECOD:\t "
            f"{100*(het_one_unique_ecod + het_both_unique_ecod)/against_size:.2f}%"
        )
    )

    # We are interested in the "at least one unique" set, mostly
    test_unique_ecod_df = pindex_df[
        against_mask
        & (
            pindex_df["ecod_label_R"].isin(unique_ecod_labels)
            | pindex_df["ecod_label_L"].isin(unique_ecod_labels)
        )
    ]
    return test_unique_ecod_df, pair_ecod_diversity


def binding_leakage_main(
    index_file: str | None = None,
    metadata_file: str | None = None,
    frac_interface_threshold: float = 0.25,
    min_intersection: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract ECOD paired binding site leakage for test and val splits.

    Parameters:
        index_file (str | None): Path to custom/intermediate index file, if not provided will use get_index().
        metadata_file (str | None): Path to custom/intermediate metadata file, if not provided will use get_metadata().
        frac_interface_threshold (float): Fraction of interface required to be covered by ECOD domain.
            Default is 0.25.
        min_intersection (int): Minimum required intersection between interface and ECOD domain.
            Default is 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The test and val split binding leakage, respectively.

    """
    log.info(
        "Starting ECOD paired binding leakage analysis script, parsing arguments..."
    )
    pindex = load_index(index_file)
    metadata = load_metadata(metadata_file)

    pindex.drop("date", axis=1, inplace=True, errors="ignore")
    meta = get_metadata()

    pindex = pd.merge(pindex, meta[["id", "date"]], how="left")
    pindex["date"] = (
        pindex["date"].astype("object").apply(lambda x: date.fromisoformat(x))
    )
    af2_date = date.fromisoformat(ClusterConfig().alphafold_cutoff_date)
    pindex["af2_train"] = pindex["date"] <= af2_date

    # Get the leakage between train and [val, test]
    all_splits = ["train", "test", "val", "invalid"]
    pindex_all_splits = pindex[pindex["split"].isin(all_splits)].reset_index(drop=True)
    (
        test_problems,
        val_problems,
        af2_problems,
        pindex_RL,
        leak_report,
    ) = get_ecod_paired_leakage(
        metadata,
        pindex_all_splits,
        frac_interface_threshold=frac_interface_threshold,
        min_intersection=min_intersection,
    )
    test_at_least_one_unique, test_ecod_pair_diversity = get_binding_leakage(
        pindex_RL, against="test"
    )
    val_at_least_one_unique, val_ecod_pair_diversity = get_binding_leakage(
        pindex_RL, against="val"
    )
    af2_at_least_one_unique, af2_ecod_pair_diversity = get_binding_leakage(
        pindex_RL, against="pinder_af2"
    )
    report = pd.DataFrame(
        [
            {
                "Measure": "Diversity",
                "Metric": "Unique ECOD pairs",
                "Test": test_ecod_pair_diversity,
                "Val": val_ecod_pair_diversity,
            },
        ]
    )
    report = pd.concat([report, leak_report], ignore_index=True)
    return (
        test_at_least_one_unique,
        val_at_least_one_unique,
        af2_at_least_one_unique,
        report,
    )
