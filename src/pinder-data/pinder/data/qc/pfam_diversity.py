from __future__ import annotations
import logging
from string import digits
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pinder.core import get_pinder_location
from pinder.core.utils import setup_logger

from pinder.data.qc.annotation_check import (
    metadata_to_ecod_pairs,
    annotate_longest_intersecting_ecod_domain,
    merge_annotated_interfaces_with_pindex,
)
from pinder.data.qc.utils import load_index, load_metadata, load_pfam_db


log = setup_logger(__name__, log_level=logging.INFO)


def load_data(
    index_file: Path | str | None = None,
    metadata_file: Path | str | None = None,
    pfam_file: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading data...")
    pindex = load_index(index_file)
    metadata = load_metadata(metadata_file)
    pfam_data = load_pfam_db(pfam_file)
    return pindex, metadata, pfam_data


def get_ecod_annotations(
    pindex: pd.DataFrame,
    metadata: pd.DataFrame,
    frac_interface_threshold: float = 0.25,
    min_intersection: int = 10,
) -> pd.DataFrame:
    ecod_RL, _, _ = metadata_to_ecod_pairs(metadata, pindex)
    ecod_RL, ecod_R, ecod_L = annotate_longest_intersecting_ecod_domain(
        ecod_RL,
        frac_interface_threshold=frac_interface_threshold,
        min_intersection=min_intersection,
    )

    pindex_with_RL = merge_annotated_interfaces_with_pindex(pindex, ecod_R, ecod_L)
    return pindex_with_RL


def process_pfam_data(pindex: pd.DataFrame, pfam_data: pd.DataFrame) -> pd.DataFrame:
    log.info("Processing Pfam data...")
    pdb_id_chain_to_pfam = pfam_data["Clan_ID"].to_dict()
    pindex_R_asym = pindex["chain_R"].apply(lambda x: x.rstrip(digits))
    pindex_L_asym = pindex["chain_L"].apply(lambda x: x.rstrip(digits))

    pindex["pfam_clan_R"] = [
        pdb_id_chain_to_pfam.get((x, y), "no_clan")
        for x, y in zip(pindex["pdb_id"], pindex_R_asym)
    ]
    pindex["pfam_clan_L"] = [
        pdb_id_chain_to_pfam.get((x, y), "no_clan")
        for x, y in zip(pindex["pdb_id"], pindex_L_asym)
    ]
    return pindex


def visualize_data(
    pindex: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    top_n_pfams: int = 50,
) -> None:
    # Visualization blocks as functions
    log.info("Generating visualizations...")
    plot_pfam_clan_distribution(pindex, output_dir, top_n_pfams)
    report_ecod_diversity(pindex)
    plot_ecod_family_distribution(pindex, output_dir, top_n_pfams)
    plot_merged_metadata_distributions(pindex, metadata, output_dir)


def get_pfam_diversity(
    pindex: pd.DataFrame,
) -> pd.DataFrame:
    test_table = pindex[pindex["split"] == "test"]
    val_table = pindex[pindex["split"] == "val"]
    pfam_test_clans = set(test_table["pfam_clan_R"].unique()) | set(
        test_table["pfam_clan_L"].unique()
    )
    pfam_val_clans = set(val_table["pfam_clan_R"].unique()) | set(
        val_table["pfam_clan_L"].unique()
    )
    pfam_all_clans = set(pindex["pfam_clan_R"].unique()) | set(
        pindex["pfam_clan_L"].unique()
    )
    pfam_test = len(pfam_test_clans) / len(pfam_all_clans) * 100
    pfam_val = len(pfam_val_clans) / len(pfam_all_clans) * 100
    log.info(f"Diversity of Pfam-clans in test set: {pfam_test:.2f}%")
    log.info(f"Diversity of Pfam-clans in val set: {pfam_val:.2f}%")
    report = pd.DataFrame(
        [
            {
                "Measure": "Diversity",
                "Metric": "Pfam clan",
                "Test": pfam_test,
                "Val": pfam_val,
            },
        ]
    )
    return report


def plot_pfam_clan_distribution(
    pindex: pd.DataFrame,
    output_dir: Path,
    top_n_pfams: int = 50,
) -> None:
    log.info("Generating Pfam clan distribution plot")
    output_dir = Path(output_dir)
    test_table = pindex[pindex["split"] == "test"]
    _ = get_pfam_diversity(pindex)
    test_pfams_concat = pd.concat(
        (test_table["pfam_clan_R"], test_table["pfam_clan_L"])
    )
    plt.figure(figsize=(10, 10))
    test_pfams_concat[test_pfams_concat != "no_clan"].value_counts().sort_values(
        ascending=False
    )[:top_n_pfams].plot(kind="barh")
    plt.title(f"Top {top_n_pfams} Pfam clans in test set")
    plt.savefig(output_dir / f"top_{top_n_pfams}_pfam_clans.png")


def report_ecod_diversity(pindex: pd.DataFrame) -> pd.DataFrame:
    test_table = pindex[pindex["split"] == "test"]
    val_table = pindex[pindex["split"] == "val"]

    indiv_ecod_test_ids = set(test_table["ecod_label_L"].unique()) | set(
        test_table["ecod_label_R"].unique()
    )

    indiv_ecod_val_ids = set(val_table["ecod_label_L"].unique()) | set(
        val_table["ecod_label_R"].unique()
    )

    indiv_ecod_all_ids = set(pindex["ecod_label_L"].unique()) | set(
        pindex["ecod_label_R"].unique()
    )
    pair_ecod_test_ids = set(test_table["ecod_pair"].unique())
    pair_ecod_val_ids = set(val_table["ecod_pair"].unique())
    pair_ecod_all_ids = set(pindex["ecod_pair"].unique())

    single_chain_test = len(indiv_ecod_test_ids) / len(indiv_ecod_all_ids) * 100
    single_chain_val = len(indiv_ecod_val_ids) / len(indiv_ecod_all_ids) * 100
    single_chain_test_val = (
        len(indiv_ecod_test_ids | indiv_ecod_val_ids) / len(indiv_ecod_all_ids) * 100
    )
    pair_test = len(pair_ecod_test_ids) / len(pair_ecod_all_ids) * 100
    pair_val = len(pair_ecod_val_ids) / len(pair_ecod_all_ids) * 100
    pair_test_val = (
        len(pair_ecod_test_ids | pair_ecod_val_ids) / len(pair_ecod_all_ids) * 100
    )

    log.info(
        (
            "Single-chain diversity of ECOD families in <test, val, test+val> set: "
            f"<{single_chain_test:.2f}%, "
            f"{single_chain_val:.2f}%, "
            f"{single_chain_test_val:.2f}%>"
        )
    )
    log.info(
        (
            "Pair diversity of ECOD families in <test, val, test+val> set: "
            f"<{pair_test:.2f}%, "
            f"{pair_val:.2f}%, "
            f"{pair_test_val:.2f}%>"
        )
    )
    report = pd.DataFrame(
        [
            {
                "Measure": "Diversity",
                "Metric": "ECOD single-chain",
                "Test": single_chain_test,
                "Val": single_chain_val,
            },
            {
                "Measure": "Diversity",
                "Metric": "ECOD chain-pair",
                "Test": pair_test,
                "Val": pair_val,
            },
        ]
    )
    return report


def plot_ecod_family_distribution(
    pindex: pd.DataFrame,
    output_dir: Path,
    top_n_pfams: int = 50,
) -> None:
    log.info("Generating ECOD family distribution plot")
    output_dir = Path(output_dir)
    test_table = pindex[pindex["split"] == "test"]
    test_ecods_concat = pd.concat(
        (test_table["ecod_label_R"], test_table["ecod_label_L"])
    )
    plt.figure(figsize=(10, 10))

    # These ECOD domains should not be considered:
    unknown_domains = ["", "F_UNCLASSIFIED", "UNK_F_TYPE"]

    test_ecods_concat[
        ~test_ecods_concat.isin(unknown_domains)
    ].value_counts().sort_values(ascending=False)[:top_n_pfams].plot(kind="barh")
    plt.title(f"Top {top_n_pfams} ECOD families in test set")
    plt.savefig(output_dir / f"top_{top_n_pfams}_ecod_families.png")


def get_merged_metadata(pindex: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged_metadata = pd.merge(
        pindex[["id", "cluster_id", "split"]], metadata, on="id", how="inner"
    )
    merged_metadata.loc[:, "interface_lengths"] = merged_metadata[
        "chain_1_residues"
    ].apply(lambda x: len(x.split(","))) + merged_metadata["chain_2_residues"].apply(
        lambda x: len(x.split(","))
    )
    # Cluster Level Diversity
    merged_metadata["min_chain_len"] = [
        min(x, y)
        for x, y in zip(merged_metadata["length1"], merged_metadata["length2"])
    ]
    return merged_metadata


def get_cluster_diversity(merged_metadata: pd.DataFrame) -> pd.DataFrame:
    diversity_test_val = (
        merged_metadata[
            (merged_metadata["split"] == "test") | (merged_metadata["split"] == "val")
        ]["cluster_id"].nunique()
        / merged_metadata[merged_metadata["min_chain_len"] > 40]["cluster_id"].nunique()
        * 100
    )
    diversity_test = (
        merged_metadata[(merged_metadata["split"] == "test")]["cluster_id"].nunique()
        / merged_metadata[merged_metadata["min_chain_len"] > 40]["cluster_id"].nunique()
        * 100
    )
    diversity_val = (
        merged_metadata[(merged_metadata["split"] == "val")]["cluster_id"].nunique()
        / merged_metadata[merged_metadata["min_chain_len"] > 40]["cluster_id"].nunique()
        * 100
    )
    log.info(
        (
            "Pinder cluster level diversity in <test, val, test+val> set: "
            f"<{diversity_test:.2f}%, "
            f"{diversity_val:.2f}%, "
            f"{diversity_test_val:.2f}%>"
        )
    )
    report = pd.DataFrame(
        [
            {
                "Measure": "Diversity",
                "Metric": "Cluster",
                "Test": diversity_test,
                "Val": diversity_val,
            },
        ]
    )
    return report


def plot_merged_metadata_distributions(
    pindex: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    merged_metadata = get_merged_metadata(pindex, metadata)
    # Interface Length Distribution
    log.info("Generating interface length distribution plot")
    plt.figure(figsize=(10, 7))
    bins = np.linspace(0, 500, 50)
    merged_metadata[merged_metadata["split"] == "test"]["interface_lengths"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="blue",
        label="test",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "train"]["interface_lengths"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="red",
        label="train",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "val"]["interface_lengths"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="green",
        label="val",
        legend=True,
        density=True,
    )
    plt.title("Interface Res-length Distribution")
    plt.xlabel("Interface Res-length")
    plt.ylabel("Density")
    plt.savefig(output_dir / "interface_length_distribution.png")

    # Intermolecular Contacts Distribution
    log.info("Generating intermolecular contacts distribution plot")
    min_val = merged_metadata["intermolecular_contacts"].min()
    max_val = merged_metadata[
        "intermolecular_contacts"
    ].max()  # Adjusted from the example for full range
    bins = np.linspace(min_val, max_val, 50)
    plt.figure(figsize=(10, 7))
    merged_metadata[merged_metadata["split"] == "test"]["intermolecular_contacts"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="blue",
        label="test",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "train"][
        "intermolecular_contacts"
    ].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="red",
        label="train",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "val"]["intermolecular_contacts"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="green",
        label="val",
        legend=True,
        density=True,
    )
    plt.title("Intermolecular Contacts Distribution")
    plt.xlabel("Intermolecular Contacts")
    plt.ylabel("Density")
    plt.savefig(output_dir / "intermolecular_contacts_distribution.png")

    # Do it again for comparison to previous example
    bins = np.linspace(min_val, 500, 50)
    plt.figure(figsize=(10, 7))
    merged_metadata[merged_metadata["split"] == "test"]["intermolecular_contacts"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="blue",
        label="test",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "train"][
        "intermolecular_contacts"
    ].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="red",
        label="train",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "val"]["intermolecular_contacts"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="green",
        label="val",
        legend=True,
        density=True,
    )
    plt.title("Intermolecular Contacts Distribution (maximum 500)")
    plt.xlabel("Intermolecular Contacts")
    plt.ylabel("Density")
    plt.savefig(output_dir / "intermolecular_contacts_distribution_max500.png")

    # Probability Distribution
    log.info("Generating probability distribution plot")
    bins = np.linspace(0, 1, 50)
    plt.figure(figsize=(10, 7))
    merged_metadata[merged_metadata["split"] == "test"]["probability"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="blue",
        label="test",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "train"]["probability"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="red",
        label="train",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "val"]["probability"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="green",
        label="val",
        legend=True,
        density=True,
    )
    plt.title("BIO Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.savefig(output_dir / "bio_probability_distributions.png")

    # Planarity Distribution
    log.info("Generating planarity distribution plot")
    min_val = merged_metadata["planarity"].min()
    max_val = merged_metadata["planarity"].max()
    log.info(f"Planarity range: {min_val} - {max_val}")
    bins = np.linspace(min_val, max_val, 50)
    plt.figure(figsize=(10, 7))
    merged_metadata[merged_metadata["split"] == "test"]["planarity"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="blue",
        label="test",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "train"]["planarity"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="red",
        label="train",
        legend=True,
        density=True,
    )
    merged_metadata[merged_metadata["split"] == "val"]["planarity"].plot(
        kind="hist",
        bins=bins,
        alpha=0.5,
        color="green",
        label="val",
        legend=True,
        density=True,
    )
    plt.title("Planarity Distribution")
    plt.xlabel("Planarity")
    plt.ylabel("Density")
    plt.savefig(output_dir / "planarity_distribution.png")

    # Planarity vs Interface Length Correlation
    sns.scatterplot(
        data=pd.concat(
            [
                merged_metadata[
                    (merged_metadata["split"] == "train")
                    & (merged_metadata["probability"] > 0.2)
                ].sample(frac=0.05),
                merged_metadata[
                    (merged_metadata["split"] == "val")
                    & (merged_metadata["probability"] > 0.5)
                ],
                merged_metadata[
                    (merged_metadata["split"] == "test")
                    & (merged_metadata["probability"] > 0.5)
                ],
            ]
        ),
        x="probability",
        y="planarity",
        hue="split",
        alpha=0.3,
    )
    plt.title("Planarity vs Interface Length Correlation")
    plt.savefig(output_dir / "planarity_vs_interface_length_correlation.png")

    # Interface Length Distribution for different oligomeric states
    plt.figure(figsize=(10, 7))
    sns.histplot(
        data=merged_metadata[(merged_metadata["oligomeric_count"] < 10)],
        x="interface_lengths",
        hue="oligomeric_count",
        bins=np.linspace(0, 500, 10),
        multiple="stack",
        palette="viridis",
    )
    plt.title("Interface Length Distribution for Different Oligomeric States")
    plt.savefig(output_dir / "interface_length_distribution_oligomeric_states.png")

    # Cluster Level Diversity
    get_cluster_diversity(merged_metadata)


def pfam_diversity_main(
    index_file: str | None = None,
    metadata_file: str | None = None,
    pfam_file: str | None = None,
    output_dir: str | Path = get_pinder_location() / "data/pfam_visualizations",
    frac_interface_threshold: float = 0.25,
    min_intersection: int = 10,
    top_n_pfams: int = 50,
) -> None:
    """Extract PFAM clan diversity and generate visualizations.

    Parameters:
        index_file (str | None): Path to custom/intermediate index file, if not provided will use get_index().
        metadata_file (str | None): Path to custom/intermediate metadata file, if not provided will use get_metadata().
        pfam_file (str | None): Optional path to Pfam data file, if not provided will fetch and cache it.
        output_dir (str | Path): Directory to store PFAM visualizations. Defaults to get_pinder_location() / 'data/pfam_visualizations'.
        frac_interface_threshold (float): Fraction of interface required to be covered by ECOD domain.
            Default is 0.25.
        min_intersection (int): Minimum required intersection between interface and ECOD domain.
            Default is 10.
        top_n_pfams (int): Number of top Pfam clans to visualize. Default is 50.

    Returns:
        None.

    """
    log.info("Starting diversity analysis script, parsing arguments...")
    pindex, metadata, pfam_data = load_data(
        index_file=index_file, metadata_file=metadata_file, pfam_file=pfam_file
    )
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    all_splits = ["train", "test", "val", "invalid"]
    pindex_all_splits = pindex[
        (pindex["split"].isin(all_splits))
        & ~(pindex["cluster_id"].str.contains("-1", regex=False))
        & ~(pindex["cluster_id"].str.contains("p_p", regex=False))
    ]

    log.info("Data loaded successfully.")

    log.info("Creating ECOD annotations")
    pindex_with_RL = get_ecod_annotations(
        pindex_all_splits,
        metadata,
        frac_interface_threshold=frac_interface_threshold,
        min_intersection=min_intersection,
    )

    pindex_pfam = process_pfam_data(pindex_with_RL, pfam_data)
    log.info("Pfam data processed successfully.")

    # make sure output dir exists
    if not output_dir.is_dir():
        output_dir.mkdir(exist_ok=True, parents=True)

    visualize_data(pindex_pfam, metadata, output_dir, top_n_pfams=top_n_pfams)
    log.info("Visualizations generated successfully.")
