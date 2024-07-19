from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from tqdm import tqdm

from pinder.core.utils import setup_logger
from pinder.core.index.utils import (
    MetadataEntry,
    IndexEntry,
    downcast_dtypes,
    get_pinder_bucket_root,
    get_supplementary_data,
)
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.config import ApoPairingConfig, ClusterConfig, get_config_hash
from pinder.data.get_apo import add_weighted_apo_score
from pinder.data.pipeline.constants import CONSIDER_LEAKED, TEST_SYSTEM_BLACKLIST


log = setup_logger(__name__)


def print_test_meta_details(test_meta: pd.DataFrame) -> None:
    """
    This function will print out the details of the test set that was generated
    by the get_split_subsets function.
    """

    total_testval_num = test_meta[test_meta["split"] == "proto-test"].shape[0]
    num_apos = test_meta[
        (test_meta["split"] == "proto-test") & (test_meta["min_both_putative_apo"] != 0)
    ].shape[0]
    num_pred = test_meta[
        (test_meta["split"] == "proto-test") & (test_meta["min_both_pred"] != 0)
    ].shape[0]
    num_heterodimer = test_meta[
        (test_meta["split"] == "proto-test") & (test_meta["heterodimer"])
    ].shape[0]
    num_homodimer = test_meta[
        (test_meta["split"] == "proto-test") & (~test_meta["heterodimer"])
    ].shape[0]

    num_apos_af2 = test_meta[
        (test_meta["af2_mm"]) & (test_meta["min_both_putative_apo"] != 0)
    ].shape[0]
    num_pred_af2 = test_meta[
        (test_meta["af2_mm"]) & (test_meta["min_both_pred"] != 0)
    ].shape[0]
    num_heterodimer_af2 = test_meta[
        (test_meta["af2_mm"]) & (test_meta["heterodimer"])
    ].shape[0]
    num_homodimer_af2 = test_meta[
        (test_meta["af2_mm"]) & (~test_meta["heterodimer"])
    ].shape[0]

    total_removed = test_meta[test_meta["split"] == "proto-test"][
        "depth_2_hits_with_comm"
    ].sum()

    print(
        f"total removed data: {total_removed}\n"
        f"total number of test+val systems: {total_testval_num}\n\n",
        f"number of test+val apos: {num_apos}\n",
        f"number of test+val preds: {num_pred}\n",
        f"number of test+val homodimers: {num_homodimer}\n",
        f"number of test+val heterodimers: {num_heterodimer}\n\n",
        f"number of test+val apos (af2): {num_apos_af2}\n",
        f"number of test+val preds (af2): {num_pred_af2}\n",
        f"number of test+val homodimers (af2): {num_homodimer_af2}\n",
        f"number of test+val heterodimers (af2): {num_heterodimer_af2}\n",
    )


def get_splits(
    pinder_dir: Path,
    config: ClusterConfig = ClusterConfig(),
    use_cache: bool = True,
) -> None:
    metadata_path = pinder_dir / "metadata.2.csv.gz"
    availability_index_path = pinder_dir / "index_with_apo.parquet"
    chk_dir = pinder_dir / "cluster" / get_config_hash(config)
    test_systems_path = chk_dir / "test_sys_table.csv"
    test_meta_output_path = chk_dir / "test_subset.csv"
    deleaked_checkpoint = chk_dir / "pindex_checkpoint.4.csv"
    if deleaked_checkpoint.is_file() and use_cache:
        log.info(f"Skipping get_splits, {deleaked_checkpoint} checkpoint exists...")
        return
    # NOTE: this needs to be the index after get_clusters.cluster checkpoint, not the original!
    index_path = chk_dir / "index.2.csv.gz"
    filtered_pindex_output_path = chk_dir / "pindex_checkpoint.3.csv"
    if test_meta_output_path.is_file() and use_cache:
        log.info(
            f"Skipping get_split_subsets, {test_meta_output_path} checkpoint exists..."
        )
        test_meta_df = read_csv_non_default_na(
            test_meta_output_path, dtype={"pdb_id": "str", "entry_id": "str"}
        )
    else:
        thresh_label = "{:.2f}".format(
            config.foldseek_af2_difficulty_threshold
        ).replace(".", "")
        af2_transitive_hits_path = (
            chk_dir / f"af2_lddt{thresh_label}_transitive_hits_mapping.csv"
        )
        test_meta_df = get_split_subsets(
            index_path,
            metadata_path,
            test_systems_path,
            availability_index_path,
            test_meta_output_path=test_meta_output_path,
            filtered_pindex_output_path=filtered_pindex_output_path,
            af2_transitive_hits_path=af2_transitive_hits_path,
            config=config,
        )
    print_test_meta_details(test_meta_df)
    ## Get splits
    filtered_index_path = chk_dir / "pindex_checkpoint.3.csv"
    test_subset_path = chk_dir / "test_subset.csv"
    deleak_map_file = chk_dir / "transitive_hits_mapping.csv"

    log.info("Finalizing test-val splits...")
    get_test_val_splits(
        original_index_path=pinder_dir / "index_with_apo.parquet",
        filtered_index_path=filtered_index_path,
        metadata_path=metadata_path,
        test_meta_path=test_subset_path,
        deleak_map_path=deleak_map_file,
        deleak_mask_outpath=deleaked_checkpoint,
        config=config,
    )


def rename_peptide_cluster_ids(
    pindex: pd.DataFrame,
    config: ClusterConfig = ClusterConfig(),
) -> pd.DataFrame:
    # rename chain cluster ids with using the "p" for all short chains
    pindex["cluster_id_R"] = pindex["cluster_id_R"].astype(str)
    pindex["cluster_id_L"] = pindex["cluster_id_L"].astype(str)
    pindex.loc[:, "cluster_id_L"] = [
        str(cluster_l) if l1 >= config.min_chain_length else "p"
        for cluster_l, l1 in zip(pindex["cluster_id_L"], pindex["length1"])
    ]
    pindex.loc[:, "cluster_id_R"] = [
        str(cluster_r) if l2 >= config.min_chain_length else "p"
        for cluster_r, l2 in zip(pindex["cluster_id_R"], pindex["length2"])
    ]
    # rename the cluster ids using the chains
    renamed_cluster_ids = []
    for cluster_id_L, cluster_id_R in pindex[["cluster_id_L", "cluster_id_R"]].values:
        id1, id2 = sorted([cluster_id_L, cluster_id_R])
        renamed_cluster_ids.append(f"cluster_{id1}_{id2}")
    pindex.loc[:, "cluster_id"] = renamed_cluster_ids
    return pindex


def get_split_subsets(
    index_path: str | Path,
    metadata_path: str | Path,
    test_systems_path: str | Path,
    availability_index_path: str | Path = "data/index_with_apo.parquet",
    test_meta_output_path: str | Path = "data/test_subset.csv",
    filtered_pindex_output_path: str | Path = "data/pindex_checkpoint.3.csv",
    af2_transitive_hits_path: str
    | Path = "data/af2_lddt070_transitive_hits_mapping.csv",
    config: ClusterConfig = ClusterConfig(),
) -> pd.DataFrame:
    """Get the split subsets. This function will generate the test set and the
    filtered pindex based on the test set. The test set will be saved to
    test_meta_output_path and the filtered pindex will be saved to
    filtered_pindex_output_path.

    Parameters:
        test_systems_path (Path): The path to the test systems.
        index_path (Path): The path to the index.
        metadata_path (Path): The path to the metadata.
        availability_index_path (Path): The path to the availability index.
        test_meta_output_path (Path): The path to save the test meta data.
        filtered_pindex_output_path (Path): The path to save the filtered pindex.
        max_depth_2_hits (int): The maximum depth 2 hits.
        max_depth_2_hits_with_comm (int): The maximum depth 2 hits with comm.
        min_depth_2_hits_with_comm (int): The minimum depth 2 hits with comm.
        top_n (int): The top n.

    """
    test_sys_table = read_csv_non_default_na(test_systems_path)

    eligible_systems = test_sys_table[
        (test_sys_table["depth_2_hits"] < config.max_depth_2_hits)
        & (test_sys_table["depth_2_hits_with_comm"] < config.max_depth_2_hits_with_comm)
    ]
    eligible_systems_to_hits = {
        k: v
        for k, v in zip(
            eligible_systems["id"], eligible_systems["depth_2_hits_with_comm"]
        )
    }

    metadata = read_csv_non_default_na(metadata_path, dtype={"entry_id": "str"})
    test_meta = pd.merge(
        eligible_systems[
            [
                "id",
                "cluster_id",
                "depth_2_hits_with_comm",
                "cluster_id_L",
                "cluster_id_R",
            ]
        ],
        metadata,
        how="inner",
        on="id",
    )
    availability_index = pd.read_parquet(availability_index_path)
    test_meta = pd.merge(
        test_meta,
        availability_index[
            [
                "id",
                "predicted_R",
                "predicted_L",
                "apo_R",
                "apo_L",
                "uniprot_R",
                "uniprot_L",
            ]
        ],
        how="left",
    )
    test_meta["min_both_putative_apo"] = (
        test_meta[["apo_R", "apo_L"]].min(axis=1).astype(int)
    )
    test_meta["min_both_pred"] = (
        test_meta[["predicted_R", "predicted_L"]].min(axis=1).astype(int)
    )
    # test_meta["min_both_putative_apo"] = test_meta["min_both_putative_apo"].fillna(0)
    # test_meta["min_both_pred"] = test_meta["min_both_pred"].fillna(0)
    test_meta["heterodimer"] = [
        (x != y) for x, y in zip(test_meta["uniprot_R"], test_meta["uniprot_L"])
    ]
    test_meta["release_date"] = test_meta["release_date"].map(
        lambda x: date.fromisoformat(x)
    )

    # Extract the af2mm IDs which pass transitive hits wrt other test/val members
    # start with most strict threshold to see max clean clusters

    af2_date = date.fromisoformat(config.alphafold_cutoff_date)
    date_dict = {
        sys_id: date.fromisoformat(date_str)
        for sys_id, date_str in zip(metadata["id"], metadata["release_date"])
        if date.fromisoformat(date_str) < af2_date
    }
    systems_before_af2mm = set(date_dict.keys())
    # Read af2mm transitive hits to derive a "clean" holdout of systems that are not part of AF2 training set
    # AND have no transitive hits to anything else that IS in the AF2 training set
    af2_transitive_hits = pd.read_csv(af2_transitive_hits_path)
    af2_transitive = {
        k: set(v.split(","))
        for k, v in zip(af2_transitive_hits["id"], af2_transitive_hits["neighbors"])
    }
    transitive_af2mm_all = {
        k: v.intersection(systems_before_af2mm)
        for k, v in tqdm(af2_transitive.items())
        if k not in systems_before_af2mm
    }
    clean_af2mm = {k for k, v in transitive_af2mm_all.items() if len(v) == 0}

    test_meta["af2_hard_pass"] = test_meta["id"].isin(clean_af2mm)
    test_meta["af2_date_pass"] = test_meta["release_date"] >= af2_date
    # Sort to prefer heterodimer and af2 deleaked "hard" cluster representative
    test_meta = test_meta.sort_values(
        [
            "heterodimer",
            "af2_hard_pass",
            "af2_date_pass",
            "min_both_putative_apo",
            "min_both_pred",
        ],
        ascending=False,
    )
    test_meta["split"] = "proto-test"
    top_n_ids = set(test_meta.groupby("cluster_id").head(config.top_n)["id"])
    test_meta["split"] = test_meta["id"].apply(
        lambda x: "proto-test" if x in top_n_ids else "proto-train"
    )
    test_meta["af2_mm_hard"] = (test_meta["af2_hard_pass"]) & (
        test_meta["split"] == "proto-test"
    )
    test_meta["af2_mm"] = (test_meta["af2_date_pass"]) & (
        test_meta["split"] == "proto-test"
    )
    af2mm_ids = set(test_meta[test_meta["af2_mm"]]["id"])
    test_meta.reset_index(drop=True, inplace=True)
    split_labels = []
    for pid in list(test_meta["id"]):
        if pid not in top_n_ids:
            split_labels.append("proto-train")
            continue
        if (eligible_systems_to_hits[pid] > config.min_depth_2_hits_with_comm) or (
            pid in af2mm_ids
        ):
            split_labels.append("proto-test")
        else:
            split_labels.append("proto-train")

    test_meta["split"] = split_labels
    filtered_pindex = read_csv_non_default_na(
        index_path, dtype={"pdb_id": "str", "entry_id": "str"}
    )
    filtered_pindex["split"] = "proto-train"
    filtered_pindex.loc[filtered_pindex["id"].isin(top_n_ids), "split"] = "proto-test"
    filtered_pindex.loc[
        filtered_pindex["id"].isin(set(test_meta[test_meta["af2_mm"]]["id"].unique())),
        "pinder_af2",
    ] = True
    test_meta.loc[test_meta["af2_mm"], "af2_difficulty"] = "Easy"
    test_meta.loc[~test_meta["af2_mm"], "af2_difficulty"] = "N/A"
    test_meta.loc[test_meta["af2_mm_hard"], "af2_difficulty"] = "Hard"
    filtered_pindex.loc[:, "pinder_af2_hard"] = (
        filtered_pindex["id"].isin(clean_af2mm) & filtered_pindex.pinder_af2
    )
    test_meta.to_csv(test_meta_output_path, index=False)
    filtered_pindex.to_csv(filtered_pindex_output_path, index=False)

    return test_meta


def get_test_val_splits(
    original_index_path: str | Path = "index_with_apo.parquet",
    filtered_index_path: str | Path = "pindex_checkpoint.3.csv",
    metadata_path: str | Path = "metadata.2.csv.gz",
    test_meta_path: str | Path = "test_subset.csv",
    deleak_map_path: str | Path = "transitive_hits_mapping.csv",
    deleak_mask_outpath: str | Path = "pindex_checkpoint.4.csv",
    config: ClusterConfig = ClusterConfig(),
) -> None:
    test_meta = read_csv_non_default_na(
        test_meta_path, dtype={"pdb_id": "str", "entry_id": "str"}
    )
    deleak_df = read_csv_non_default_na(deleak_map_path)
    deleak_map = {}
    not_used = set()
    for pid, neighbors in zip(deleak_df["id"], deleak_df["neighbors"]):
        if isinstance(neighbors, str):
            if CONSIDER_LEAKED in neighbors:
                not_used.add(pid)
                continue
            systems = set(neighbors.split(","))
        else:
            systems = set()
        deleak_map[pid] = systems

    filtered_pindex = read_csv_non_default_na(
        filtered_index_path, dtype={"pdb_id": "str", "entry_id": "str"}
    )
    system_to_cluster = {
        k: v for k, v in zip(filtered_pindex["id"], filtered_pindex["cluster_id"])
    }
    cluster_to_system: dict[str, set[str]] = defaultdict(set)
    for cluster_id, df in filtered_pindex.groupby("cluster_id"):
        cluster_to_system[cluster_id] = cluster_to_system[cluster_id].union(set(df.id))

    deleaked_systems = set()
    cumulative_test = 0
    for sys_id in test_meta[test_meta["split"] == "proto-test"]["id"]:
        sys_deleak = deleak_map.get(sys_id, set())
        sys_cluster = system_to_cluster[sys_id]
        try:
            deleaked_systems |= sys_deleak | cluster_to_system[sys_cluster]
            cumulative_test += len(sys_deleak | cluster_to_system[sys_cluster])
        except TypeError:
            log.error(f"{sys_id}, {sys_deleak}, {sys_cluster}")

    ranked_representatives = test_meta[test_meta["split"] == "proto-test"].sort_values(
        "depth_2_hits_with_comm", ascending=True
    )

    cumulative_deleaked = []
    for i, leaked in enumerate(ranked_representatives["depth_2_hits_with_comm"]):
        if i == 0:
            cumulative_deleaked.append(leaked)
        else:
            cumulative_deleaked.append(leaked + cumulative_deleaked[-1])
    ranked_representatives["cumulative_deleak_counts"] = np.array(cumulative_deleaked)

    metadata = read_csv_non_default_na(metadata_path, dtype={"entry_id": "str"})
    filtered_pindex = pd.merge(
        filtered_pindex,
        metadata[["id", "length1", "length2"]],
        how="inner",
        on="id",
    )
    filtered_pindex = rename_peptide_cluster_ids(filtered_pindex, config=config)

    short_ids = set(
        metadata[
            (metadata["length1"] < config.min_chain_length)
            | (metadata["length2"] < config.min_chain_length)
        ]["id"].unique()
    )

    used = filtered_pindex[
        filtered_pindex["id"].isin(set(ranked_representatives["id"].unique()))
    ]
    communities_used = set(used["cluster_id_L"].values).union(
        set(used["cluster_id_R"].values)
    )

    short_id_community_overlap = filtered_pindex[filtered_pindex["id"].isin(short_ids)]
    short_id_community_overlap = short_id_community_overlap[
        (
            short_id_community_overlap["cluster_id_L"].isin(communities_used)
            | short_id_community_overlap["cluster_id_R"].isin(communities_used)
        )
    ]

    short_id_community_overlap_ids = set(short_id_community_overlap["id"].unique())
    total_deleaked_systems = deleaked_systems | short_id_community_overlap_ids

    filtered_pindex["leakage_mask"] = filtered_pindex["id"].isin(total_deleaked_systems)

    test_val_ids = set(ranked_representatives["id"].unique())

    pindex = pd.read_parquet(original_index_path)

    afmm_ids = set(filtered_pindex[filtered_pindex["pinder_af2"]]["id"])
    pindex["split"] = "train"
    pindex.loc[pindex["id"].isin(test_val_ids), "split"] = "test"
    pindex["deleak_mask"] = False
    pindex.loc[
        pindex["id"].isin(total_deleaked_systems - test_val_ids), "deleak_mask"
    ] = True
    pindex["pinder_af2"] = False
    pindex.loc[pindex["id"].isin(afmm_ids), "pinder_af2"] = True
    pindex.loc[(pindex["split"] == "test") & (~pindex["pinder_af2"]), "split"] = "val"
    to_test = set(
        pindex[(~pindex["deleak_mask"]) & (pindex["split"] == "val")].sample(
            len(test_val_ids) // 2 - pindex["pinder_af2"].sum()
        )["id"]
    )
    pindex.loc[pindex["id"].isin(to_test), "split"] = "test"

    pindex.drop("pinder_af2_hard", axis=1, inplace=True)
    col_order = list(pindex.columns)
    pindex_final = pd.merge(
        pindex.drop(columns=["cluster_id"], errors="ignore"),
        filtered_pindex[["id", "cluster_id"]],
        how="inner",
        on="id",
    )
    pindex_final = pindex_final[col_order].copy()
    pindex_final.loc[pindex_final.deleak_mask, "split"] = "invalid"
    # Keep all systems, but don't include any deleak_mask = True in train/test/val.
    # pindex_final[~pindex_final["deleak_mask"]].to_csv(deleak_mask_outpath, index=False)
    pindex_final.to_csv(deleak_mask_outpath, index=False)


def get_train_noisy_apo(
    pinder_dir: Path,
    index: pd.DataFrame,
    config: ApoPairingConfig = ApoPairingConfig(),
) -> pd.DataFrame:
    noisy_config = ApoPairingConfig(
        max_interface_miss_frac=config.max_interface_miss_frac * 2.5,
        max_refine_rmsd=config.max_refine_rmsd * 2.5,
        min_seq_identity=0.05,
        min_aligned_apo_res_frac=config.min_aligned_apo_res_frac / 2.5,
    )
    metric_dir = pinder_dir / "apo_metrics"
    metric_pqt = metric_dir / "two_sided_apo_monomer_metrics.parquet"
    assert metric_pqt.is_file()
    two_sided = pd.read_parquet(metric_pqt)
    two_sided.loc[:, "pdb_id"] = [pid.split("__")[0] for pid in list(two_sided.id)]
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

    # Only patch those systems which do NOT yet have an apo pairing
    train_no_apo = index.query('split == "train" and (~apo_R or ~apo_L)').reset_index(
        drop=True
    )
    train_no_apo_R = train_no_apo.query("~apo_R").reset_index(drop=True)
    train_no_apo_L = train_no_apo.query("~apo_L").reset_index(drop=True)
    two_sided_R = (
        two_sided[two_sided["id"].isin(set(train_no_apo_R.id))]
        .query('unbound_body == "R"')
        .reset_index(drop=True)
    )
    two_sided_L = (
        two_sided[two_sided["id"].isin(set(train_no_apo_L.id))]
        .query('unbound_body == "L"')
        .reset_index(drop=True)
    )
    two_sided_eval = pd.concat([two_sided_R, two_sided_L], ignore_index=True)
    potential_apo = (
        two_sided_eval.query(
            f"Fmiss_R <= {noisy_config.max_interface_miss_frac} and Fmiss_L <= {noisy_config.max_interface_miss_frac}"
        )
        .query(f"sequence_identity >= {noisy_config.min_seq_identity}")
        .query(f"refine_rmsd < {noisy_config.max_refine_rmsd}")
        .query(f"frac_aligned >= {noisy_config.min_aligned_apo_res_frac}")
        # .query('~invalid_coverage') keep this in there
        .sort_values(["id", "max_miss"], ascending=False)
        .reset_index(drop=True)
    )

    potential_apo = add_weighted_apo_score(potential_apo, config=noisy_config)
    potential_apo.loc[:, "apo_pdb"] = potential_apo.apo_monomer_id + ".pdb"
    potential_apo.to_parquet(
        metric_dir / "scored_noisy_train_apo_pairings.parquet", index=False
    )
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
    col_order = list(index.columns)

    index.loc[index.apo_R_pdb.isna(), "apo_R_pdb"] = ""
    index.loc[index.apo_R_pdbs.isna(), "apo_R_pdbs"] = ""
    index.loc[index.apo_L_pdb.isna(), "apo_L_pdb"] = ""
    index.loc[index.apo_L_pdbs.isna(), "apo_L_pdbs"] = ""

    index_rest = index[~index["id"].isin(set(train_no_apo.id))].reset_index(drop=True)
    index_patch = index[index["id"].isin(set(train_no_apo.id))].reset_index(drop=True)

    for side, df in condensed_pairings.groupby("unbound_body"):
        df = df.drop("unbound_body", axis=1).reset_index(drop=True)
        canon_pdb_col = f"apo_{side}_pdb"
        pdbs_col = f"apo_{side}_pdbs"
        index_patch_side = index_patch[~index_patch[f"apo_{side}"]].reset_index(
            drop=True
        )
        index_rest_side = index_patch[index_patch[f"apo_{side}"]].reset_index(drop=True)
        index_patch_side.drop(
            [canon_pdb_col, pdbs_col], axis=1, inplace=True, errors="ignore"
        )
        rename_cols = {
            "canonical_pdb": canon_pdb_col,
            "apo_pdb": pdbs_col,
        }
        df.rename(rename_cols, axis=1, inplace=True)
        index_patch_side = pd.merge(index_patch_side, df, how="left")
        index_patch_side.loc[index_patch_side[pdbs_col].isna(), pdbs_col] = ""
        index_patch_side.loc[index_patch_side[canon_pdb_col].isna(), canon_pdb_col] = ""
        index_patch_side.loc[:, f"apo_{side}"] = index_patch_side[canon_pdb_col] != ""
        index_patch = pd.concat(
            [index_patch_side[list(index_patch.columns)], index_rest_side],
            ignore_index=True,
        )
    index_patched = pd.concat(
        [index_rest, index_patch[list(index_rest.columns)]], ignore_index=True
    )
    index_patched = index_patched[col_order].copy()
    # Add apo_R/L_quality column to distinguish which class the apo belongs to: validated vs. noisy pairing
    R_patched_ids = set(index.query("~apo_R").id).intersection(
        set(index_patched.query("apo_R").id)
    )
    L_patched_ids = set(index.query("~apo_L").id).intersection(
        set(index_patched.query("apo_L").id)
    )

    index_patched.reset_index(drop=True, inplace=True)
    # Anything that was "patched" is low quality
    index_patched.loc[index_patched.id.isin(R_patched_ids), "apo_R_quality"] = "low"
    index_patched.loc[index_patched.id.isin(L_patched_ids), "apo_L_quality"] = "low"
    # Everything else is high quality
    index_patched.loc[index_patched.apo_R_quality.isna(), "apo_R_quality"] = "high"
    index_patched.loc[index_patched.apo_L_quality.isna(), "apo_L_quality"] = "high"
    # Everything without apo is "" quality
    index_patched.loc[~index_patched.apo_R, "apo_R_quality"] = ""
    index_patched.loc[~index_patched.apo_L, "apo_L_quality"] = ""
    return index_patched


def add_neff_to_index(
    index: pd.DataFrame,
) -> pd.DataFrame:
    monomer_neff = get_supplementary_data("monomer_neff")
    col_order = list(index.columns)
    index.drop(["chain1_neff", "chain2_neff"], axis=1, inplace=True)
    index = pd.merge(
        index, monomer_neff[["id", "chain1_neff", "chain2_neff"]], how="left"
    )
    # There should be no NaN, but to be safe
    index["chain1_neff"] = index["chain1_neff"].fillna(0.0)
    index["chain2_neff"] = index["chain2_neff"].fillna(0.0)
    index = index[col_order].copy()
    return index


def construct_final_index(
    pinder_dir: Path,
    apo_config: ApoPairingConfig = ApoPairingConfig(),
    use_cache: bool = True,
    blacklist_invalid_ids: dict[str, str] = TEST_SYSTEM_BLACKLIST,
) -> None:
    index_checkpoint_file = pinder_dir / "index.parquet"
    meta_checkpoint_file = pinder_dir / "metadata.parquet"

    if use_cache and index_checkpoint_file.is_file() and meta_checkpoint_file.is_file():
        log.info(f"Index and metadata checkpoints exist. Skipping...")
    ialign_dir = pinder_dir / "ialign_metrics"
    chk5_path = ialign_dir / "pindex_checkpoint.5.parquet"
    df5 = pd.read_parquet(chk5_path)
    metadata_path = pinder_dir / "metadata.2.csv.gz"
    metadata = read_csv_non_default_na(metadata_path, dtype={"entry_id": "str"})
    # Add "noisy" potentially invalid apo pairings to train split that got filtered out
    df5 = get_train_noisy_apo(pinder_dir, df5, apo_config)

    index_empty_str_cols = [
        "apo_R_pdb",
        "apo_L_pdb",
        "apo_R_pdbs",
        "apo_L_pdbs",
        "predicted_R_pdb",
        "predicted_L_pdb",
        "apo_R_quality",
        "apo_L_quality",
    ]
    for c in index_empty_str_cols:
        df5[c] = df5[c].fillna("")

    index_schema = IndexEntry.__annotations__
    index_fields = index_schema.keys()
    df5 = df5[index_fields].copy()
    # Add per-chain Neff values calculated using pre-computed MSAs
    df5 = add_neff_to_index(df5)
    # Downcast float and int columns and cast str object to category
    final_index = downcast_dtypes(df5)
    # Create union of uniprot_R/L and cluster_id_R/L categorical columns so they can be compared
    union_cols = [("uniprot_R", "uniprot_L"), ("cluster_id_R", "cluster_id_L")]
    for R_col, L_col in union_cols:
        cat_union = union_categoricals(
            [final_index[R_col], final_index[L_col]]
        ).categories
        final_index[R_col] = final_index[R_col].cat.set_categories(cat_union)
        final_index[L_col] = final_index[L_col].cat.set_categories(cat_union)

    for row in tqdm(final_index.to_dict(orient="records")):
        entry = IndexEntry(**row)
        assert isinstance(entry, IndexEntry)

    final_index.loc[final_index.id.isin(blacklist_invalid_ids.keys()), "split"] = (
        "invalid"
    )
    final_index.loc[final_index.id.isin(blacklist_invalid_ids.keys()), "pinder_xl"] = (
        False
    )
    # Sort split categorical column by custom order, with invalid split members last.
    split_order = ["test", "val", "train", "invalid"]
    final_index["split"] = final_index["split"].cat.reorder_categories(split_order)
    final_index = final_index.sort_values("split").reset_index(drop=True)
    final_index.to_parquet(index_checkpoint_file, index=False, engine="pyarrow")

    meta_empty_str_cols = [
        "label",
        "biol_details",
        "complex_type",
        "ECOD_ids_R",
        "ECOD_ids_L",
        "ECOD_names_R",
        "ECOD_names_L",
        "ECOD_intersection_R",
        "ECOD_intersection_L",
    ]
    for c in meta_empty_str_cols:
        metadata[c] = metadata[c].fillna("")

    meta_schema = MetadataEntry.__annotations__
    # Downcast float and int columns and cast str object to category
    metadata = downcast_dtypes(metadata)
    supplemental_cols = [
        "ECOD_ids_R",
        "ECOD_ids_L",
        "ECOD_intersection_R",
        "ECOD_intersection_L",
        "chain_1_residues",
        "chain_2_residues",
    ]
    supplemental_cols = [c for c in supplemental_cols if c in metadata.columns]
    supp_meta = metadata[["id"] + supplemental_cols].copy()
    supp_meta.to_parquet(
        pinder_dir / "supplementary_metadata.parquet", index=False, engine="pyarrow"
    )
    metadata.drop(columns=supplemental_cols, inplace=True)
    meta_fields = meta_schema.keys()
    final_metadata = metadata[meta_fields].copy()
    # Sort metadata to match order of IDs in index
    final_metadata = final_metadata.set_index("id")
    final_metadata = final_metadata.reindex(index=final_index["id"])
    final_metadata = final_metadata.reset_index()
    for row in tqdm(final_metadata.to_dict(orient="records")):
        entry = MetadataEntry(**row)
        assert isinstance(entry, MetadataEntry)

    final_metadata.to_parquet(meta_checkpoint_file, index=False, engine="pyarrow")
