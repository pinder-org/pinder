from __future__ import annotations

import logging
import pickle
from datetime import date
from pathlib import Path
from string import digits

import numpy as np
import pandas as pd
from tqdm import tqdm

from pinder.core import get_pinder_location
from pinder.core.utils import setup_logger
from pinder.data.alignment_utils import Interface
from pinder.data.config import (
    ContactConfig,
    ClusterConfig,
    IalignConfig,
    get_config_hash,
)
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.qc import ialign


log = setup_logger(__name__, log_level=logging.WARNING)

af2_date = date.fromisoformat(ClusterConfig().alphafold_cutoff_date)


def add_alignment_cols_to_index(index: pd.DataFrame) -> pd.DataFrame:
    index["pdb_id"] = index.pdb_id.astype("object")
    index["chain_R"] = index.chain_R.astype("object")
    index["chain_L"] = index.chain_L.astype("object")
    index.loc[:, "asymR"] = index.chain_R.str.rstrip(digits)
    index.loc[:, "asymL"] = index.chain_L.str.rstrip(digits)
    index.loc[:, "pdb1"] = index.pdb_id + "_" + index.asymR
    index.loc[:, "pdb2"] = index.pdb_id + "_" + index.asymL
    index.loc[index.asymR == index.asymL, "asm_count"] = 1
    index.loc[index.asymR != index.asymL, "asm_count"] = 2
    index["asm_count"] = index.asm_count.astype(int)
    return index


def get_hit_interfaces(
    pinder_dir: Path,
    hits: pd.DataFrame,
    config: ContactConfig = ContactConfig(),
) -> dict[tuple[str, str], Interface]:
    interface_pkl = (
        pinder_dir.parent
        / "foldseek_contacts"
        / get_config_hash(config)
        / "interfaces.pkl"
    )
    with open(interface_pkl, "rb") as f:
        interfaces = pickle.load(f)
    qt_ids = set(hits["query"]).union(set(hits["target"]))
    hit_interfaces: dict[tuple[str, str], Interface] = {}
    for k, v in tqdm(interfaces.items()):
        k1, k2 = k
        pdb_id = k1.split("__")[0]
        ch1 = k1.split("__")[1].split("_")[0].rstrip(digits)
        ch2 = k2.split("__")[1].split("_")[0].rstrip(digits)
        c1 = pdb_id + "_" + ch1
        c2 = pdb_id + "_" + ch2
        if c1 in qt_ids or c2 in qt_ids:
            hit_interfaces[k] = v
    return hit_interfaces


def reformat_hits(hits: pd.DataFrame, ref_chains: set[str]) -> pd.DataFrame:
    if ".pdb" in hits.target.values[0]:
        hits["query"] = hits["query"].str.replace(".pdb", "", regex=False)
        hits["target"] = hits["target"].str.replace(".pdb", "", regex=False)
    # Assign true if 'query' chain is part of af2/ref monomer chains
    hits.loc[:, "query_ref_chain"] = hits["query"].isin(ref_chains)
    # Assign true if 'target' chain is part of af2/ref monomer chains
    hits.loc[:, "target_ref_chain"] = hits["target"].isin(ref_chains)
    hits.loc[:, "qt"] = [
        ":".join(sorted([q, t])) for q, t in zip(hits["query"], hits["target"])
    ]
    hits = (
        hits.sort_values("alignment_score", ascending=False)
        .drop_duplicates("qt", keep="first")
        .reset_index(drop=True)
    )
    hits_reordered = []
    args = zip(
        hits["query"],
        hits["target"],
        hits["query_ref_chain"],
        hits["alignment_score"],
        hits["qstart"],
        hits["qend"],
        hits["tstart"],
        hits["tend"],
        hits["alnlen"],
        hits["qlen"],
        hits["tlen"],
    )
    for q, t, q_ref_chain, sc, qstart, qend, tstart, tend, alnlen, qlen, tlen in tqdm(
        args
    ):
        query_pdb, query_ch = q.split("_")
        target_pdb, target_ch = t.split("_")
        hits_reordered.append(
            {
                "ref_pdb_id": query_pdb if q_ref_chain else target_pdb,
                "hit_pdb_id": target_pdb if q_ref_chain else query_pdb,
                "ref_chain": query_ch if q_ref_chain else target_ch,
                "hit_chain": target_ch if q_ref_chain else query_ch,
                "alignment_score": sc,
                "ref_start": qstart if q_ref_chain else tstart,
                "ref_end": qend if q_ref_chain else tend,
                "hit_start": tstart if q_ref_chain else qstart,
                "hit_end": tend if q_ref_chain else qend,
                "alnlen": alnlen,
                "ref_len": qlen if q_ref_chain else tlen,
                "hit_len": tlen if q_ref_chain else qlen,
            }
        )
    hits = pd.DataFrame(hits_reordered)
    return hits


def get_ref_alignment_hits(
    index: pd.DataFrame,
    pinder_dir: Path,
    alignment_type: str,
    pinder_subset: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    output_pqt = pinder_dir / f"{pinder_subset}-{alignment_type}-hits.parquet"
    if use_cache and output_pqt.is_file():
        log.info(f"Skipping alignment filtering, {output_pqt} exists...")
        hits = pd.read_parquet(output_pqt)
        return hits
    if alignment_type == "foldseek":
        aln_dir = pinder_dir / "foldseek/foldseek_dbs"
    elif alignment_type == "mmseqs":
        aln_dir = pinder_dir / "mmseqs2/mmseqs_dbs"
    aln_pqt = aln_dir / "alignment.parquet"
    log.info(f"Reading {aln_pqt}...")
    df = pd.read_parquet(aln_pqt)
    if ".pdb" in df.target.values[0]:
        df["query"] = df["query"].str.replace(".pdb", "", regex=False)
        df["target"] = df["target"].str.replace(".pdb", "", regex=False)
    index_subset = index.query(pinder_subset).reset_index(drop=True)
    ref_ids = set(index_subset.pdb1).union(set(index_subset.pdb2))
    hits = df[df["query"].isin(ref_ids) | df["target"].isin(ref_ids)].reset_index(
        drop=True
    )
    hits.to_parquet(output_pqt, index=False)
    return hits


def get_subset_interfaces(
    index: pd.DataFrame,
    hits: pd.DataFrame,
    interfaces: dict[tuple[str, str], Interface],
    pinder_subset: str,
) -> pd.DataFrame:
    subset_interfaces = []
    for pid in set(index.query(pinder_subset).id):
        R, L = pid.split("--")
        interface_key = (R + "-R", L + "-L")
        try:
            inter = interfaces[interface_key]
        except KeyError as e:
            print(
                pid,
                " has no interfaces, number of hits: ",
                hits.query(f'id == "{pid}"').shape[0],
            )
            continue
        R_int = inter.indices1
        L_int = inter.indices2
        pdb_id = R.split("__")[0]
        R_ch = R.split("_")[-2].rstrip(digits)
        L_ch = L.split("_")[-2].rstrip(digits)
        subset_interfaces.append(
            {
                "id": pid,
                "interface_key": interface_key,
                "ref_pdb_id": pdb_id,
                "ref_chain_R": R_ch,
                "ref_chain_L": L_ch,
                "ref_R_interface": R_int,
                "ref_L_interface": L_int,
            }
        )
    subset_interfaces = pd.DataFrame(subset_interfaces)
    return subset_interfaces


def get_subset_hits(
    index: pd.DataFrame,
    metadata: pd.DataFrame,
    pinder_dir: Path,
    alignment_type: str,
    pinder_subset: str,
) -> pd.DataFrame:
    hits = get_ref_alignment_hits(index, pinder_dir, alignment_type, pinder_subset)
    if alignment_type == "mmseqs":
        score_col = "pident"
        # Make it on scale of 0:1
        hits["pident"] = hits["pident"] / 100
    else:
        score_col = "lddt"
    hits.rename({score_col: "alignment_score"}, axis=1, inplace=True)

    index_subset = index.query(pinder_subset).reset_index(drop=True)
    subset_ids = set(index_subset.pdb1).union(set(index_subset.pdb2))

    interfaces = get_hit_interfaces(pinder_dir, hits=hits)
    hits = reformat_hits(hits, subset_ids)
    hits = pd.merge(
        hits,
        metadata[["entry_id", "release_date"]]
        .rename({"entry_id": "hit_pdb_id", "release_date": "date"}, axis=1)
        .drop_duplicates()
        .reset_index(drop=True),
        how="left",
    )
    hits["date"] = hits["date"].astype("object").apply(lambda x: date.fromisoformat(x))

    hits = pd.merge(
        hits,
        metadata[["id", "entry_id"]]
        .rename({"entry_id": "ref_pdb_id"}, axis=1)
        .drop_duplicates()
        .reset_index(drop=True),
        how="left",
    )
    ref_pdb_ids = set(index_subset.pdb_id)
    hits.loc[:, "subset_member"] = hits.ref_pdb_id.isin(ref_pdb_ids)
    hits.loc[:, "pdb_id_pair"] = hits.ref_pdb_id + ":" + hits.hit_pdb_id
    hits = hits.query("subset_member").reset_index(drop=True)
    # Add interface info of reference members in alignment hits
    subset_interfaces = get_subset_interfaces(index, hits, interfaces, pinder_subset)

    hit_interfaces = pd.merge(
        hits,
        subset_interfaces[
            ["id", "ref_pdb_id", "ref_chain_R", "ref_R_interface"]
        ].rename(
            {"ref_chain_R": "ref_chain", "ref_R_interface": "ref_interface"}, axis=1
        ),
        how="left",
    )
    R_hits = hit_interfaces[~hit_interfaces.ref_interface.isna()].reset_index(drop=True)
    L_hits = hit_interfaces[hit_interfaces.ref_interface.isna()].reset_index(drop=True)
    L_hits.drop("ref_interface", axis=1, inplace=True)
    L_hits = pd.merge(
        L_hits,
        subset_interfaces[
            ["id", "ref_pdb_id", "ref_chain_L", "ref_L_interface"]
        ].rename(
            {"ref_chain_L": "ref_chain", "ref_L_interface": "ref_interface"}, axis=1
        ),
        how="left",
    )
    L_hits = L_hits[~L_hits.ref_interface.isna()].reset_index(drop=True)

    hit_interfaces = pd.concat([R_hits, L_hits], ignore_index=True)
    hit_interfaces.loc[:, "interface_size"] = [
        len(i) for i in list(hit_interfaces.ref_interface)
    ]
    hit_interfaces.loc[:, "interface_intersect"] = [
        len(i.intersection(set(range(s, e + 1))))
        for i, s, e in zip(
            list(hit_interfaces.ref_interface),
            hit_interfaces.ref_start,
            hit_interfaces.ref_end,
        )
    ]
    hit_interfaces.loc[:, "ref_pdb_ch"] = (
        hit_interfaces.ref_pdb_id + "_" + hit_interfaces.ref_chain
    )
    hit_interfaces.loc[:, "hit_pdb_ch"] = (
        hit_interfaces.hit_pdb_id + "_" + hit_interfaces.hit_chain
    )
    hit_interfaces = hit_interfaces.query("ref_pdb_id != hit_pdb_id").reset_index(
        drop=True
    )
    hit_interfaces.loc[:, "frac_intersect"] = (
        hit_interfaces.interface_intersect / hit_interfaces.interface_size
    )
    return hit_interfaces


def get_paired_hits(
    index: pd.DataFrame,
    metadata: pd.DataFrame,
    pinder_dir: Path,
    pinder_subset: str,
) -> pd.DataFrame:
    hit_list = []
    for alignment_type in ["foldseek", "mmseqs"]:
        alignment_hits = get_subset_hits(
            index=index,
            metadata=metadata,
            pinder_dir=pinder_dir,
            alignment_type=alignment_type,
            pinder_subset=pinder_subset,
        )
        alignment_hits.loc[:, "alignment_type"] = alignment_type
        hit_list.append(alignment_hits)
    all_hits = pd.concat(hit_list, ignore_index=True)

    R_hits = pd.merge(
        all_hits,
        index[["id", "pdb_id", "pdb1", "cluster_id", "split"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(
            {
                "pdb_id": "hit_pdb_id",
                "id": "hit_id",
                "cluster_id": "hit_cluster",
                "split": "hit_split",
                "pdb1": "hit_pdb_ch",
            },
            axis=1,
        )
        .drop_duplicates()
        .reset_index(drop=True),
        how="left",
    )
    L_hits = R_hits[R_hits.hit_id.isna()].reset_index(drop=True)
    R_hits = R_hits[~R_hits.hit_id.isna()].reset_index(drop=True)
    L_hits.drop(["hit_id", "hit_cluster", "hit_split"], axis=1, inplace=True)
    L_hits = pd.merge(
        L_hits,
        index[["id", "pdb_id", "pdb2", "cluster_id", "split"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(
            {
                "pdb_id": "hit_pdb_id",
                "id": "hit_id",
                "cluster_id": "hit_cluster",
                "split": "hit_split",
                "pdb2": "hit_pdb_ch",
            },
            axis=1,
        )
        .drop_duplicates()
        .reset_index(drop=True),
        how="left",
    )
    L_hits = L_hits[~L_hits.hit_id.isna()].reset_index(drop=True)

    all_hits = pd.concat([R_hits, L_hits], ignore_index=True)
    all_hits = pd.merge(all_hits, index[["id", "cluster_id"]], how="left")
    all_hits = all_hits.sort_values("alignment_score", ascending=False).drop_duplicates(
        [
            "id",
            "ref_pdb_id",
            "hit_pdb_id",
            "ref_chain",
            "hit_chain",
            "hit_id",
            "alignment_type",
        ],
        keep="first",
    )
    paired_leaks = (
        all_hits.drop("ref_interface", axis=1)
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )
    return paired_leaks


def find_potential_leaks(
    index: pd.DataFrame,
    metadata: pd.DataFrame,
    pinder_dir: Path = get_pinder_location(),
    ialign_batch_size: int = 20_000,
    max_workers: int | None = None,
    use_cache: bool = True,
    config: IalignConfig = IalignConfig(),
) -> pd.DataFrame:
    cache_dir = pinder_dir / "ialign_metrics"
    cache_dir.mkdir(exist_ok=True, parents=True)
    ialign_checkpoint = cache_dir / "metrics.parquet"
    alignment_hits_checkpoint = cache_dir / "potential_alignment_leaks.parquet"
    potential_leaks_checkpoint = cache_dir / "ialign_potential_leaks.parquet"
    if use_cache and potential_leaks_checkpoint.is_file():
        potential_leaks = pd.read_parquet(potential_leaks_checkpoint)
        return potential_leaks

    if use_cache and alignment_hits_checkpoint.is_file():
        potential_leaks = pd.read_parquet(alignment_hits_checkpoint)
    else:
        paired_hits = get_paired_hits(
            index, metadata, pinder_dir, pinder_subset="pinder_xl"
        )
        paired_hits = (
            paired_hits.sort_values("alignment_score", ascending=False)
            .drop_duplicates(
                [
                    "ref_pdb_id",
                    "hit_pdb_id",
                    "ref_chain",
                    "hit_chain",
                    "id",
                    "hit_id",
                    "alignment_type",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )
        subset_pairs = []
        for pinder_subset in ["pinder_xl", "pinder_af2"]:
            pairs = paired_hits[
                paired_hits["id"].isin(set(index.query(pinder_subset).id))
            ].copy()
            if "af2" in pinder_subset:
                # Only look at hits to pre-AF2 training cutoff date
                pairs = pairs[pairs.date <= af2_date].reset_index(drop=True)
            else:
                # We don't consider hits to the same cluster or to invalid/test split as leaks for PINDER
                pairs = pairs.query(
                    'hit_split != "invalid" and hit_split != "test" and cluster_id != hit_cluster'
                ).reset_index(drop=True)
            pairs.loc[:, "pinder_subset"] = pinder_subset
            subset_pairs.append(pairs)
        potential_leaks = pd.concat(subset_pairs, ignore_index=True)
        # These pairs now go to iAlign calculation
        potential_leaks.to_parquet(alignment_hits_checkpoint, index=False)
    ialign_pairs = (
        potential_leaks[["id", "hit_id"]].drop_duplicates().reset_index(drop=True)
    )
    if use_cache and ialign_checkpoint.is_file():
        ialign_scores = pd.read_parquet(ialign_checkpoint)
    else:
        _ = ialign.process_in_batches(
            ialign_pairs,
            batch_size=ialign_batch_size,
            cache_dir=cache_dir,
            n_jobs=max_workers,
            config=config,
        )
        ialign_scores = pd.concat(
            [pd.read_parquet(p) for p in cache_dir.glob("batch_*.parquet")],
            ignore_index=True,
        )
        ialign_scores.loc[:, "id_pair"] = (
            ialign_scores["query_id"] + ":" + ialign_scores["hit_id"]
        )
        ialign_scores.to_parquet(ialign_checkpoint, index=False)

    potential_leaks = pd.merge(
        potential_leaks, ialign_scores.rename({"query_id": "id"}, axis=1), how="left"
    )
    potential_leaks.loc[:, "log_pvalue"] = potential_leaks["P-value"].apply(np.log10)
    potential_leaks["log_pvalue"] = potential_leaks["log_pvalue"].fillna(0.0)
    potential_leaks.to_parquet(potential_leaks_checkpoint, index=False)
    return potential_leaks


def get_alignment_similarity(
    pinder_dir: Path,
    cluster_config: ClusterConfig = ClusterConfig(),
    ialign_config: IalignConfig = IalignConfig(),
    use_cache: bool = True,
) -> None:
    cache_dir = pinder_dir / "ialign_metrics"
    cache_dir.mkdir(exist_ok=True, parents=True)
    af2_checkpoint_file = cache_dir / "pindex_checkpoint.5.parquet"
    if use_cache and af2_checkpoint_file.is_file():
        log.info(f"{af2_checkpoint_file} checkpoint exists. Skipping...")
        return
    chk_dir = pinder_dir / "cluster" / get_config_hash(cluster_config)
    metadata_path = pinder_dir / "metadata.2.csv.gz"
    chk3_path = chk_dir / "pindex_checkpoint.3.csv"
    chk4_path = chk_dir / "pindex_checkpoint.4.csv"
    df4 = read_csv_non_default_na(chk4_path, dtype={"pdb_id": "str"})
    df3 = read_csv_non_default_na(chk3_path, dtype={"pdb_id": "str"})

    df4.loc[df4.cluster_id.str.contains("-1"), "split"] = "invalid"
    df3.cluster_id_R = df3.cluster_id_R.astype(str)
    df3.cluster_id_L = df3.cluster_id_L.astype(str)

    metadata = read_csv_non_default_na(metadata_path, dtype={"entry_id": "str"})
    df3 = pd.merge(df3, metadata[["id", "length1", "length2"]], how="left")
    df3.loc[:, "cluster_id_L"] = [
        f"cluster_{cluster_l}" if l1 >= cluster_config.min_chain_length else "cluster_p"
        for cluster_l, l1 in zip(df3["cluster_id_L"], df3["length1"])
    ]
    df3.loc[:, "cluster_id_R"] = [
        f"cluster_{cluster_r}" if l2 >= cluster_config.min_chain_length else "cluster_p"
        for cluster_r, l2 in zip(df3["cluster_id_R"], df3["length2"])
    ]
    df4 = pd.merge(
        df4.drop(columns=["cluster_id_R", "cluster_id_L"], errors="ignore"),
        df3[["id", "cluster_id_R", "cluster_id_L"]],
        how="left",
    )
    df4 = pd.merge(df4, metadata[["id", "release_date"]], how="left")
    df4.loc[df4.split == "test", "pinder_xl"] = True
    df4 = add_alignment_cols_to_index(df4)
    potential_leaks = find_potential_leaks(
        df4, metadata, pinder_dir, config=ialign_config
    )
    potential_leaks.loc[:, "date_leak"] = [
        d <= af2_date for d in list(potential_leaks.date)
    ]
    af_leaks = potential_leaks.query('pinder_subset == "pinder_af2"').reset_index(
        drop=True
    )
    xl_leaks = potential_leaks.query('pinder_subset == "pinder_xl"').reset_index(
        drop=True
    )
    query_list = [
        f"log_pvalue < {ialign_config.log_pvalue_threshold}",
        f"RMSD < {ialign_config.rmsd_threshold}",
        f"`IS-score` > {ialign_config.is_score_threshold}",
    ]
    query_str = " and ".join(query_list)
    af_leak_ids = set(af_leaks.query("date_leak").query(query_str).id)
    xl_train_leak_ids = set(
        xl_leaks.query(query_str)
        .query("hit_split == 'train' and hit_cluster != cluster_id")
        .id
    )
    xl_val_leak_ids = set(
        xl_leaks.query(query_str)
        .query("hit_split == 'val' and hit_cluster != cluster_id")
        .id
    )
    df4.loc[:, "pinder_af2_timesplit"] = df4.pinder_af2
    df4.loc[:, "ialign_af2_similar"] = (df4.id.isin(af_leak_ids)) & (
        df4.split == "test"
    )
    df4.loc[:, "ialign_train_similar"] = df4.id.isin(xl_train_leak_ids)
    df4.loc[:, "ialign_val_similar"] = df4.id.isin(xl_val_leak_ids)
    df4.loc[(df4.id.isin(af_leak_ids)) & (df4.split == "test"), "pinder_af2"] = False
    df4.to_parquet(af2_checkpoint_file, index=False)
