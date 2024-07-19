from __future__ import annotations
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
from plotly.graph_objs._figure import Figure
from tqdm import tqdm

from pinder.core import get_index, get_metadata, get_pinder_location
from pinder.core.index.utils import get_supplementary_data, get_pinder_bucket_root
from pinder.core.utils.cloud import gcs_read_dataframe
from pinder.core.utils.log import setup_logger
from pinder.data.config import ClusterConfig, IalignConfig
from pinder.data.plot import (
    Theme,
    DarkTheme,
    LightTheme,
    Colors,
    LinePlot,
    ViolinPlot,
    constants as pc,
    figure_utils as futils,
)
from pinder.data.plot.datasets import get_dataset_indices
from pinder.eval.dockq.method import (
    add_pinder_set,
    CapriClass,
)


log = setup_logger(__name__)
pindex = get_index()
meta = get_metadata()


def patch_af2mm_monomer_types(metrics: pd.DataFrame) -> pd.DataFrame:
    af2mm = metrics.query("method_name == 'af2mm'").reset_index(drop=True)
    rest = metrics.query("method_name != 'af2mm'").reset_index(drop=True)
    af2mm_holo = af2mm.copy()
    af2mm_apo = af2mm.copy()
    af2mm_pred = af2mm.copy()
    af2mm_holo.loc[:, "monomer_name"] = "holo"
    af2mm_apo.loc[:, "monomer_name"] = "apo"
    af2mm_pred.loc[:, "monomer_name"] = "predicted"

    pred_ids = set(pindex.query("pinder_af2 and predicted_R and predicted_L").id)
    apo_ids = set(pindex.query("pinder_af2 and apo_R and apo_L").id)
    af2mm_apo = af2mm_apo[af2mm_apo["id"].isin(apo_ids)].reset_index(drop=True)
    af2mm_pred = af2mm_pred[af2mm_pred["id"].isin(pred_ids)].reset_index(drop=True)
    af2mm = pd.concat([af2mm_holo, af2mm_apo, af2mm_pred], ignore_index=True)
    metrics = pd.concat([af2mm, rest], ignore_index=True)
    return metrics


def get_dockq_metrics() -> pd.DataFrame:
    data_root = get_pinder_bucket_root() + "/figure_data/"
    dockq = gcs_read_dataframe(data_root + "dockq.parquet")
    return dockq


def get_ialign_metrics() -> pd.DataFrame:
    data_root = get_pinder_bucket_root() + "/figure_data/"
    ialign = gcs_read_dataframe(data_root + "ialign.parquet")
    ialign.loc[:, "log_pvalue"] = ialign["P-value"].apply(np.log10)
    ialign["log_pvalue"] = ialign["log_pvalue"].fillna(0.0)
    return ialign


def get_penalized_dockq() -> pd.DataFrame:
    cc = CapriClass()
    metrics = get_dockq_metrics()
    metrics.loc[:, "CAPRI_rank"] = metrics.CAPRI.apply(lambda x: cc[x])
    metrics.loc[:, "decoy"] = metrics.model_name + ".pdb"
    # BiotiteDockQ columns that are superceded by MethodMetrics
    metrics.drop(["system", "method"], axis=1, inplace=True, errors="ignore")
    log.info("Adding pinder subset...")
    metrics_penalty = add_pinder_set(metrics, allow_missing=False, custom_index="")
    metrics_penalty = remove_af2mm_nonaf2(metrics_penalty)
    dockq = metrics_penalty.copy()
    dockq.loc[:, "uuid"] = [
        "___".join([r[c] for c in ["method_name", "id", "monomer_name", "model_name"]])
        for i, r in dockq.iterrows()
    ]
    dockq_cols = [
        "uuid",
        "method_name",
        "id",
        "monomer_name",
        "pinder_set",
        "model_name",
        "iRMS",
        "LRMS",
        "Fnat",
        "DockQ",
        "CAPRI",
        "rank",
        "CAPRI_rank",
    ]
    dockq = dockq[dockq_cols].drop_duplicates().reset_index(drop=True)
    assert (
        len(set(dockq.uuid))
        == dockq.drop("pinder_set", axis=1).drop_duplicates().shape[0]
    )
    dockq.loc[:, "pinder_set"] = [pc.LABELS[ps] for ps in list(dockq.pinder_set)]
    dockq = dockq.drop_duplicates(["uuid", "pinder_set"], keep="first").reset_index(
        drop=True
    )
    return dockq


def select_top_n(
    data: pd.DataFrame,
    sort_col: str,
    ascending: bool = True,
    n: int = 1,
    group_cols: list[str] = ["pinder_set", "method_name", "id", "monomer_name"],
) -> pd.DataFrame:
    top_n = (
        data.sort_values(sort_col, ascending=ascending)
        .groupby(group_cols, as_index=False)
        .head(n)
        .reset_index(drop=True)
        .sort_values("DockQ", ascending=False)
        .drop_duplicates(group_cols, keep="first")
        .reset_index(drop=True)
    )
    return top_n


def remove_af2mm_nonaf2(metrics: pd.DataFrame) -> pd.DataFrame:
    af2mm_methods = [
        "af2mm",
        "af2mm_wt",
        "af2mm_truncated",
        "af2mm_full-length",
    ]
    af2 = metrics[metrics["method_name"].isin(af2mm_methods)].reset_index(drop=True)
    non_af2 = metrics[~metrics["method_name"].isin(af2mm_methods)].reset_index(
        drop=True
    )
    pinder_af2 = ["pinder_af2"]
    af2 = af2.query(f"pinder_set in {pinder_af2}").reset_index(drop=True)
    metrics = pd.concat([af2, non_af2]).reset_index(drop=True)
    return metrics


def get_subsampled_train(split_index: pd.DataFrame) -> pd.DataFrame:
    train = split_index.query("split == 'train'").reset_index(drop=True)
    train.loc[:, "apo_count"] = train[["apo_R", "apo_L"]].sum(axis=1).astype(int)
    train.loc[:, "pred_count"] = (
        train[["predicted_R", "predicted_L"]].sum(axis=1).astype(int)
    )
    train.loc[:, "apo_pred_available"] = -(
        (train.apo_count > 0) & (train.pred_count > 0)
    ).astype("int")
    train.loc[:, "apo_available"] = -(train.apo_count > 0).astype("int")
    train.loc[:, "pred_available"] = -(train.pred_count > 0).astype("int")
    split_meta = get_metadata().copy()
    train = pd.merge(
        train,
        split_meta[
            [
                "id",
                "method",
                "num_atom_types",
                "max_var_1",
                "max_var_2",
                "length_resolved_1",
                "length_resolved_2",
                "resolution",
            ]
        ],
        how="inner",
        on="id",
    )
    train.loc[:, "is_xray"] = -np.array([("RAY" in x) for x in train["method"]]).astype(
        int
    )
    train = train[
        (
            train["num_atom_types"] >= 4
        )  # ensures that proteins contain full spectrum of atoms - this is not always the case for low quality structures
        & (
            train["max_var_1"] < 0.98
        )  # top 1 component of the PCA (of coords) should be lower than 0.98 to ignore low complexity elongated structures
        & (train["max_var_2"] < 0.98)
        & (train["length_resolved_1"] > 40)  # length filter for the resolved residues
        & (train["length_resolved_2"] > 40)
    ]
    train["resolution"] = train["resolution"].astype(float)
    sample_ids = set(
        train.sort_values(
            ["apo_available", "pred_available", "is_xray", "resolution"], ascending=True
        )
        .groupby("cluster_id", as_index=False, observed=True)
        .head(3)
        .id
    )
    sampled_train = split_index[split_index["id"].isin(sample_ids)].reset_index(
        drop=True
    )
    return sampled_train


def get_deleaked_sr(
    ialign_data: pd.DataFrame,
    dockq_data: pd.DataFrame,
    log_pvalue_threshold: float,
    rmsd_threshold: float,
    is_score_threshold: float,
    test_id_col: str = "query_id",
) -> dict[str, str | int | float]:
    leak_ids = set(
        ialign_data.query(
            f"log_pvalue < {log_pvalue_threshold} and RMSD < {rmsd_threshold} and `IS-score` > {is_score_threshold}"
        )[test_id_col]
    )
    deleak = dockq_data[~dockq_data["id"].isin(leak_ids)].reset_index(drop=True)
    tot_deleaked = deleak.shape[0]
    tot_success = deleak.query('CAPRI != "Incorrect"').shape[0]
    if tot_deleaked == 0:
        sr = np.nan
    else:
        sr = tot_success / tot_deleaked
    deleaked_sr = {
        "log(P-value)": log_pvalue_threshold,
        "IS-score": is_score_threshold,
        "success": tot_success,
        "total": tot_deleaked,
        "sr": sr,
    }
    return deleaked_sr


def get_diffdock_ialign_sr(
    metrics: pd.DataFrame,
    ialign: pd.DataFrame,
    ialign_config: IalignConfig = IalignConfig(),
) -> pd.DataFrame:
    metric_pqt = (
        get_pinder_location() / "publication_data" / "diffdock_ialign_sr.parquet"
    )
    if metric_pqt.is_file():
        clipped = pd.read_parquet(metric_pqt)
        return clipped
    oracle = (
        metrics.query('monomer_name == "holo"')
        .sort_values("DockQ", ascending=False)
        .drop_duplicates(["id", "eval_dir", "method_name"], keep="first")
        .reset_index(drop=True)
    )
    oracle.loc[oracle.eval_dir.str.contains("dips"), "split"] = "dips_equidock"
    oracle.loc[oracle.eval_dir.str.contains("sequence"), "split"] = "sequence"
    oracle.loc[oracle.eval_dir.str.contains("structure"), "split"] = "pinder"
    oracle.loc[:, "method_eval"] = oracle.method_name + ":" + oracle.split

    split_names = ["pinder", "sequence", "dips_equidock"]
    indices = get_dataset_indices(split=None)
    indices = indices[indices["split_type"].isin(split_names)].reset_index(drop=True)
    index_labs = {
        "sequence": "Sequence identity (40%)",
        "pinder": "PINDER splits",
        "dips_equidock": "DIPS splits",
    }
    metric_list = []
    for k, index in tqdm(indices.groupby("split_type")):
        if "dips" in k:
            # DIPS training used all train split members
            train_sampled = (
                index.query('split == "train"').reset_index(drop=True).copy()
            )
        else:
            train_sampled = get_subsampled_train(index)
        matching = ialign[
            ialign["query_id"].isin(set(index.query('split != "invalid"').id))
            | ialign["hit_id"].isin(set(index.query('split != "invalid"').id))
        ].reset_index(drop=True)
        matching = matching[
            matching["query_id"].isin(set(train_sampled.id))
            | matching["hit_id"].isin(set(train_sampled.id))
        ].reset_index(drop=True)
        for ref in ["query", "hit"]:
            ref_cols = {
                "id": f"{ref}_id",
                "split": f"{ref}_split",
                "cluster_id": f"{ref}_cluster",
            }
            matching = matching.merge(
                index[["id", "split", "cluster_id"]].rename(columns=ref_cols),
                how="left",
            )
        matching = matching[~matching["query_split"].isna()].reset_index(drop=True)
        matching = matching[~matching["hit_split"].isna()].reset_index(drop=True)
        matching.loc[:, "split_pair"] = [
            ":".join(sorted([q, h]))
            for q, h in zip(matching["query_split"], matching["hit_split"])
        ]
        matching.loc[:, "split_type"] = k
        matching.loc[:, "split_label"] = index_labs[k]
        metric_list.append(matching)
    split_metrics = pd.concat(metric_list, ignore_index=True)
    train_metrics = split_metrics.query('split_pair == "test:train"').reset_index(
        drop=True
    )
    train_metrics.loc[:, "test_id"] = [
        qid if qsplit == "test" else hid
        for qid, hid, qsplit in zip(
            train_metrics["query_id"],
            train_metrics["hit_id"],
            train_metrics["query_split"],
        )
    ]
    thresh_sr = []
    for split, oracle_df in oracle.groupby("method_eval"):
        if "dips" in split:
            split_type = "dips_equidock"
        elif "sequence" in split:
            split_type = "sequence"
        else:
            split_type = "pinder"
        sampled_subset = train_metrics.query(
            f'split_type == "{split_type}"'
        ).reset_index(drop=True)
        for thresh in tqdm(np.arange(-150, -1, 0.5)):
            oracle_sr = get_deleaked_sr(
                sampled_subset,
                oracle_df,
                log_pvalue_threshold=thresh,
                rmsd_threshold=ialign_config.rmsd_threshold,
                is_score_threshold=ialign_config.is_score_threshold,
            )
            oracle_sr["rank"] = "Oracle"
            oracle_sr["split"] = split
            thresh_sr.append(oracle_sr)
    ialign_sr = pd.DataFrame(thresh_sr)
    ialign_sr["sr"] = ialign_sr["sr"] * 100
    ialign_sr.loc[:, "split_label"] = [s.split(":")[1] for s in list(ialign_sr.split)]
    ialign_sr.loc[ialign_sr.split_label == "sequence", "split_label"] = "Sequence (40%)"
    ialign_sr.loc[ialign_sr.split_label == "pinder", "split_label"] = "PINDER"
    ialign_sr.loc[ialign_sr.split_label == "dips_equidock", "split_label"] = (
        "DIPS-equidock"
    )
    dips_max_sr = ialign_sr.query('split_label == "DIPS-equidock"').sr.max()
    seq_max_sr = ialign_sr.query('split_label == "Sequence (40%)"').sr.max()
    pinder_max_sr = ialign_sr.query('split_label == "PINDER"').sr.max()
    clipped = ialign_sr.copy()
    clipped.loc[
        (clipped["log(P-value)"] < -50) & (clipped["split_label"] == "PINDER"), "sr"
    ] = pinder_max_sr
    clipped.loc[
        (clipped["log(P-value)"] < -50) & (clipped["split_label"] == "DIPS-equidock"),
        "sr",
    ] = dips_max_sr
    clipped.loc[
        (clipped["log(P-value)"] < -50) & (clipped["split_label"] == "Sequence (40%)"),
        "sr",
    ] = seq_max_sr
    if not metric_pqt.parent.is_dir():
        metric_pqt.parent.mkdir(parents=True)
    clipped.to_parquet(metric_pqt, index=False)
    return clipped


def get_timesplit_pinder_xl_ids(
    config: ClusterConfig = ClusterConfig(),
) -> set[str]:
    xl_meta = meta[meta["id"].isin(set(pindex.query("pinder_xl").id))].reset_index(
        drop=True
    )
    af2_date = date.fromisoformat(config.alphafold_cutoff_date)
    xl_meta["af2"] = xl_meta["release_date"].apply(
        lambda rd: date.fromisoformat(rd) > af2_date
    )
    all_af2_ids = set(xl_meta.query("af2").id)
    return all_af2_ids


def get_af2mm_ialign_sr(
    metrics: pd.DataFrame,
    ialign: pd.DataFrame,
    config: ClusterConfig = ClusterConfig(),
    ialign_config: IalignConfig = IalignConfig(),
) -> pd.DataFrame:
    metric_pqt = get_pinder_location() / "publication_data" / "af2mm_ialign_sr.parquet"
    if not metric_pqt.parent.is_dir():
        metric_pqt.parent.mkdir(parents=True)
    if metric_pqt.is_file():
        clipped = pd.read_parquet(metric_pqt)
        return clipped

    metrics = metrics.query('method_name == "af2mm"').reset_index(drop=True)
    oracle = metrics.sort_values("DockQ", ascending=False).drop_duplicates(
        "id", keep="first"
    )
    top1 = metrics.sort_values("rank").drop_duplicates("id", keep="first")
    af2_date = date.fromisoformat(config.alphafold_cutoff_date)
    for ref in ["query", "hit"]:
        ref_cols = {
            "id": f"{ref}_id",
            "split": f"{ref}_split",
            "cluster_id": f"{ref}_cluster",
            "release_date": f"{ref}_release",
        }
        ialign = ialign.merge(
            pindex[["id", "split", "cluster_id"]].rename(columns=ref_cols), how="left"
        ).merge(meta[["id", "release_date"]].rename(columns=ref_cols), how="left")
    ialign["hit_date_leak"] = ialign.hit_release.apply(
        lambda rd: date.fromisoformat(rd) <= af2_date
    )
    ialign["query_date_leak"] = ialign.query_release.apply(
        lambda rd: date.fromisoformat(rd) <= af2_date
    )
    all_af2_ids = get_timesplit_pinder_xl_ids(config=config)
    oracle_af2 = oracle[oracle["id"].isin(all_af2_ids)].reset_index(drop=True)
    top1_af2 = top1[top1["id"].isin(all_af2_ids)].reset_index(drop=True)
    ialign_timesplit = ialign.query("~query_date_leak and hit_date_leak").reset_index(
        drop=True
    )
    thresh_sr = []
    for thresh in np.arange(-150, -1, 0.5):
        oracle_sr = get_deleaked_sr(
            ialign_timesplit,
            oracle_af2,
            log_pvalue_threshold=thresh,
            rmsd_threshold=ialign_config.rmsd_threshold,
            is_score_threshold=ialign_config.is_score_threshold,
        )
        oracle_sr["rank"] = "Oracle"
        thresh_sr.append(oracle_sr)
        top1_sr = get_deleaked_sr(
            ialign_timesplit,
            top1_af2,
            log_pvalue_threshold=thresh,
            rmsd_threshold=ialign_config.rmsd_threshold,
            is_score_threshold=ialign_config.is_score_threshold,
        )
        top1_sr["rank"] = "Top 1"
        thresh_sr.append(top1_sr)
    ialign_sr = pd.DataFrame(thresh_sr)
    ialign_sr["sr"] = ialign_sr["sr"] * 100
    oracle_max_sr = ialign_sr.query('rank == "Oracle"').sr.max()
    top1_max_sr = ialign_sr.query('rank == "Top 1"').sr.max()
    clipped = ialign_sr.copy()
    clipped.loc[
        (clipped["log(P-value)"] < -50) & (clipped["rank"] == "Oracle"), "sr"
    ] = oracle_max_sr
    clipped.loc[
        (clipped["log(P-value)"] < -50) & (clipped["rank"] == "Top 1"), "sr"
    ] = top1_max_sr
    clipped.to_parquet(metric_pqt, index=False)
    return clipped


def get_af2mm_neff_sr(
    metrics: pd.DataFrame,
    config: ClusterConfig = ClusterConfig(),
) -> pd.DataFrame:
    metric_pqt = get_pinder_location() / "publication_data" / "af2mm_neff_sr.parquet"
    if not metric_pqt.parent.is_dir():
        metric_pqt.parent.mkdir(parents=True)
    if metric_pqt.is_file():
        clipped = pd.read_parquet(metric_pqt)
        return clipped

    metrics = metrics.query('method_name == "af2mm"').reset_index(drop=True)
    oracle = metrics.sort_values("DockQ", ascending=False).drop_duplicates(
        "id", keep="first"
    )
    top1 = metrics.sort_values("rank").drop_duplicates("id", keep="first")
    all_af2_ids = get_timesplit_pinder_xl_ids(config=config)
    oracle_af2 = oracle[oracle["id"].isin(all_af2_ids)].reset_index(drop=True)
    top1_af2 = top1[top1["id"].isin(all_af2_ids)].reset_index(drop=True)
    paired_neff = get_supplementary_data("paired_neff")
    oracle_af2 = pd.merge(oracle_af2, paired_neff, how="left")
    top1_af2 = pd.merge(top1_af2, paired_neff, how="left")
    thresh_sr = []
    for neff_thresh in tqdm(np.arange(0.0, 2500, 50)):
        neff_data = oracle_af2.query(f"neff <= {neff_thresh}").reset_index(drop=True)
        success = neff_data.query('CAPRI != "Incorrect"').shape[0]
        tot = neff_data.shape[0]
        if tot == 0:
            sr = np.nan
        else:
            sr = success / tot
        thresh_sr.append(
            {
                "Neff": neff_thresh,
                "sr": sr,
                "total": tot,
                "success": success,
                "rank": "Oracle",
            }
        )
        neff_data = top1_af2.query(f"neff <= {neff_thresh}").reset_index(drop=True)
        success = neff_data.query('CAPRI != "Incorrect"').shape[0]
        tot = neff_data.shape[0]
        if tot == 0:
            sr = np.nan
        else:
            sr = success / tot
        thresh_sr.append(
            {
                "Neff": neff_thresh,
                "sr": sr,
                "total": tot,
                "success": success,
                "rank": "Top 1",
            }
        )
    neff_sr = pd.DataFrame(thresh_sr)
    neff_sr["sr"] = neff_sr["sr"] * 100
    oracle_max_sr = neff_sr.query('rank == "Oracle"').sr.max()
    top1_max_sr = neff_sr.query('rank == "Top 1"').sr.max()
    clipped = neff_sr.copy()
    clipped.loc[(clipped["Neff"] > 1000) & (clipped["rank"] == "Oracle"), "sr"] = (
        oracle_max_sr
    )
    clipped.loc[(clipped["Neff"] > 1000) & (clipped["rank"] == "Top 1"), "sr"] = (
        top1_max_sr
    )
    clipped.to_parquet(metric_pqt, index=False)
    return clipped


def sr_curve(
    data: pd.DataFrame,
    color: str,
    x: str = "log(P-value)",
    y: str = "sr",
    theme: Theme = LightTheme(),
) -> Figure:
    lp = LinePlot(theme=theme)
    custom_xaxis_title = "<b>" + pc.LABELS.get(x, x) + "</b>"
    custom_yaxis_title = "<b>" + pc.LABELS.get(y, y) + "</b>"
    if color == "split_label":
        color_map = {
            "PINDER": "#44BB99",
            "Sequence (40%)": "#FFAABB",
            "DIPS-equidock": "#99DDFF",
        }
    else:
        color_map = pc.rank_color_map
    fig = lp.lineplot(
        data=data,
        x=x,
        color=color,
        y=y,
        labels=pc.LABELS,
        color_discrete_map=color_map,
        width=885,
        height=650,
        custom_xaxis_title=custom_xaxis_title,
        custom_yaxis_title=custom_yaxis_title,
        hide_legend_title=True,
        grid_x=False,
        line_dash=color if color == "split_label" else None,
    )
    return fig


def reference_based_violin_plots(
    dockq: pd.DataFrame,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    theme: Theme = LightTheme(),
    format: str = "pdf",
) -> None:
    fig_subdir = fig_dir / "rmsd-violins"
    rmsd_methods = [
        "af2mm",
        "diffdockpp_subset1_train2",
        "frodock",
        "hdock",
        "patchdock",
    ]
    dockq.loc[:, "monomer_name_label"] = [
        m[0].upper() + m[1:] if m != "predicted" else "Predicted (AF2)"
        for m in dockq.monomer_name
    ]
    dockq.loc[:, "Method"] = [
        "<b>" + pc.METHOD_LABELS.get(m, m) + "</b>" for m in dockq.method_name
    ]
    irms_data = (
        dockq[dockq["method_name"].isin(rmsd_methods)]
        .query("iRMS <= 30")
        .reset_index(drop=True)
    )
    lrms_data = (
        dockq[dockq["method_name"].isin(rmsd_methods)]
        .query("LRMS <= 60")
        .reset_index(drop=True)
    )
    vp = ViolinPlot(theme=theme)
    for pinder_set in pc.PINDER_SETS:
        for data, metric in [(irms_data, "iRMS"), (lrms_data, "LRMS")]:
            data = data.query(f'pinder_set == "{pinder_set}"').reset_index(drop=True)
            cat_orders = {
                "monomer_name_label": [
                    monomer
                    for monomer in ["Holo", "Apo", "Predicted (AF2)"]
                    if monomer in set(data.monomer_name_label)
                ],
                "Method": [
                    mn
                    for mn in list(pc.METHOD_LABELS.values())
                    if mn in set(data.Method)
                ],
            }
            kwargs = dict(
                data=data,
                x=metric,
                y="Method",
                color="monomer_name_label",
                color_discrete_map=pc.monomer_color_map,
                facet_col="monomer_name_label",
                category_orders=cat_orders,
                width=1500,
                height=1400,
                labels=pc.LABELS,
                show_legend=False,
                hide_xaxis_title=False,
                hide_yaxis_title=True,
                shared_xaxis_title=True,
                custom_xaxis_title=f"<b> {pc.LABELS.get(metric, metric)}</b>",
                grid_x=True,
                grid_y=False,
                shared_xaxis_y_loc=-0.07,
            )
            fig = vp.violinplot(**kwargs)
            futils.write_fig(fig, fig_subdir / f"{metric}-violin-{pinder_set}.{format}")


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
    config: ClusterConfig = ClusterConfig(),
    ialign_config: IalignConfig = IalignConfig(),
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    log.info("Fetching DockQ metrics...")
    metrics = get_dockq_metrics()
    log.info("Fetching iAlign metrics...")
    ialign = get_ialign_metrics()
    log.info("Calculating AF2MM ialign success rates...")
    af2_sr = get_af2mm_ialign_sr(
        metrics, ialign=ialign, config=config, ialign_config=ialign_config
    )
    log.info("Calculating DiffDock-PP ialign success rates...")
    diffdock_sr = get_diffdock_ialign_sr(
        metrics=metrics, ialign=ialign, ialign_config=ialign_config
    )
    for sr, color, lab in zip(
        [af2_sr, diffdock_sr], ["rank", "split_label"], ["af2mm", "diffdock"]
    ):
        fig = sr_curve(data=sr.query("`log(P-value)` >= -51"), color=color, theme=theme)
        fig_subdir = fig_dir / f"{lab}_ialign_sr"
        output_file = fig_subdir / f"{lab}-success-rate-vs-ialign.{format}"
        futils.write_fig(fig, output_file)
        fig.show()
    af2_sr = get_af2mm_neff_sr(metrics, config=config)
    min_neff = af2_sr.query("total > 0").Neff.min()
    fig = sr_curve(
        data=af2_sr.query(f"Neff <= 1002 and Neff >= {min_neff-1}"),
        x="Neff",
        color="rank",
        theme=theme,
    )
    fig_subdir = fig_dir / "af2mm_neff_sr"
    output_file = fig_subdir / f"af2mm-success-rate-vs-neff.{format}"
    futils.write_fig(fig, output_file)

    # LRMS and iRMS violin plots per pinder set
    metrics = get_penalized_dockq()
    reference_based_violin_plots(dockq=metrics)


if __name__ == "__main__":
    main()
