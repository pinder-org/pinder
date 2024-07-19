from __future__ import annotations
from pathlib import Path

import pandas as pd

from pinder.core import get_index, get_metadata, get_pinder_location
from pinder.core.index.utils import get_pinder_bucket_root
from pinder.core.utils.cloud import gcs_read_dataframe
from pinder.core.utils.log import setup_logger
from pinder.data.plot import (
    Theme,
    DarkTheme,
    LightTheme,
    BarPlot,
    ViolinPlot,
    constants as pc,
    figure_utils as futils,
)
from pinder.data.plot.performance import (
    remove_af2mm_nonaf2,
    patch_af2mm_monomer_types,
    select_top_n,
)
from pinder.eval.dockq.method import add_pinder_set


log = setup_logger(__name__)

pindex = get_index()
meta = get_metadata()


def get_ref_free_metrics(metric_type: str, xtal: bool) -> pd.DataFrame:
    data_root = get_pinder_bucket_root() + "/figure_data/validity/"
    group = "native" if xtal else "method"
    pqt_name = f"pinder_{group}_{metric_type}.parquet"
    metrics = gcs_read_dataframe(data_root + pqt_name)
    return metrics


def add_xtal_dummy_dockq(pinder_metrics: pd.DataFrame) -> pd.DataFrame:
    pinder_metrics.loc[:, "rank"] = 1
    pinder_metrics.loc[:, "DockQ"] = 1.0
    pinder_metrics.loc[:, "iRMS"] = 0.0
    pinder_metrics.loc[:, "LRMS"] = 0.0
    pinder_metrics.loc[:, "CAPRI"] = "High"
    pinder_metrics.loc[:, "CAPRI_rank"] = 3
    return pinder_metrics


def split_pinder_masked_unmasked(
    pinder_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pinder_masked = pinder_metrics[pinder_metrics.holo_mask].reset_index(drop=True)
    pinder_unmasked = pinder_metrics[~pinder_metrics.holo_mask].reset_index(drop=True)
    pinder_unmasked.loc[:, "method_name"] = "Crystal structures (raw)"
    pinder_masked.loc[:, "method_name"] = "Crystal structures (holo sequence mask)"
    pinder_masked = add_xtal_dummy_dockq(pinder_masked)
    pinder_unmasked = add_xtal_dummy_dockq(pinder_unmasked)
    return pinder_masked, pinder_unmasked


def get_success_rate(data: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    total_poses = (
        data.groupby(["Method", "monomer_name", "pinder_set"], as_index=False)
        .size()
        .rename({"size": "total_poses"}, axis=1)
    )
    capri_poses = (
        data.query('CAPRI != "Incorrect"')
        .groupby(["Method", "monomer_name", "pinder_set"], as_index=False)
        .size()
        .rename({"size": "capri_poses"}, axis=1)
    )
    clash_pass = (
        data.query(f"passes_{filter_name}")
        .groupby(["Method", "monomer_name", "pinder_set"], as_index=False)
        .size()
        .rename({"size": "passes_min_dist"}, axis=1)
    )
    capri_clash_pass = (
        data.query(f'passes_{filter_name} and CAPRI != "Incorrect"')
        .groupby(["Method", "monomer_name", "pinder_set"], as_index=False)
        .size()
        .rename({"size": f"capri_{filter_name}"}, axis=1)
    )
    sr = total_poses.merge(capri_poses).merge(clash_pass).merge(capri_clash_pass)
    sr.loc[:, "percent_capri"] = (sr.capri_poses / sr.total_poses) * 100
    sr.loc[:, f"percent_capri_{filter_name}"] = (
        sr[f"capri_{filter_name}"] / sr.total_poses
    ) * 100
    return sr


def combine_native_and_predicted(
    pinder_metrics: pd.DataFrame,
    method_metrics: pd.DataFrame,
    metric_type: str,
) -> pd.DataFrame:
    if metric_type == "voromqa":
        cols = pc.VOROMQA_COLUMNS
    else:
        cols = pc.CLASH_COLUMNS

    pinder_masked, pinder_unmasked = split_pinder_masked_unmasked(pinder_metrics)
    combined = pd.concat(
        [method_metrics[cols], pinder_masked[cols], pinder_unmasked[cols]]
    ).reset_index(drop=True)

    # Might not want to add penalty to ref-free plots, but staying consistent.
    combined = add_pinder_set(combined, allow_missing=False)
    combined = remove_af2mm_nonaf2(combined)
    combined = combined[cols + ["pinder_set"]]
    combined.loc[:, "Method"] = [pc.LABELS.get(m, m) for m in combined.method_name]
    combined.loc[:, "pinder_set"] = [
        pc.LABELS.get(ds, ds) for ds in combined.pinder_set
    ]
    if metric_type == "voromqa":
        combined.loc[:, "passes_vmqa_clash"] = (
            combined.sel_voromqa_v1_clash_score <= 0.1
        )
    else:
        combined.loc[:, "passes_min_dist"] = combined.min_dist > 1.2
    return combined


def validity_bar_plot(
    sr_data: pd.DataFrame,
    sr_metric: str,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    theme: Theme = LightTheme(),
    format: str = "pdf",
) -> None:
    bp = BarPlot(theme=theme)
    fig_subdir = fig_dir / "pose-validity"
    for pinder_set in pc.PINDER_SETS:
        for mono in ["holo", "apo", "predicted"]:
            pose_count = sr_data.query(
                f'pinder_set == "{pinder_set}" and monomer_name == "{mono}"'
            ).reset_index(drop=True)
            pose_count["xtal"] = pose_count.Method.str.contains("Crystal")
            pose_count = (
                pose_count[
                    (~pose_count.xtal)
                    | (pose_count.xtal & (pose_count.top_n == "Top 1"))
                ]
                .drop(columns="xtal")
                .reset_index(drop=True)
            )
            fig = bp.grouped_stacked_bar(
                data=pose_count,
                x="Method",
                group_col="top_n",
                y_cols=["percent_capri", f"percent_capri_{sr_metric}"],
                labels=pc.LABELS,
                y_title="Percentage of predictions",
                legend_font_size=20,
                font_size=24,
                x_order=[
                    m for m in pc.METHOD_LABELS.values() if m in set(pose_count.Method)
                ],
                legend_orientation="v",
            )
            futils.write_fig(
                fig, fig_subdir / f"capri-{sr_metric}-sr--{pinder_set}_{mono}.{format}"
            )


def ref_free_violin_plot(
    metrics: pd.DataFrame,
    metric_col: str,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    theme: Theme = LightTheme(),
    format: str = "pdf",
) -> None:
    method_order = [
        "Crystal structures<br>(no mask)</br>",
        "Crystal structures<br>(holo sequence mask)</br>",
        "Crystal structures",
        "AlphaFold-Multimer",
        "DiffDock-PP",
        "FRODOCK",
        "HDOCK",
        "PatchDock",
    ]
    fig_subdir = fig_dir / "ref-free"
    theme.marker_line_color = "rgba(214, 214, 214, 0.1)"
    vp = ViolinPlot(theme=theme)
    for pinder_set in pc.PINDER_SETS:
        data = metrics.query(f'pinder_set == "{pinder_set}"').reset_index(drop=True)
        data = data.query(
            'method_name != "Crystal structures (holo sequence mask)"'
        ).reset_index(drop=True)
        data.loc[data.method_name == "Crystal structures (raw)", "method_name"] = (
            "Crystal structures"
        )
        data = data.query(
            '(method_name == "Crystal structures" and monomer_name == "holo") or (method_name != "Crystal structures")'
        ).reset_index(drop=True)
        # Copy xtal holo into other monomers for facet col
        # only show xtal structures as holo
        xtal_holo = data.query(
            '(method_name == "Crystal structures" and monomer_name == "holo")'
        ).reset_index(drop=True)
        xtal_apo = xtal_holo.copy()
        xtal_pred = xtal_holo.copy()
        xtal_apo.loc[:, "monomer_name"] = "apo"
        xtal_pred.loc[:, "monomer_name"] = "predicted"
        data = pd.concat([data, xtal_apo, xtal_pred]).reset_index(drop=True)
        data.loc[:, "monomer_name_label"] = [
            m[0].upper() + m[1:] if m != "predicted" else "Predicted (AF2)"
            for m in data.monomer_name
        ]
        data.loc[:, "Method"] = [
            "<b>" + pc.LABELS.get(m, m) + "</b>" for m in data.method_name
        ]
        kwargs = dict(
            data=data,
            x=metric_col,
            y="Method",
            color="monomer_name_label",
            color_discrete_map=pc.monomer_color_map,
            category_orders={
                "Method": [
                    "<b>" + k + "</b>"
                    for k in method_order
                    if "<b>" + k + "</b>" in set(data.Method)
                ],
                "monomer_name_label": ["Holo", "Apo", "Predicted (AF2)"],
            },
            height=1400,
            width=1500,
            labels=pc.LABELS,
            show_legend=False,
            hide_xaxis_title=False,
            hide_yaxis_title=True,
            shared_xaxis_title=True,
            grid_x=False,
            grid_y=False,
            vrect=[0.0, 1.2],
            facet_col="monomer_name_label",
            facet_col_wrap=3,
            marker_line_width=0.1,
            marker_size=5,
        )
        fig = vp.violinplot(**kwargs)
        fig = futils.update_layout(
            fig,
            font=dict(size=24, family="Helvetica"),
            margin=dict(
                l=50,
                r=50,
                b=140,
                t=50,
            ),
        )
        fig = fig.for_each_trace(
            lambda t: t.update(
                width=0.6,
                scalemode="width",
                opacity=1.0,
                meanline=dict(visible=True, color="#b3b3b3", width=2),
                box=dict(
                    visible=True,
                    width=0.2,
                    fillcolor="rgba(214, 214, 214, 0.1)",
                    line=dict(width=1, color="#333"),
                ),
            )
        )
        futils.write_fig(
            fig,
            fig_subdir / f"{metric_col.replace('_', '-')}-violin-{pinder_set}.{format}",
        )


def copy_xtal_apo_pred_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics.loc[:, "xtal"] = metrics.Method.str.contains("Crystal")
    metrics = metrics.query(
        "Method != 'Crystal structures<br>(holo sequence mask)</br>'"
    ).reset_index(drop=True)
    metrics.loc[metrics.Method.str.contains("Crystal"), "Method"] = "Crystal structures"
    metrics = metrics.query(
        '(xtal == False) or (xtal == True and monomer_name == "holo")'
    ).reset_index(drop=True)
    holo = metrics.query('xtal == True and monomer_name == "holo"').reset_index(
        drop=True
    )
    apo = holo.copy()
    apo.loc[:, "monomer_name"] = "apo"
    pred = holo.copy()
    pred.loc[:, "monomer_name"] = "predicted"
    metrics = pd.concat([metrics, apo, pred]).reset_index(drop=True)
    return metrics


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    pinder_clashes = get_ref_free_metrics("vdw_clashes", xtal=True)
    pinder_vmqa = get_ref_free_metrics("voromqa", xtal=True)
    method_clashes = get_ref_free_metrics("vdw_clashes", xtal=False)
    method_vmqa = get_ref_free_metrics("voromqa", xtal=False)
    # Remove sequence-split eval from metrics
    method_vmqa = method_vmqa.query(
        'method_name != "diffdockpp_seq_subset1_train1"'
    ).reset_index(drop=True)
    method_clashes = method_clashes.query(
        'method_name != "diffdockpp_seq_subset1_train1"'
    ).reset_index(drop=True)

    clash_all = combine_native_and_predicted(pinder_clashes, method_clashes, "min_dist")
    vmqa_all = combine_native_and_predicted(pinder_vmqa, method_vmqa, "voromqa")
    clash_all = patch_af2mm_monomer_types(clash_all)
    vmqa_all = patch_af2mm_monomer_types(vmqa_all)

    # Make it top1, top5, oracle
    method_top1_clashes = select_top_n(clash_all, "rank", n=1)
    method_top1_vmqa = select_top_n(vmqa_all, "rank", n=1)
    method_top5_clashes = select_top_n(clash_all, "rank", n=5)
    method_top5_vmqa = select_top_n(vmqa_all, "rank", n=5)
    method_oracle_clashes = select_top_n(clash_all, "DockQ", n=1, ascending=False)
    method_oracle_vmqa = select_top_n(vmqa_all, "DockQ", n=1, ascending=False)
    rank_sr = []
    for rank, data in zip(
        ["Top 1", "Top 5", "Oracle"],
        [method_top1_vmqa, method_top5_vmqa, method_oracle_vmqa],
    ):
        data = get_success_rate(data, "vmqa_clash")
        data.loc[:, "top_n"] = rank
        rank_sr.append(data)
    sr_vmqa = pd.concat(rank_sr, ignore_index=True)
    rank_sr = []
    for rank, data in zip(
        ["Top 1", "Top 5", "Oracle"],
        [method_top1_clashes, method_top5_clashes, method_oracle_clashes],
    ):
        data = get_success_rate(data, "min_dist")
        data.loc[:, "top_n"] = rank
        rank_sr.append(data)
    sr_min_dist = pd.concat(rank_sr, ignore_index=True)
    # Only plot holo xtal structures as reference -- repeat them for apo/predicted in the plots
    sr_vmqa = copy_xtal_apo_pred_metrics(sr_vmqa)
    sr_min_dist = copy_xtal_apo_pred_metrics(sr_min_dist)
    # No confidence model trained for DiffDock-PP, remove Top1/5 ranked values
    sr_vmqa = sr_vmqa[
        ~((sr_vmqa.Method == "DiffDock-PP") & (sr_vmqa.top_n.isin(["Top 1", "Top 5"])))
    ].reset_index(drop=True)
    sr_min_dist = sr_min_dist[
        ~(
            (sr_min_dist.Method == "DiffDock-PP")
            & (sr_min_dist.top_n.isin(["Top 1", "Top 5"]))
        )
    ].reset_index(drop=True)

    bar_args = [(sr_vmqa, "vmqa_clash"), (sr_min_dist, "min_dist")]
    for data, column in bar_args:
        validity_bar_plot(
            data,
            column,
            fig_dir=fig_dir,
            theme=theme,
            format=format,
        )
    violin_args = [
        (clash_all.query("min_dist <= 4").reset_index(drop=True), "min_dist"),
        (
            clash_all.query("min_dist_vdw_ratio < 2").reset_index(drop=True),
            "min_dist_vdw_ratio",
        ),
        (
            vmqa_all.query(
                "sel_voromqa_v1_energy_norm <= 1 and sel_voromqa_v1_energy_norm >= -1.1"
            ).reset_index(drop=True),
            "sel_voromqa_v1_clash_score",
        ),
    ]
    for data, column in violin_args:
        ref_free_violin_plot(
            metrics=data,
            metric_col=column,
            fig_dir=fig_dir,
            theme=theme,
            format=format,
        )


if __name__ == "__main__":
    main()
