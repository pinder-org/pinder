from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from scipy.stats import gaussian_kde

from pinder.core import get_pinder_location
from pinder.core.index.utils import get_pinder_bucket_root
from pinder.core.utils.cloud import gcs_read_dataframe
from pinder.data.plot import (
    Colors,
    DarkTheme,
    LightTheme,
    Theme,
    constants as pc,
    figure_utils as futils,
)
from pinder.data.plot.datasets import get_dataset_indices
from pinder.data.plot.image import image_grid
from pinder.data.plot.performance import get_ialign_metrics
from pinder.data.plot.plot import format_axes, format_legend, format_text
from pinder.data.qc import annotation_check, pfam_diversity, uniprot_leakage


def multi_level_leakage_diversity_bar_plots(
    leakage: pd.DataFrame,
    y_col: str = "Percentage",
    primary_x_col: str = "Split",
    primary_x_order: list[str] | None = ["Test", "Val"],
    secondary_x_col: str = "Metric",
    color_col: str = "Dataset",
    color_col_order: list[str] | None = [
        "PINDER",
        "ProteinFlow",
        "DIPS-equidock",
        "PPIRef",
    ],
    width: int = 1600,
    height: int = 650,
    bar_width: float = 0.1,
    y_title: str | None = None,
    color_discrete_map: dict[str, str] = pc.dataset_color_map,
    legend: dict[str, Any] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xref="paper",
        x=0.01,
        font=dict(size=32),
    ),
    theme: Theme = LightTheme(),
) -> Figure:
    figure_width = width
    figure_height = height
    if primary_x_order:
        primary_vals = primary_x_order
    else:
        primary_vals = list(set(leakage[primary_x_col]))
    if color_col_order:
        color_group_names = color_col_order
    else:
        color_group_names = list(set(leakage[color_col]))

    measures = sorted(set(leakage[secondary_x_col]))
    leakage[secondary_x_col] = leakage[secondary_x_col].astype("category")
    leakage[secondary_x_col] = leakage[secondary_x_col].cat.reorder_categories(measures)
    leakage = leakage.sort_values(secondary_x_col).reset_index(drop=True)
    n_measures = len(measures)
    split_tick_pos = [
        (bar_width * len(color_group_names)) + bar_width / 2 + k
        for k in range(n_measures)
    ]

    base = 0
    offset = 0.0
    bar_count = 0
    figure_data = []
    offset_positions = []
    split_positions = []
    for i, split in enumerate(primary_vals):
        for j, dataset in enumerate(color_group_names):
            offset = bar_width * bar_count + (bar_width / len(primary_vals)) * i
            if j % 2 == 0 and j > 0:
                split_positions.extend([offset + k for k in range(n_measures)])
            offset_positions.append(offset)
            bar_count += 1
            bar = go.Bar(
                y=list(
                    leakage.query(
                        f'{color_col} == "{dataset}" and {primary_x_col} == "{split}"'
                    )[y_col]
                ),
                base=base,
                width=bar_width,
                offset=offset,
                textposition="outside",
                texttemplate="%{y:.1f}%",
                textfont=dict(size=26),
                outsidetextfont=dict(size=18),
                name=dataset,
                legendgroup=dataset,
                showlegend=False,
                marker=dict(
                    color=color_discrete_map[dataset],
                    line=dict(width=2, color=theme.marker_line_color),
                ),
            )
            figure_data.append(bar)
            # Add pseudo-trace to use for legend group since bar plots can't have legend symbol size scaled
            scatter = go.Scatter(
                y=None,
                x=[-5],
                marker_symbol="square",
                name=dataset,
                legendgroup=dataset,
                showlegend=i == 0,
                marker=dict(
                    color=color_discrete_map[dataset],
                    size=200,
                    line=dict(width=2, color=theme.marker_line_color),
                ),
            )
            figure_data.append(scatter)

    fig = go.Figure(data=figure_data)
    tickvals = sorted(split_positions)
    ticktext = []
    for i in range(n_measures):
        ticktext.extend(primary_vals)

    annotations = [
        dict(x=X, y=Y, text=t, xref="x", yref="paper", showarrow=False)
        for X, Y, t in zip(split_tick_pos, [-0.15] * n_measures, measures)
    ]
    fig.update_layout(
        xaxis=dict(title=primary_x_col),
        xaxis2=dict(title=secondary_x_col, overlaying="x", side="top"),
        yaxis=dict(title=f"{y_col} (%)"),
        width=figure_width,
        height=figure_height,
        template="simple_white",
        xaxis_tickvals=tickvals,
        xaxis_ticktext=ticktext,
        xaxis_range=[-0.1, n_measures],
        yaxis_range=[0.0, leakage[y_col].max() + 5],
        annotations=annotations,
        legend=legend,
    )

    # Add vertical lines to separate each primary axis label (split) as if its a separate group
    midpoint = (bar_width * len(color_group_names)) + bar_width / 4
    for k in range(n_measures):
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=k + midpoint,
            y0=0.0,
            x1=k + midpoint,
            y1=-0.05,
            line=dict(color="black", width=2),
        )

    # Add vertical lines to separate each measure as if its a facet plot
    for k in range(n_measures - 1):
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=k + 0.95,
            y0=0.0,
            x1=k + 0.95,
            y1=-0.15,
            line=dict(color="black", width=2),
        )

    fig = futils.remove_xaxis_titles(fig)
    fig = futils.remove_legend_title(fig)
    fig = futils.update_layout(
        fig,
        font=dict(size=22, family="Helvetica"),
        margin=dict(
            l=100,
            r=20,
            b=100,
            t=50,
        ),
    )
    fig = futils.bold_trace_name(fig)
    fig = futils.bold_annotations(fig)
    if y_title:
        y_title_text = f"<b>{y_title}</b>"
    else:
        y_title_text = f"<b>{y_col}</b>"

    fig = futils.update_axes(
        fig,
        "y",
        theme.axis_line_color,
        theme.tick_color,
        theme.axis_title_color,
        showgrid=False,
        linewidth=theme.axis_linewidth,
        title_text=y_title_text,
    )
    fig = futils.update_axes(
        fig,
        "x",
        theme.axis_line_color,
        theme.tick_color,
        theme.axis_title_color,
        showgrid=False,
        linewidth=theme.axis_linewidth,
    )
    return fig


def ialign_vs_idist_density(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme: Theme = LightTheme(),
) -> None:
    cache_pqt = (
        get_pinder_location() / "publication_data" / "idist_vs_ialign_score_kde.parquet"
    )
    if cache_pqt.is_file():
        idist = pd.read_parquet(cache_pqt)
    else:
        # Load the merged DataFrame
        ialign_idist_pqt = (
            get_pinder_bucket_root()
            + "/figure_data/ialign_vs_idist/ialign_idist_common.parquet"
        )
        idist = gcs_read_dataframe(ialign_idist_pqt)
        xy = np.vstack([idist["IS-score"], idist["idist"]])
        z = gaussian_kde(xy)(xy)
        logz = np.log10(z)
        idist.loc[:, "log_z"] = logz
        idist.to_parquet(cache_pqt, index=False)
    labels = {
        "log_z": "Log density",
        "idist": "iDist(I,J)",
        "IS-score": "iAlign<sub>IS-score</sub>(I,J)",
    }
    fig = px.scatter(
        idist,
        x="IS-score",
        y="idist",
        color="log_z",
        template=theme.template,
        width=850,
        height=650,
        color_continuous_scale="Agsunset",
        labels=labels,
    )
    fig = fig.add_vline(x=0.3, line_width=4, line_dash="dash", line_color=Colors.navy)
    fig = fig.add_hline(y=0.03, line_width=4, line_dash="dash", line_color=Colors.navy)
    fig = format_axes(
        fig,
        x="IS-score",
        y="idist",
        theme=theme,
        labels=labels,
        shared_xaxis_title=True,
        shared_yaxis_title=True,
        shared_xaxis_y_loc=-0.175,
        shared_yaxis_x_loc=-0.2,
    )
    fig = format_text(fig)
    fig = format_legend(fig, show_legend=False)
    fig_subdir = fig_dir / "idist_vs_ialign"
    output_file = fig_subdir / f"idist_vs_ialign_kde.{format}"
    futils.write_fig(fig, output_file)


def get_ialign_leakage() -> pd.DataFrame:
    split_names = ["pinder", "ppiref", "proteinflow", "dips_equidock"]
    ialign = get_ialign_metrics()
    indices = get_dataset_indices(split=None)
    indices = indices[indices["split_type"].isin(split_names)].reset_index(drop=True)
    leak_report = []
    for (k, label), index in indices.groupby(["split_type", "Dataset"]):
        matching = ialign[
            ialign["query_id"].isin(set(index.query('split != "invalid"').id))
            | ialign["hit_id"].isin(set(index.query('split != "invalid"').id))
        ].reset_index(drop=True)
        for side in ["query", "hit"]:
            col_map = {
                "id": f"{side}_id",
                "split": f"{side}_split",
                "cluster_id": f"{side}_cluster",
            }
            matching = pd.merge(
                matching,
                index[["id", "split", "cluster_id"]].rename(columns=col_map),
                how="left",
            )
            matching = matching[~matching[f"{side}_split"].isna()].reset_index(
                drop=True
            )
        matching.loc[:, "split_pair"] = [
            ":".join(sorted([q, h]))
            for q, h in zip(matching["query_split"], matching["hit_split"])
        ]
        matching.loc[:, "split_type"] = k
        matching.loc[:, "split_label"] = label
        test_metrics = matching.query('split_pair == "test:train"').reset_index(
            drop=True
        )
        test_metrics.loc[:, "test_id"] = [
            qid if qsplit == "test" else hid
            for qid, hid, qsplit in zip(
                test_metrics["query_id"],
                test_metrics["hit_id"],
                test_metrics["query_split"],
            )
        ]
        val_metrics = matching.query('split_pair == "train:val"').reset_index(drop=True)
        val_metrics.loc[:, "val_id"] = [
            qid if qsplit == "val" else hid
            for qid, hid, qsplit in zip(
                val_metrics["query_id"],
                val_metrics["hit_id"],
                val_metrics["query_split"],
            )
        ]
        val_tot = index.query('split == "val"').shape[0]
        test_tot = index.query('split == "test"').shape[0]
        train = index.query('split == "train"').reset_index(drop=True)
        val_leaks = set(
            val_metrics[
                val_metrics["query_id"].isin(set(train.id))
                | val_metrics["hit_id"].isin(set(train.id))
            ]
            .query("log_pvalue < -9 and `IS-score` > 0.3 and RMSD < 5.0")
            .val_id
        )
        test_leaks = set(
            test_metrics[
                test_metrics["query_id"].isin(set(train.id))
                | test_metrics["hit_id"].isin(set(train.id))
            ]
            .query("log_pvalue < -9 and `IS-score` > 0.3 and RMSD < 5.0")
            .test_id
        )
        val_leak_pct = (len(val_leaks) / val_tot) * 100
        test_leak_pct = (len(test_leaks) / test_tot) * 100
        leak_report.append(
            {
                "Dataset": label,
                "Measure": "Leakage",
                "Metric": "iAlign interface pair",
                "Test": round(test_leak_pct, 2),
                "Val": round(val_leak_pct, 2),
            }
        )
    leak_report = pd.DataFrame(leak_report)
    return leak_report


def get_summary_diversity_and_leakage_stats(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    use_cache: bool = True,
) -> pd.DataFrame:
    output_pqt = fig_dir / "data/diversity_and_leakage_report.parquet"
    if use_cache and output_pqt.is_file():
        report = pd.read_parquet(output_pqt)
        return report

    ds_names = [
        "DIPS-equidock",
        "PPIRef",
        "ProteinFlow",
        "PINDER",
    ]
    ds_keys = [n.replace("-", "_").lower() for n in ds_names]
    ds_reports = []
    for ds_name, ds_key in zip(ds_names, ds_keys):
        if "pinder" in ds_name.lower():
            index_path = None
        else:
            foreign_root = get_pinder_bucket_root() + "/foreign_databases/"
            index_path = foreign_root + f"{ds_key}_mapped_index.parquet"
        _, _, _, ecod_report = annotation_check.binding_leakage_main(
            index_file=index_path
        )
        pindex, metadata, pfam_data = pfam_diversity.load_data(
            index_file=index_path, metadata_file=None, pfam_file=None
        )
        all_splits = ["train", "test", "val", "invalid"]
        pindex_all_splits = pindex[
            (pindex["split"].isin(all_splits))
            & ~(pindex["cluster_id"].str.contains("-1", regex=False))
            & ~(pindex["cluster_id"].str.contains("p_p", regex=False))
        ]
        pindex_with_RL = pfam_diversity.get_ecod_annotations(
            pindex_all_splits,
            metadata,
            frac_interface_threshold=0.25,
            min_intersection=10,
        )
        pindex_pfam = pfam_diversity.process_pfam_data(pindex_with_RL, pfam_data)
        ecod_diversity = pfam_diversity.report_ecod_diversity(pindex_pfam)
        pf_diversity = pfam_diversity.get_pfam_diversity(pindex_pfam)
        merged_metadata = pfam_diversity.get_merged_metadata(pindex_pfam, metadata)
        cluster_diversity = pfam_diversity.get_cluster_diversity(merged_metadata)
        uniprot_leaks = uniprot_leakage.report_uniprot_test_val_leakage(
            index_path=index_path
        )
        report = pd.concat(
            [
                ecod_report,
                ecod_diversity,
                cluster_diversity,
                pf_diversity,
                uniprot_leaks,
            ],
            ignore_index=True,
        )
        report.loc[:, "Dataset"] = ds_name
        ds_reports.append(report)
    report = pd.concat(ds_reports, ignore_index=True)
    ialign_report = get_ialign_leakage()
    report = pd.concat([report, ialign_report], ignore_index=True)
    if not output_pqt.parent.is_dir():
        output_pqt.parent.mkdir(parents=True)
    report.to_parquet(output_pqt, index=False)
    return report


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
    figsize: tuple[int, int] = (13, 10),
    plot_width: int = 1600,
    plot_height: int = 650,
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    report = get_summary_diversity_and_leakage_stats(fig_dir=fig_dir)
    report = report.melt(
        id_vars=["Measure", "Metric", "Dataset"], value_vars=["Test", "Val"]
    ).rename(columns={"variable": "Split", "value": "Percentage"})
    fig_subdir = fig_dir / "dataset_comparison"
    images = []
    for measure, df in report.groupby("Measure"):
        fig = multi_level_leakage_diversity_bar_plots(
            df,
            y_title=f"{measure} (%)",
            width=plot_width,
            height=plot_height,
            theme=theme,
        )
        png_file = fig_subdir / f"{measure.lower()}_comparison.png"
        futils.write_fig(fig, png_file)
        images.append(png_file)

    image_grid(
        images,
        per_row=1,
        top=None,
        hspace=None,
        right=None,
        wspace=None,
        figsize=figsize,
        img_width=plot_width,
        img_height=plot_height,
        output_png=fig_subdir / f"leakage-diversity-comparison.{format}",
    )
    ialign_vs_idist_density(fig_dir=fig_dir, format=format, theme=theme)


if __name__ == "__main__":
    main()
