from __future__ import annotations
from pathlib import Path

import pandas as pd
from plotly.graph_objs._figure import Figure

from pinder.core import get_index, get_metadata, get_pinder_location
from pinder.core.index.utils import get_pinder_bucket_root, get_supplementary_data
from pinder.core.utils.cloud import gcs_read_dataframe
from pinder.data.plot import constants as pc, figure_utils as futils
from pinder.data.plot import BarPlot, DarkTheme, LightTheme, Theme, ViolinPlot
from pinder.data.plot.image import image_grid


def oligomer_barplot(
    oligomer_counts: pd.DataFrame, theme: Theme = LightTheme()
) -> Figure:
    bp = BarPlot(theme=theme)
    fig = bp.barplot(
        oligomer_counts,
        x="oligomeric_state",
        y="percent_total",
        color="Dataset",
        color_discrete_map=pc.dataset_color_map,
        height=650,
        width=1050,
        labels={"percent_total": "Percentage of systems (%)"},
        category_orders={"Dataset": list(pc.dataset_color_map.keys())},
        custom_yaxis_title="Percentage of systems (%)",
        hide_xaxis_title=True,
        text="oligomer_count",
        grid_x=False,
        grid_y=True,
        show_legend=True,
        hide_legend_title=True,
        text_template="%{text}",
    )
    legend = dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xref="paper",
        x=0.01,
        font=dict(size=32),
    )
    fig = fig.update_layout(legend=legend)
    return fig


def get_dataset_indices(split: str | None = "test") -> pd.DataFrame:
    ds_names = list(pc.dataset_color_map.keys())
    ds_keys = [n.replace("-", "_").lower() for n in ds_names]

    dataset_indices = []
    for ds_name, ds_key in zip(ds_names, ds_keys):
        if "pinder" in ds_name.lower():
            df = get_index()
        else:
            foreign_root = get_pinder_bucket_root() + "/foreign_databases/"
            df = gcs_read_dataframe(foreign_root + f"{ds_key}_mapped_index.parquet")
        df.loc[:, "split_type"] = ds_key
        df.loc[:, "Dataset"] = ds_name
        if split:
            df = df.query(f'split == "{split}"').reset_index(drop=True)
        dataset_indices.append(df)
    dataset_indices = pd.concat(dataset_indices, ignore_index=True)
    return dataset_indices


def get_sequence_based_oligomer_state() -> pd.DataFrame:
    entities = get_supplementary_data("entity_metadata")
    seq_counts = (
        entities.drop_duplicates(["entry_id", "sequence"])
        .groupby("entry_id", as_index=False, observed=True)
        .size()
        .sort_values("size")
        .rename(columns={"size": "unique_seqs", "entry_id": "pdb_id"})
    )
    asym_counts = (
        entities.groupby("entry_id", as_index=False, observed=True)
        .size()
        .rename(columns={"size": "n_chains", "entry_id": "pdb_id"})
    )
    seq_counts = pd.merge(seq_counts, asym_counts, how="left")
    seq_counts.loc[
        (seq_counts.unique_seqs == 1) & (seq_counts.n_chains == 2), "oligomeric_state"
    ] = "Homodimer"
    seq_counts.loc[
        (seq_counts.unique_seqs == 2) & (seq_counts.n_chains == 2), "oligomeric_state"
    ] = "Heterodimer"
    seq_counts.loc[seq_counts.n_chains > 2, "oligomeric_state"] = "Oligomer"
    return seq_counts


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    vp = ViolinPlot(theme=theme)
    meta = get_metadata()
    dataset_indices = (
        get_dataset_indices().query('Dataset != "Sequence"').reset_index(drop=True)
    )
    dataset_indices = pd.merge(
        dataset_indices,
        meta[
            [
                "id",
                "planarity",
                "probability",
                "intermolecular_contacts",
                "oligomeric_count",
                "resolution",
            ]
        ],
        how="left",
    )
    seq_counts = get_sequence_based_oligomer_state()
    dataset_indices = pd.merge(
        dataset_indices, seq_counts[["pdb_id", "oligomeric_state"]], how="left"
    )
    total_counts = (
        dataset_indices.drop_duplicates(["Dataset", "pdb_id"])
        .groupby("Dataset", as_index=False)
        .size()
        .rename(columns={"size": "total"})
    )
    oligomer_counts = (
        dataset_indices.drop_duplicates(["Dataset", "pdb_id"])
        .groupby(["Dataset", "oligomeric_state"], as_index=False)
        .size()
        .rename(columns={"size": "oligomer_count"})
    ).merge(total_counts)
    oligomer_counts.loc[:, "percent_total"] = (
        oligomer_counts["oligomer_count"] / oligomer_counts["total"]
    ) * 100
    fig_subdir = fig_dir / "dataset_comparison"
    images = [fig_subdir / "datasets-oligomer-barplot.png"]
    fig = oligomer_barplot(oligomer_counts, theme=theme)
    futils.write_fig(fig, images[0])
    plot_subsets = [
        (dataset_indices, "resolution"),
        (dataset_indices, "intermolecular_contacts"),
        (dataset_indices, "probability"),
        (dataset_indices, "planarity"),
    ]
    for data, metric in plot_subsets:
        kwargs = dict(
            data=data,
            x="Dataset",
            y=metric,
            color="Dataset",
            color_discrete_map=pc.dataset_color_map,
            category_orders={"Dataset": list(pc.dataset_color_map.keys())},
            width=1050,
            height=650,
            labels=pc.LABELS,
            show_legend=False,
            hide_xaxis_title=True,
            grid_x=False,
        )
        if metric == "probability":
            kwargs["span"] = [-0.1, 1.1]
        fig = vp.violinplot(**kwargs)
        png_file = fig_subdir / f"datasets-{metric}-violinplot.png"
        futils.write_fig(fig, png_file)
        images.append(png_file)

    image_grid(
        images,
        per_row=3,
        top=0.99,
        hspace=0.01,
        right=0.99,
        wspace=0.01,
        output_png=fig_subdir / f"datasets-properties-grid.{format}",
    )


if __name__ == "__main__":
    main()
