from __future__ import annotations

from pathlib import Path

import pandas as pd

from plotly.graph_objs._figure import Figure
from tqdm import tqdm

from pinder.core import get_index, get_metadata, get_pinder_location
from pinder.core.index.utils import setup_logger
from pinder.data.plot import constants as pc, figure_utils as futils
from pinder.data.plot import Colors, DarkTheme, LightTheme, Theme, ViolinPlot
from pinder.data.plot.image import image_grid

log = setup_logger(__name__)

pindex = get_index()
meta = get_metadata()


def get_annotation_metadata() -> pd.DataFrame:
    annotation_meta = pd.merge(pindex, meta, how="left")
    # Break into train/val/xl/s/af2
    annotation_meta["split"] = annotation_meta["split"].astype("object")
    nontest = annotation_meta.query('split != "test"').reset_index(drop=True)
    test = annotation_meta.query('split == "test"').reset_index(drop=True)
    test_xl = test.query("pinder_xl").reset_index(drop=True)
    test_s = test.query("pinder_s").reset_index(drop=True)
    test_af = test.query("pinder_af2").reset_index(drop=True)
    test_af.loc[:, "split"] = "PINDER-AF2"
    test_xl.loc[:, "split"] = "PINDER-XL"
    test_s.loc[:, "split"] = "PINDER-S"

    test = pd.concat([test_xl, test_s, test_af]).reset_index(drop=True)
    annotation_meta = pd.concat([nontest, test]).reset_index(drop=True)

    annotation_meta.loc[annotation_meta.split == "val", "split"] = "Val"
    annotation_meta.loc[annotation_meta.split == "train", "split"] = "Train"
    annotation_meta.loc[annotation_meta.split == "invalid", "split"] = "Invalid"
    return annotation_meta


def split_annotation_violins(
    annotation_df: pd.DataFrame,
    metric: str,
    theme: Theme = LightTheme(),
    span: list[int] | None = None,
    width: int = 950,
    height: int = 600,
    showlegend: bool = False,
) -> Figure:
    vp = ViolinPlot(theme=theme)
    fig = vp.violinplot(
        annotation_df,
        x="split",
        y=metric,
        color="split",
        color_discrete_map=pc.split_color_map,
        width=width,
        box=True,
        category_orders={
            "split": list(pc.split_color_map.keys()),
        },
        height=height,
        span=span,
        hide_xaxis_title=True,
        grid_y=True,
        grid_x=False,
        custom_yaxis_title=pc.LABELS.get(metric, metric),
        hide_legend_title=True,
        marker_line_width=0.1,
        marker_size=8,
        marker_opacity=0.9,
        show_legend=showlegend,
        shared_yaxis_title=pc.LABELS.get(metric, metric),
        shared_yaxis_x_loc=-0.14,
    )
    top_margin = 60 if len(pc.LABELS.get(metric, metric)) > 30 else 30
    fig = futils.update_layout(fig, margin=dict(t=top_margin, b=0, r=30, l=140, pad=0))
    return fig


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    fig_subdir = fig_dir / "annotation-split-distributions"
    annotation_meta = get_annotation_metadata()
    images = []
    for metric in tqdm(
        pc.prop_plot_cols,
        desc="Generating plots per annotation...",
        total=len(pc.prop_plot_cols),
    ):
        png_file = fig_subdir / f"{metric}-splits-violin.png"
        clip_axes = {
            "resolution": 10,
            "length1": 2000,
            "length2": 2000,
            "buried_sasa": 20_000,
            "intermolecular_contacts": 500,
            "charged_charged_contacts": 75,
            "charged_polar_contacts": 75,
            "charged_apolar_contacts": 100,
            "polar_polar_contacts": 50,
            "apolar_apolar_contacts": 100,
            "apolar_polar_contacts": 100,
            "missing_interface_residues_8A": 10,
        }
        if metric in clip_axes:
            annotation_meta.loc[annotation_meta[metric] > clip_axes[metric], metric] = (
                clip_axes[metric]
            )
        fig = split_annotation_violins(annotation_meta, metric, theme=theme)
        futils.write_fig(fig, png_file)
        images.append(png_file)

    log.info("Generating figure grid...")
    image_grid(
        images,
        per_row=5,
        top=None,  # 0.99,
        hspace=None,  # 0.01,
        right=None,  # 0.99,
        wspace=None,  # 0.01,
        figsize=(18, 10),
        img_width=int(950 * 1.5),
        img_height=int(600 * 1.5),
        output_png=fig_subdir / f"annotation-split-distributions-grid.{format}",
    )


if __name__ == "__main__":
    main()
