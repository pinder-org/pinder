from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from plotly.graph_objs._figure import Figure
from tqdm import tqdm

from pinder.core import get_index, get_metadata, get_pinder_location
from pinder.core.utils.log import setup_logger
from pinder.core.utils.retry import exponential_retry
from pinder.data.plot import constants as pc, figure_utils as futils
from pinder.data.plot import BarPlot, DarkTheme, LightTheme, Theme
from pinder.data.plot.image import image_grid
from pinder.data.qc import pfam_diversity as pf


log = setup_logger(__name__)
pindex = get_index()
meta = get_metadata()


def copy_pinder_set(metadata: pd.DataFrame) -> pd.DataFrame:
    meta_sets = []
    for subset in ["pinder_xl", "pinder_s", "pinder_af2"]:
        df = metadata.query(subset).reset_index(drop=True)
        df.loc[:, "pinder_set"] = pc.LABELS.get(subset)
        meta_sets.append(df)

    for split in ["train", "val", "invalid"]:
        df = metadata.query(f'split == "{split}"').reset_index(drop=True)
        df.loc[:, "pinder_set"] = pc.LABELS.get(split)
        meta_sets.append(df)
    metadata = pd.concat(meta_sets, ignore_index=True)
    return metadata


def get_pfam_metadata(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    frac_interface_threshold: float = 0.25,
    min_intersection: int = 10,
) -> pd.DataFrame:
    data_dir = fig_dir / "data"
    output_pqt = data_dir / "pfam_metadata.parquet"
    if output_pqt.is_file():
        pfam_meta = pd.read_parquet(output_pqt)
        return pfam_meta
    pindex, metadata, pfam_data = pf.load_data()
    pindex_with_RL = pf.get_ecod_annotations(
        pindex,
        metadata,
        frac_interface_threshold=frac_interface_threshold,
        min_intersection=min_intersection,
    )
    pindex_pfam = pf.process_pfam_data(pindex_with_RL, pfam_data)
    pindex_pfam.loc[:, "pfam_clan"] = [
        " : ".join(
            sorted(
                [clan for clan in [R, L] if isinstance(clan, str) and clan != "no_clan"]
            )
        )
        for R, L in zip(pindex_pfam.pfam_clan_R, pindex_pfam.pfam_clan_L)
    ]
    pindex_pfam.loc[pindex_pfam.pfam_clan == "", "pfam_clan"] = "Not defined"
    pfam_meta = copy_pinder_set(pfam_meta)
    if not data_dir.is_dir():
        data_dir.mkdir(exist_ok=True, parents=True)
    pfam_meta.to_parquet(output_pqt, index=False, engine="pyarrow")
    return pfam_meta


@exponential_retry(max_retries=10, exceptions=(requests.exceptions.HTTPError,))
def get_panther_id(uniprot_id: str) -> dict[str, str]:
    base_url = "https://rest.uniprot.org/uniprotkb/"
    url = base_url + uniprot_id + ".txt"
    failed_response: dict[str, str] = {"uniprot": uniprot_id, "panther_id": ""}
    panther_info: dict[str, str]
    try:
        r = requests.get(url).text
    except ConnectionError:
        return failed_response
    for line in r.split("\n"):
        if line.startswith("DR   PANTHER;"):
            panther_id = line.split(";")[1].strip()
            if ":SF" in panther_id:
                continue
            else:
                if (line.split(";")[2].strip() == "-") or (
                    line.split(";")[2].strip() == "UNCHARACTERIZED"
                ):
                    panther_info = {
                        "uniprot": uniprot_id,
                        "panther_id": line.split(";")[1].strip(),
                    }
                    return panther_info
                panther_info = {
                    "uniprot": uniprot_id,
                    "panther_id": line.split(";")[2].strip(),
                }
                return panther_info
    return failed_response


def fetch_panther_ids(pinder_dir: Path = get_pinder_location()) -> pd.DataFrame:
    output_pqt = pinder_dir / "panther_ids.parquet"
    if output_pqt.is_file():
        panther_df = pd.read_parquet(output_pqt)
        return panther_df

    all_uniprots = set(pindex.uniprot_R).union(set(pindex.uniprot_L)) - {"UNDEFINED"}
    panther_data = []
    for uni in tqdm(all_uniprots):
        panther_data.append(get_panther_id(uni))
    panther_df = pd.DataFrame(panther_data)
    panther_df.to_parquet(output_pqt, index=False, engine="pyarrow")
    return panther_df


def get_panther_metadata(pinder_dir: Path = get_pinder_location()) -> pd.DataFrame:
    fig_dir = pinder_dir / "publication_figures"
    data_dir = fig_dir / "data"
    output_pqt = data_dir / "panther_class_metadata.parquet"
    if output_pqt.is_file():
        panther_meta = pd.read_parquet(output_pqt)
        return panther_meta
    panther_df = fetch_panther_ids(pinder_dir=pinder_dir)
    for side in ["R", "L"]:
        panther_meta = pd.merge(
            pindex,
            panther_df.rename(
                columns={"uniprot": f"uniprot_{side}", "panther_id": f"panther_{side}"}
            ),
            how="left",
        )
        panther_meta[f"panther_{side}"] = panther_meta[f"panther_{side}"].fillna("")
    class_labs = []
    for R, L in zip(panther_meta.panther_R, panther_meta.panther_L):
        id_list = []
        ids = {R, L}
        for pid in ids:
            if isinstance(pid, str) and pid != "":
                pid = pid.title()
                id_list.append(pid)
        if len(id_list) == 1 and len(id_list[0]) > 60:
            # ID is too long, add a line break between spaces in family ID
            parts = id_list[0].split(" ")
            first_part = parts.pop(0)
            while len(first_part) <= 60 and len(parts):
                first_part += " " + parts.pop(0)
            class_pair = "<br>".join([first_part, " ".join(parts)])
        elif len(" : ".join(sorted(id_list))) > 60:
            # ID is too long, add a line break between second ID
            class_pair = " :<br>".join(sorted(id_list))
        else:
            class_pair = " : ".join(sorted(id_list))
        class_labs.append(class_pair)
    panther_meta.loc[:, "panther_class"] = class_labs
    panther_meta.loc[panther_meta.panther_class == "", "panther_class"] = "Not defined"
    panther_meta = copy_pinder_set(panther_meta)
    if not data_dir.is_dir():
        data_dir.mkdir(exist_ok=True, parents=True)
    panther_meta.to_parquet(output_pqt, index=False, engine="pyarrow")
    return panther_meta


def family_diversity_bar_plot(
    class_size: pd.DataFrame,
    pinder_set: str,
    class_col: str = "pfam_clan",
    top_n: int = 50,
    class_label: str | None = None,
    theme: Theme = LightTheme(),
    height: int = 1800,
    width: int = 1200,
    grid_x: bool = True,
    grid_y: bool = False,
    hide_legend_title: bool = True,
    hide_yaxis_title: bool = True,
    show_legend: bool = False,
    **kwargs: Any,
) -> Figure:
    labels = deepcopy(pc.LABELS)
    if class_label:
        labels[class_col] = class_label

    bp = BarPlot(theme=theme)
    data = (
        class_size.query(
            f'pinder_set == "{pinder_set}" and {class_col} != "Not defined"'
        )
        .sort_values("percent_reps", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .sort_values("percent_reps")
    )
    fig = bp.barplot(
        data,
        x="percent_reps",
        y=class_col,
        color="pinder_set",
        text="percent_reps",
        labels=labels,
        color_discrete_map=pc.split_color_map,
        height=height,
        width=width,
        grid_x=grid_x,
        grid_y=grid_y,
        hide_legend_title=hide_legend_title,
        hide_yaxis_title=hide_yaxis_title,
        show_legend=show_legend,
        **kwargs,
    )
    fig = fig.update_layout(
        title=f"{labels.get(class_col, class_col)} - {pinder_set}",
        margin=dict(
            l=120,
            r=200,
            b=25,
            t=100,
        ),
    )
    return fig


def family_grid_plot(
    family_metadata: pd.DataFrame,
    class_col: str,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    include_invalid: bool = False,
    random_cluster_rep: bool = True,
    top_n: int = 50,
    format: str = "pdf",
    theme: Theme = LightTheme(),
) -> None:
    subdir_name = class_col.replace("_", "-")
    fig_subdir = fig_dir / f"{subdir_name}-diversity"
    if random_cluster_rep:
        cluster_reps = (
            family_metadata.query(f'{class_col} != "Not defined"')
            .sort_values(class_col)
            .drop_duplicates(["cluster_id", "pinder_set"], keep="first")
            .reset_index(drop=True)
        )
    else:
        cluster_reps = family_metadata.copy()

    class_size = cluster_reps.groupby(["pinder_set", class_col], as_index=False).size()
    tot_size = (
        cluster_reps.groupby(["pinder_set"], as_index=False)
        .size()
        .rename({"size": "total_systems"}, axis=1)
    )
    class_size = pd.merge(class_size, tot_size, how="left")
    class_size.loc[:, "percent_reps"] = (
        class_size["size"] / class_size["total_systems"]
    ) * 100

    images = []
    for split in pc.split_color_map:
        if not include_invalid and split == "Invalid":
            continue
        png_file = fig_subdir / f"{split}-{subdir_name}-barplot.png"
        fig = family_diversity_bar_plot(
            class_size, split, class_col, top_n=top_n, theme=theme
        )
        futils.write_fig(fig, png_file)
        images.append(png_file)

    image_grid(
        images,
        per_row=3,
        top=None,  # 0.99,
        hspace=None,  # 0.01,
        right=None,  # 0.99,
        wspace=None,  # 0.01,
        figsize=(18, 18),
        img_width=1100,
        img_height=1800,
        output_png=fig_subdir / f"{subdir_name}-split-diversity-grid.{format}",
    )


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
    include_invalid: bool = False,
    random_cluster_rep: bool = True,
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    log.info("Generating PFAM clan plots...")
    family_grid_plot(
        get_pfam_metadata(),
        class_col="pfam_clan",
        fig_dir=fig_dir,
        include_invalid=include_invalid,
        random_cluster_rep=random_cluster_rep,
        theme=theme,
        format=format,
    )
    log.info("Generating panther class plots...")
    family_grid_plot(
        get_panther_metadata(),
        class_col="panther_class",
        fig_dir=fig_dir,
        include_invalid=include_invalid,
        random_cluster_rep=random_cluster_rep,
        top_n=25,
        theme=theme,
        format=format,
    )


if __name__ == "__main__":
    main()
