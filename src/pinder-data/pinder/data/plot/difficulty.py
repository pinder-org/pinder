from __future__ import annotations
from pathlib import Path

import pandas as pd
from plotly.graph_objs._figure import Figure
from tqdm import tqdm

from pinder.core import get_index, get_pinder_location, PinderSystem
from pinder.core.utils.process import process_map
from pinder.data.plot import (
    Theme,
    DarkTheme,
    LightTheme,
    BarPlot,
    constants as pc,
    figure_utils as futils,
)


pindex = get_index()


def get_diff(pid: str) -> pd.DataFrame:
    ps = PinderSystem(pid)
    apo = ps.unbound_difficulty("apo")
    pred = ps.unbound_difficulty("predicted")
    diff = pd.DataFrame([apo, pred])
    diff.loc[:, "id"] = pid
    return diff


def get_difficulties(
    split: str = "test",
    parallel: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame:
    split_ids = set(pindex.query(f'split == "{split}"').id)
    diff = process_map(
        get_diff,
        split_ids,
        parallel=parallel,
        max_workers=max_workers,
    )
    diff_df = pd.concat(diff).reset_index(drop=True)
    if split == "test":
        xl_ids = set(pindex.query("pinder_xl").id)
        s_ids = set(pindex.query("pinder_s").id)
        af2_ids = set(pindex.query("pinder_af2").id)
        subsets = [
            (xl_ids, "PINDER-XL"),
            (s_ids, "PINDER-S"),
            (af2_ids, "PINDER-AF2"),
        ]
    else:
        subsets = [(split_ids, split.title())]

    diff_dfs = []
    for ids, label in subsets:
        subset_diff = diff_df[diff_df["id"].isin(ids)].reset_index(drop=True)
        subset_diff.loc[:, "pinder_set"] = label
        diff_dfs.append(subset_diff)
    all_diff = pd.concat(diff_dfs, ignore_index=True)
    split_index = pindex[pindex["id"].isin(split_ids)].reset_index(drop=True)
    apo_presence = split_index[["id", "apo_R", "apo_L"]].copy()
    af_presence = split_index[["id", "predicted_R", "predicted_L"]].copy()
    apo_presence["both_exist"] = apo_presence.apo_R & apo_presence.apo_L
    af_presence["both_exist"] = af_presence.predicted_R & af_presence.predicted_L
    apo_presence.loc[:, "monomer_name"] = "apo"
    af_presence.loc[:, "monomer_name"] = "predicted"
    monomer_presence = pd.concat(
        [
            apo_presence.drop(columns=["apo_R", "apo_L"]),
            af_presence.drop(columns=["predicted_R", "predicted_L"]),
        ],
        ignore_index=True,
    )
    all_diff = pd.merge(all_diff, monomer_presence, how="left")
    return all_diff


def get_paired_difficulties(all_diff: pd.DataFrame) -> pd.DataFrame:
    paired = all_diff.query("both_exist")
    diff_count = paired.groupby(
        ["monomer_name", "pinder_set", "difficulty"], as_index=False
    ).size()
    tot_count = (
        paired.groupby(["monomer_name", "pinder_set"], as_index=False)
        .size()
        .rename(columns={"size": "total"})
    )
    diff_count = diff_count.merge(tot_count)
    diff_count.loc[:, "percent_total"] = (
        diff_count["size"] / diff_count["total"]
    ) * 100
    return diff_count


def dataset_difficulty_barplot(
    diff_count: pd.DataFrame, theme: Theme = LightTheme()
) -> Figure:
    bp = BarPlot(theme=theme)
    # change column names for plotting
    diff_count = diff_count.rename(
        columns={
            "monomer_name": "Monomer",
            "difficulty": "Flexibility",
            "pinder_set": "Pinder Set",
        }
    )
    diff_count.loc[:, "Monomer"] = [
        pc.LABELS.get(mono, mono) for mono in list(diff_count.Monomer)
    ]
    return bp.barplot(
        diff_count,
        x="Monomer",
        y="percent_total",
        color="Flexibility",
        facet_col="Pinder Set",
        color_discrete_map=pc.flexibility_color_map,
        height=650,
        width=1600,
        labels={"percent_total": "Percentage of systems (%)"},
        category_orders={
            "monomer_name_label": ["Holo", "Apo", "Predicted (AF2)"],
            "Pinder Set": pc.PINDER_SETS,
        },
        custom_yaxis_title="Percentage of systems (%)",
        hide_xaxis_title=True,
        text="percent_total",
        facet_col_wrap=4,
        grid_x=False,
        grid_y=False,
    )


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    pinder_dir = get_pinder_location()
    data_dir = pinder_dir / "data/difficulties"
    if not data_dir.is_dir():
        data_dir.mkdir(exist_ok=True, parents=True)

    diff_csv = data_dir / "difficulties.csv"

    if diff_csv.is_file():
        difficulties = pd.read_csv(diff_csv)
    else:
        difficulties = get_difficulties()
        difficulties.to_csv(diff_csv, index=False)

    diff_count = get_paired_difficulties(difficulties)
    fig = dataset_difficulty_barplot(diff_count, theme=theme)
    output_file = fig_dir / f"difficulty/test-subset-difficulty-barplot.{format}"
    futils.write_fig(fig, output_file)


if __name__ == "__main__":
    main()
