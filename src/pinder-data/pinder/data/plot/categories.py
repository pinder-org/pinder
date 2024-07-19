from __future__ import annotations
from pathlib import Path

import pandas as pd
from plotly.graph_objs._figure import Figure


from pinder.core import get_index, get_metadata, get_pinder_location
from pinder.core.index.utils import get_supplementary_data
from pinder.core.utils.log import setup_logger
from pinder.data.plot import (
    BarPlot,
    Theme,
    DarkTheme,
    LightTheme,
    Colors,
    constants as pc,
    figure_utils as futils,
)
from pinder.data.plot.difficulty import get_difficulties
from pinder.data.plot.performance import (
    select_top_n,
    get_penalized_dockq,
)


log = setup_logger(__name__)

pindex = get_index()
meta = get_metadata()


def annotation_hit_rate_barplot(
    df: pd.DataFrame,
    bin_col: str,
    facet_col: str = "monomer_name",
    theme: Theme = LightTheme(),
) -> Figure:
    df.loc[:, "monomer_name"] = [pc.LABELS.get(mn, mn) for mn in list(df.monomer_name)]
    ordered_colors = [Colors.green, Colors.blue, Colors.pink]
    annotation_color_map = {}
    for i, bin_label in enumerate(
        list(
            df.sort_values("category_order").drop_duplicates("category_order")[bin_col]
        )
    ):
        annotation_color_map[bin_label] = ordered_colors[i]

    annotation_color_map.update(pc.category_color_map)
    bp = BarPlot(theme=theme)
    cat_orders = {
        "monomer_name": [
            monomer
            for monomer in ["Holo", "Apo", "Predicted (AF2)"]
            if monomer in set(df.monomer_name)
        ],
        "method_name": [
            mn for mn in list(pc.METHOD_LABELS.values()) if mn in set(df.method_name)
        ],
        bin_col: list(
            df.sort_values("category_order").drop_duplicates("category_order")[bin_col]
        ),
        "Neff": ["High", "Medium", "Low"],
        "novelty": ["None", "Single", "Both"],
        "top_n": ["Top 1", "Top 5", "Oracle"],
    }

    fig = bp.barplot(
        data=df,
        x="hit_rate",
        y="method_name",
        color=bin_col,
        color_discrete_map=annotation_color_map,
        category_orders=cat_orders,
        height=1200,
        width=1600 if bin_col != "difficulty" else 1400,
        text="hit_rate",
        facet_col=facet_col,
        barmode="group",
        custom_xaxis_title="CAPRI hit rate (%)",
        shared_xaxis_title=True,
        hide_yaxis_title=True,
        hide_legend_title=True,
        grid_x=True,
        grid_y=False,
        shared_xaxis_y_loc=-0.075,
    )
    fig = fig.update_traces(textangle=0)
    if "_bin" in bin_col:
        fig.layout.legend.tracegroupgap = 50
    return fig


def get_categorical_annotation_hit_rate(
    dockq_annotation: pd.DataFrame, annotation: str
) -> pd.DataFrame:
    oracle_metrics = select_top_n(dockq_annotation, "DockQ", n=1, ascending=False)
    top_5_oracle_metrics = select_top_n(dockq_annotation, "rank", n=5)
    top_1_metrics = select_top_n(dockq_annotation, "rank", n=1)
    rank_metrics = []
    for lab, df in zip(
        ["Oracle", "Top 1", "Top 5"],
        [oracle_metrics, top_1_metrics, top_5_oracle_metrics],
    ):
        df.loc[:, "top_n"] = lab
        df.loc[:, "method_name"] = [pc.LABELS.get(mn, mn) for mn in df.method_name]
        df["CAPRI_hit"] = df.CAPRI != "Incorrect"
        rank_metrics.append(df)
    combined_metrics = pd.concat(rank_metrics, ignore_index=True)
    # No confidence model trained for DiffDock-PP, remove Top1/5 ranked values
    combined_metrics = combined_metrics[
        ~(
            (combined_metrics.method_name == "DiffDock-PP")
            & (combined_metrics.top_n.isin(["Top 1", "Top 5"]))
        )
    ].reset_index(drop=True)
    total_systems = (
        combined_metrics.groupby(
            ["method_name", "monomer_name", annotation, "pinder_set", "top_n"],
            as_index=False,
        )
        .size()
        .rename({"size": "total_systems"}, axis=1)
    )
    hit_rate = combined_metrics.groupby(
        ["method_name", "monomer_name", annotation, "pinder_set", "top_n"],
        as_index=False,
    ).agg({"CAPRI_hit": "sum"})
    hit_rate = hit_rate.merge(total_systems)
    hit_rate.loc[:, "hit_rate"] = (hit_rate.CAPRI_hit / hit_rate.total_systems) * 100
    return hit_rate


def bin_annotation(
    dockq_annotation: pd.DataFrame, annotation: str, plot_label: str, digits: int = 0
) -> pd.DataFrame:
    test_annot = dockq_annotation.drop_duplicates("id").reset_index(drop=True)
    bin_col = f"{annotation}_bin"
    bin_labels, bins = pd.qcut(
        test_annot[annotation].astype(float), 3, labels=False, retbins=True
    )
    test_annot.loc[:, bin_col] = bin_labels

    bin_map = {i: round(cut, digits) for i, cut in enumerate(bins[1:])}
    bin_map_labels = {}
    for i, v in bin_map.items():
        val = round(v, digits)
        if i > 0:
            prev_val = round(bin_map[i - 1], digits)
        else:
            prev_val = 0.0

        if digits == 0:
            val = int(val)
            prev_val = int(prev_val)
        if i == 0:
            bin_map_labels[i] = f"{plot_label} < {val}"
        elif i == 1:
            bin_map_labels[i] = f"{prev_val} ≤ {plot_label} < {val}"
        else:
            bin_map_labels[i] = f"{plot_label} > {prev_val}"

    binned = pd.merge(dockq_annotation, test_annot[["id", bin_col]], how="left")
    bin_hitrate = get_categorical_annotation_hit_rate(binned, bin_col)
    bin_hitrate.loc[:, "category_order"] = bin_hitrate[bin_col]
    bin_col_vals = [bin_map_labels[bin] for bin in list(bin_hitrate[bin_col])]
    bin_hitrate[bin_col] = bin_hitrate[bin_col].astype("object").astype("str")
    bin_hitrate.loc[:, bin_col] = bin_col_vals
    return bin_hitrate


def success_rate_by_neff(
    dockq: pd.DataFrame,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme: Theme = LightTheme(),
) -> None:
    paired_neff = get_supplementary_data("paired_neff")
    dockq_neff = pd.merge(dockq, paired_neff, how="left")
    neff_hit_rate = bin_annotation(dockq_neff, "neff", "Neff")
    fig_subdir = fig_dir / "neff-performance"
    for pinder_set in pc.PINDER_SETS:
        fig = annotation_hit_rate_barplot(
            neff_hit_rate.query(
                f'pinder_set == "{pinder_set}" and monomer_name == "holo"'
            ).reset_index(drop=True),
            "neff_bin",
            facet_col="top_n",
            theme=theme,
        )
        futils.write_fig(fig, fig_subdir / f"hitrate--Neff--holo-{pinder_set}.{format}")


def success_rate_by_flexibility(
    dockq: pd.DataFrame,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme: Theme = LightTheme(),
) -> None:
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

    difficulties = difficulties.query("both_exist").reset_index(drop=True)
    # merge metrics with difficulty using system (metrics) and id (difficulty)
    metrics_difficulty = pd.merge(
        dockq.query('monomer_name != "holo"').reset_index(drop=True),
        difficulties[["id", "monomer_name", "difficulty", "pinder_set"]],
    )
    metrics_difficulty.loc[
        metrics_difficulty.difficulty == "Difficult", "difficulty"
    ] = "Flexible"
    hit_rate = get_categorical_annotation_hit_rate(metrics_difficulty, "difficulty")
    cat_map = {"Rigid-body": 0, "Medium": 1, "Flexible": 2}
    hit_rate.loc[:, "category_order"] = [cat_map[d] for d in hit_rate.difficulty]
    fig_dir = get_pinder_location() / "publication_figures"
    fig_subdir = fig_dir / "difficulty-performance"
    for pinder_set in pc.PINDER_SETS:
        for top_n in pc.rank_color_map.keys():
            fig = annotation_hit_rate_barplot(
                hit_rate.query(
                    f'pinder_set == "{pinder_set}" and top_n == "{top_n}"'
                ).reset_index(drop=True),
                "difficulty",
                theme=theme,
            )
            filename = f"hitrate-{top_n.replace(' ', '_').lower()}--diff--{pinder_set}.{format}"
            futils.write_fig(fig, fig_subdir / filename)


def success_rate_by_annotation(
    dockq: pd.DataFrame,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme: Theme = LightTheme(),
) -> None:
    plot_annotations = {
        "buried_sasa": "ΔSASA",
        "intermolecular_contacts": "Contacts",
        "n_residues": "Residues",
        "resolution": "Resolution (Å)",
    }
    dockq_annotation = pd.merge(
        dockq, meta[["id"] + list(plot_annotations.keys())], how="left"
    )
    fig_subdir = fig_dir / "annotation_hitrates"
    for annot, lab in plot_annotations.items():
        subdir = fig_subdir / f"{annot}-performance"
        digits = int(annot == "resolution")
        annot_hitrate = bin_annotation(dockq_annotation, annot, lab, digits=digits)
        for pinder_set in pc.PINDER_SETS:
            for top_n in pc.rank_color_map.keys():
                fig = annotation_hit_rate_barplot(
                    annot_hitrate.query(
                        f'pinder_set == "{pinder_set}" and top_n == "{top_n}"'
                    ).reset_index(drop=True),
                    f"{annot}_bin",
                    theme=theme,
                )
                futils.write_fig(
                    fig, subdir / f"hitrate--{annot}--{pinder_set}--{top_n}.{format}"
                )


def success_rate_by_chain_novelty(
    dockq: pd.DataFrame,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme: Theme = LightTheme(),
) -> None:
    test = pindex.query('split == "test"').reset_index(drop=True)
    R_counts = (
        pindex.query('split == "train"')
        .groupby("cluster_id_R", as_index=False, observed=True)
        .size()
        .rename({"cluster_id_R": "cluster_id"}, axis=1)
    )
    L_counts = (
        pindex.query('split == "train"')
        .groupby("cluster_id_L", as_index=False, observed=True)
        .size()
        .rename({"cluster_id_L": "cluster_id"}, axis=1)
    )
    cluster_counts = pd.concat([R_counts, L_counts], ignore_index=True)
    cluster_counts = cluster_counts.groupby(
        "cluster_id", as_index=False, observed=True
    ).agg({"size": "sum"})
    cluster_counts = {
        rec["cluster_id"]: rec["size"]
        for rec in cluster_counts.to_dict(orient="records")
    }
    matched = []
    for pid, cr, cl in zip(test["id"], test["cluster_id_R"], test["cluster_id_L"]):
        r = cluster_counts.get(cr, 0)
        l = cluster_counts.get(cl, 0)
        if (l == 0) and (r == 0):
            novelty_lab = "Both"
        elif l == 0 or r == 0:
            novelty_lab = "Single"
        else:
            novelty_lab = "Neither"
        matched.append(
            {
                "id": pid,
                "R_match": r,
                "L_match": l,
                "novelty": novelty_lab,
            }
        )
    matched_chains = pd.DataFrame(matched)
    dockq_novelty = pd.merge(
        dockq,
        matched_chains[["id", "novelty"]].drop_duplicates().reset_index(drop=True),
        how="left",
    )
    # novelty_violins(dockq_novelty, pinder_set = "PINDER-XL")
    novelty_hitrate = get_categorical_annotation_hit_rate(dockq_novelty, "novelty")

    cat_map = {"Neither": 0, "Single": 1, "Both": 2}
    novelty_hitrate.loc[:, "category_order"] = [
        cat_map[d] for d in novelty_hitrate.novelty
    ]

    fig_subdir = fig_dir / "chain-novelty-performance"
    for pinder_set in pc.PINDER_SETS:
        fig = annotation_hit_rate_barplot(
            novelty_hitrate.query(
                f'pinder_set == "{pinder_set}" and monomer_name == "holo"'
            ).reset_index(drop=True),
            "novelty",
            facet_col="top_n",
            theme=theme,
        )
        futils.write_fig(
            fig, fig_subdir / f"hitrate--holo--novelty--{pinder_set}.{format}"
        )


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
) -> None:
    theme = LightTheme() if theme_name.lower() == "light" else DarkTheme()
    dockq = get_penalized_dockq()
    dockq = dockq.query(
        'method_name != "diffdockpp_dips_holo_train1" and method_name != "diffdockpp_seq_subset1_train1"'
    ).reset_index(drop=True)
    plot_methods = [
        success_rate_by_annotation,
        success_rate_by_flexibility,
        success_rate_by_chain_novelty,
        success_rate_by_neff,
    ]
    for plot_method in plot_methods:
        plot_method(dockq, fig_dir=fig_dir, format=format, theme=theme)


if __name__ == "__main__":
    main()
