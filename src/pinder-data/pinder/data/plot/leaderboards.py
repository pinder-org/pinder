from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from pinder.core import get_index, get_pinder_location
from pinder.core.utils.log import setup_logger
from pinder.data.plot import constants as pc
from pinder.data.plot.performance import get_dockq_metrics, remove_af2mm_nonaf2
from pinder.eval.dockq.method import (
    MethodMetrics,
    add_pinder_set,
    get_expected_counts,
    subsets,
    CapriClass,
)

log = setup_logger(__name__)

pindex = get_index()
expected_counts = get_expected_counts(pindex, subsets)


def get_monomer_counts() -> dict[str, list[str]]:
    mono_counts = {}
    for dataset in ["pinder_xl", "pinder_s", "pinder_af2"]:
        dataset_counts = []
        for monomer in ["holo", "apo", "predicted"]:
            expected = expected_counts[(dataset, monomer)]["count"]
            dataset_counts.append("{:,}".format(expected))
        mono_counts[dataset.replace("2", "")] = dataset_counts
    return mono_counts


def get_leaderboard_table(
    leaderboards: pd.DataFrame,
    metrics: pd.DataFrame,
    monomer: str,
    dataset: str,
    summary_only: bool = True,
) -> pd.DataFrame:
    table = leaderboards.query(
        f"Dataset == '{dataset}' and Monomer == '{monomer}'"
    ).reset_index(drop=True)
    missing_count = []
    for method_name, df in metrics.groupby("method_name"):
        df = df.query(
            f"monomer_name == '{monomer}' and pinder_set == '{dataset}'"
        ).reset_index(drop=True)
        if not df.shape[0]:
            continue
        missing = len(set(df.query('model_name == "missing_decoy_1"').id))
        missing_count.append(
            {
                "Method": method_name,
                "Dataset": dataset,
                "Monomer": monomer,
                "Missing systems": missing,
            }
        )
    missing_count = pd.DataFrame(missing_count)
    table = pd.merge(missing_count, table, how="left")
    if summary_only:
        return table[pc.LEADERBOARD_SUMMARY_COLS]
    return table


def get_leaderboard_dict(metrics: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    mm = MethodMetrics(Path("./"), custom_index="")
    mm._metrics = metrics.copy()
    leaderboard = mm.get_leaderboard_entry()
    leaderboard_dict: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in ["pinder_xl", "pinder_s", "pinder_af2"]:
        subset_key = dataset.split("pinder_")[1].replace("2", "")
        leaderboard_dict[subset_key] = {}
        for monomer in ["holo", "apo", "predicted"]:
            table = get_leaderboard_table(leaderboard, metrics, monomer, dataset)
            if monomer == "predicted":
                monomer = "pred"
            leaderboard_dict[subset_key][monomer] = table
    return leaderboard_dict


def generate_latex_rows(df: pd.DataFrame) -> str:
    metric_max_map = {
        metric: np.argmax(df[metric]) for metric in pc.LEADERBOARD_METRIC_COLS
    }
    row_data = []
    for i, row in df.iterrows():
        row_str = [
            row.Method,
        ]
        for metric in pc.LEADERBOARD_METRIC_COLS:
            # No confidence model
            if "diffdock" in row.Method.lower() and "top" in metric.lower():
                val = "*"
            else:
                val = str(round(row[metric], 2))
            if i == metric_max_map[metric]:
                val = r"\textbf{" + val + "}"
            row_str.append(val)
        row_str.append(str(row["Missing systems"]))
        row_tex = " & ".join(row_str) + r" \\"
        row_data.append(row_tex)
    rows = "\n".join(row_data)
    return rows


def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str = "dockq_metrics_holo_pinder_af2",
) -> str:
    row_data = generate_latex_rows(df)
    tex: str = (
        pc.MONOMER_LEADERBOARD_TEMPLATE.replace("$CAPTION", caption)
        .replace("$ROWS", row_data)
        .replace("$LABEL", label)
    )
    return tex


def generate_latex_rows_all(df: pd.DataFrame) -> str:
    metric_max_map = {
        metric: np.argmax(df[metric]) for metric in pc.LEADERBOARD_METRIC_COLS
    }
    row_data = []
    for i, row in df.iterrows():
        row_str = [
            row.Method,
        ]
        for metric in pc.LEADERBOARD_METRIC_COLS:
            # No confidence model
            if "diffdock" in row.Method.lower() and "top" in metric.lower():
                val = "*"
            else:
                val = str(round(row[metric], 2))
            if i == metric_max_map[metric]:
                val = r"\textbf{" + val + "}"
            row_str.append(val)
        row_str.append(str(row["Missing systems"]))
        row_text = "& " + " & ".join(row_str) + r" \\"
        row_data.append(row_text)

    rows = "\n".join(row_data)
    return rows


def generate_latex_table_all(
    dfs: list[pd.DataFrame],
    subset: str,
    label: str = "dockq_metrics_holo_pinder_af2",
) -> str:
    # Format pinder_xl/s/af2 into latex format command equivalent
    subset_cmd = subset.replace("_", "").replace("2", "")
    n_methods = max([len(set(df.Method)) for df in dfs])
    caption = (
        f"DockQ CAPRI classification evaluation metrics for the \{subset_cmd} test set across {pc.INT2WORD[n_methods]} evaluated docking methods. "
        "The leftmost column shows the input type (holo/apo/predicted) along with the number of evaluated systems. "
        "Methods are ranked alphabetically, results for the highest performing method are highlighted as bold. "
        "The rightmost column shows the number of systems not predicted by the respective method."
    )
    holo_rows = generate_latex_rows_all(dfs[0])
    apo_rows = generate_latex_rows_all(dfs[1])
    pred_rows = generate_latex_rows_all(dfs[2])
    mono_counts = get_monomer_counts()
    subset_counts = mono_counts[subset]
    holo_count, apo_count, pred_count = subset_counts
    tex: str = (
        pc.SUBSET_LEADERBOARD_TEMPLATE.replace("$CAPTION", caption)
        .replace("$ROWS_HOLO", holo_rows)
        .replace("$ROWS_APO", apo_rows)
        .replace("$ROWS_AF2", pred_rows)
        .replace("$HOLO_COUNT", holo_count)
        .replace("$APO_COUNT", apo_count)
        .replace("$PRED_COUNT", pred_count)
        .replace("$DS_HEADING", subset_cmd)
        .replace("$LABEL", label)
        .replace("$N_METHODS", str(n_methods))
    )
    return tex


def generate_per_monomer_leaderboard_tables(
    leaderboard_dict: dict[str, dict[str, pd.DataFrame]],
) -> str:
    mono_labels = {
        "apo": "unbound (apo) structures",
        "holo": "bound (holo) structures",
        "predicted": "structures predicted by AlphaFold 2",
        "pred": "structures predicted by AlphaFold 2",
    }
    table_contents = ""
    for subset, monomer_dict in leaderboard_dict.items():
        for monomer, df in monomer_dict.items():
            df = df[df["Method"].isin(pc.LEADERBOARD_METHODS)].reset_index(drop=True)
            df.loc[:, "Method"] = [pc.LABELS.get(m, m) for m in list(df.Method)]
            num_methods = pc.INT2WORD[len(set(df.Method))]
            table_contents += "\n" * 3
            caption = (
                f"DockQ CAPRI classification evaluation metrics for the \pinder{subset} test set "
                f"across {num_methods} evaluated docking methods using {mono_labels[monomer]} as input. "
                "Methods are ranked alphabetically, results for the highest performing method are highlighted as bold."
            )
            table_str = generate_latex_table(
                df,
                caption=caption,
                label=f"dockq_metrics_{monomer}_pinder_{subset}",
            )
            table_contents += table_str
    return table_contents


def generate_per_subset_leaderboard_tables(
    leaderboard_dict: dict[str, dict[str, pd.DataFrame]],
) -> str:
    table_contents = ""
    for subset, monomer_dict in leaderboard_dict.items():
        df_list = []
        for df in monomer_dict.values():
            df = df[df["Method"].isin(pc.LEADERBOARD_METHODS)].reset_index(drop=True)
            df.loc[:, "Method"] = [pc.LABELS.get(m, m) for m in list(df.Method)]
            df_list.append(df.copy())
        table_contents += "\n" * 3
        table_str = generate_latex_table_all(
            df_list,
            subset=f"pinder_{subset}",
            label=f"dockq_metrics_pinder_{subset}_all",
        )
        table_contents += table_str
    return table_contents


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    theme_name: str = "light",
) -> None:
    log.info("Fetching metrics...")
    table_dir = fig_dir / "leaderboard_tables"
    if not table_dir.is_dir():
        table_dir.mkdir(parents=True)
    cc = CapriClass()
    metrics = get_dockq_metrics()
    metrics.loc[:, "CAPRI_rank"] = metrics.CAPRI.apply(lambda x: cc[x])
    metrics.loc[:, "decoy"] = metrics.model_name + ".pdb"
    # BiotiteDockQ columns that are superceded by MethodMetrics
    metrics.drop(["system", "method"], axis=1, inplace=True, errors="ignore")
    log.info("Adding pinder subset...")
    metrics_penalty = add_pinder_set(metrics, allow_missing=False, custom_index="")
    metrics_penalty = remove_af2mm_nonaf2(metrics_penalty)
    log.info("Generating leaderboard dict...")
    leaderboard_dict = get_leaderboard_dict(metrics_penalty)
    tables = generate_per_subset_leaderboard_tables(leaderboard_dict)
    output_file = table_dir / "per_subset_leaderboard_tables.tex"
    with output_file.open(mode="w") as f:
        f.write(tables)

    tables = generate_per_monomer_leaderboard_tables(leaderboard_dict)
    output_file = table_dir / "per_monomer_leaderboard_tables.tex"
    with output_file.open(mode="w") as f:
        f.write(tables)


if __name__ == "__main__":
    main()
