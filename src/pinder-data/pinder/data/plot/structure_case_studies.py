from __future__ import annotations
import os
from pathlib import Path

import colorsys
import pandas as pd
import plotly.express as px
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


from pinder.core import get_pinder_location
from pinder.data.plot import constants as pc
from pinder.data.plot.image import image_grid
from pinder.data.plot.performance import get_dockq_metrics

pinder_root = get_pinder_location()


def get_oracle_per_id_monomer_method(dockq: pd.DataFrame) -> pd.DataFrame:
    oracle = (
        dockq.sort_values("DockQ", ascending=False)
        .drop_duplicates(["monomer_name", "method_name", "id"], keep="first")
        .reset_index(drop=True)
    )
    oracle.loc[:, "id_monomer_name"] = oracle.id + "--" + oracle.monomer_name
    oracle.loc[:, "pdb_id"] = [id.split("__")[0] for id in list(oracle.id)]
    oracle.loc[:, "pdb_id_monomer"] = oracle.pdb_id + "--" + oracle.monomer_name
    return oracle


def get_case_subset(oracle_data: pd.DataFrame, id_monomers: list[str]) -> pd.DataFrame:
    case_subset = (
        oracle_data.query(f"id_monomer_name in {id_monomers}")
        .sort_values(["id", "monomer_name"])
        .reset_index(drop=True)[
            [
                "id",
                "monomer_name",
                "method_name",
                "DockQ",
                "eval_dir",
                "decoy",
                "receptor_chain",
                "ligand_chain",
                "pdb_id",
                "pdb_id_monomer",
            ]
        ]
    )
    return case_subset


def clamp_rgb_value(value: float) -> float:
    """Clamp the RGB values to ensure they are within the valid range [0, 1]."""
    return min(1.0, max(0.0, value))


def generate_color_palette(
    n_methods: int = 12,
    lightness_shift: float = 0.2,
) -> dict[int, tuple[str, str]]:
    """Generate a color palette with n_methods distinct colors,
    and for each color, generate a lighter/darker shade with better perceptual distinction.

    Args:
    - n_methods (int): Number of methods, defaults to 12.
    - lightness_shift (float): Amount to lighten/darken for the paired color.

    Returns:
    - dict: A dictionary of colors with keys as method numbers and values as hex codes.
    """
    base_colors = []
    paired_colors = []

    # Generate base colors evenly spaced around the color wheel
    for i in range(n_methods):
        hue = i / n_methods
        base_color = colorsys.hsv_to_rgb(
            hue, 1, 0.5
        )  # Saturated, medium lightness color

        # Convert to Lab color space for perceptually uniform adjustments
        rgb_base = sRGBColor(base_color[0], base_color[1], base_color[2])
        lab_base = convert_color(rgb_base, LabColor)

        # Adjust lightness for the paired color
        lab_lighter = LabColor(
            lab_base.lab_l + lightness_shift * 100, lab_base.lab_a, lab_base.lab_b
        )
        rgb_lighter = convert_color(lab_lighter, sRGBColor)

        # Convert to hex and clamp values to valid range
        base_colors.append(
            "#"
            + "".join(
                f"{int(clamp_rgb_value(x)*255):02x}" for x in rgb_base.get_value_tuple()
            )
        )
        paired_colors.append(
            "#"
            + "".join(
                f"{int(clamp_rgb_value(x)*255):02x}"
                for x in rgb_lighter.get_value_tuple()
            )
        )

    palette = {
        i: (c1, c2) for i, (c1, c2) in enumerate(zip(base_colors, paired_colors))
    }
    return palette


def generate_chimerax_native_pred_overlay(
    native: Path,
    models: list[dict[str, Path | str]],
    png_file: Path,
    color_palette: dict[str, tuple[str, str]],
    # label1: str,
    # label2: str,
    native_R_chain: str = "R",
    native_L_chain: str = "L",
    label_color: str = "#797979ff",
    pad: float = -0.6,
) -> str:
    script = f"""
open {native}
select #1/{native_R_chain}
color sel dark gray
select #1/{native_L_chain}
color sel light gray
set bgColor white
graphics silhouettes true
graphics silhouettes width 4
select subtract #1
lighting shadows false
lighting gentle
preset 'overall look' 'publication 2 (depth-cued)'
select #1/R
color sel #8b8b8bff
select #1/L
color sel dark gray
select #1/R
color sel dim gray
select #1/L
color sel light gray
select subtract #1
select subtract #2
hide target p
lighting flat
lighting soft
graphics silhouettes true
graphics silhouettes width 5
"""

    model_start = 2
    for i, model in enumerate(models):
        R_color, L_color = color_palette[str(model["method_name"])]
        R_ch = model["R_chain"]
        L_ch = model["L_chain"]
        model_idx = model_start + i
        script += f"""open {model['path']}
        select #{model_idx}/{R_ch}
        color sel {R_color}
        sel #{model_idx}/{L_ch}
        color sel {L_color}
        select subtract #{model_idx}
        mmaker #{model_idx}/{R_ch} to #1/{native_R_chain}
        """

    script += f"""
    select #1-{len(models) + 1}
    view sel orient pad {pad}
    select subtract #1-{len(models) + 1}
    """

    model_start = 2
    label_idx = len(models) + 2
    label_count = 0
    for i, model in enumerate(models):
        rest = ",".join([str(model_start + j) for j, m in enumerate(models) if j != i])
        model_png_file = model["png_file"]
        model_idx = model_start + i
        label1 = model["method_name"]
        label2 = model["label"]
        label_count += 2
        script += f"""
        show #!{model_idx} models
        hide #!{rest} models
        hide #!1 models
        hide target p
        windowsize width 2432 height 1462
        hide atoms
        save {model_png_file} width 2432 height 1462 supersample 4 transparentBackground true
        """
        # 2dlab text '{label1}' color {label_color} size 48 x .03 y .92
        # 2dlab text '{label2}' color {label_color} size 48 x .03 y .88
        # hide #{label_idx}.{label_count-1}
        # hide #{label_idx}.{label_count}

    pdb_id, method_n, mono_name = Path(model_png_file).stem.split("--")
    native_png = f"{pdb_id}--native--{mono_name}.png"
    native_png_file = Path(model_png_file).parent / native_png
    rest = ",".join([str(k) for k in list(range(model_start, len(models) + 2))])
    script += f"""
    hide #!{rest} models
    show #!1 models
    hide target p
    windowsize width 2432 height 1462
    hide atoms
    save {native_png_file} width 2432 height 1462 supersample 4 transparentBackground true
    """
    # 2dlab text 'Ground truth (PDB ID: {pdb_id})' color {label_color} size 48 x .03 y .92
    #     script += f"""
    # select #1-2
    # view sel orient pad -0.5
    # zoom 1.5
    # windowsize width 2432 height 1462
    # save {png_file} width 2432 height 1462 supersample 4 transparentBackground true
    # """
    return script


def generate_chimera_grid_scripts(
    data: pd.DataFrame,
    cxs_dir: Path,
    group_label: str,
    color_pal: dict[str, tuple[str, str]],
    gcs_model_root: str,
) -> pd.DataFrame:
    cxs_dir = cxs_dir / group_label
    cxs_png_dir = cxs_dir / "images"
    cxs_png_dir.mkdir(exist_ok=True, parents=True)
    cxs_script_dir = cxs_dir / "scripts"
    cxs_script_dir.mkdir(exist_ok=True, parents=True)
    structure_dir = cxs_dir / "structures"
    structure_dir.mkdir(exist_ok=True, parents=True)

    model_info = []
    for i, (method_name, df) in enumerate(data.groupby("method_name")):
        for j, r in df.iterrows():
            gcs_uri = f"{gcs_model_root}/{r.eval_dir}/{r.method_name}/{r.id}/{r.monomer_name}_decoys/{r.decoy}"
            id_monomer_dir = structure_dir / f"{r['id']}__{r.monomer_name}"
            id_monomer_dir.mkdir(exist_ok=True, parents=True)
            native = pinder_root / "pdbs" / (r["id"] + ".pdb")
            model = id_monomer_dir / f"{r.method_name}__{r.decoy}"
            if not model.is_file():
                os.system(f"gsutil cp {gcs_uri} {model}")
            pdb_id = r["id"].split("__")[0].upper()
            method_name = (
                pc.METHOD_LABELS.get(r.method_name, r.method_name)
                .replace("<br>", " ")
                .replace("</br>", "")
            )
            label = f"(PDB ID: {pdb_id}, DockQ: {round(r.DockQ, 2)})"
            png_file = cxs_png_dir / f"{pdb_id}--{r.method_name}--{r.monomer_name}.png"
            model_info.append(
                {
                    "path": model,
                    "method_name": method_name,
                    "R_chain": r.receptor_chain,
                    "L_chain": r.ligand_chain,
                    "label": label,
                    "png_file": png_file,
                    "DockQ": round(r.DockQ, 2),
                }
            )
            monomer_name = r.monomer_name

    padding = -0.6
    if (
        group_label == "oracle_hardest_mean"
        and pdb_id == "1GH7"
        and monomer_name == "apo"
    ):
        padding = -0.2
    if (
        group_label == "oracle_singleton"
        and pdb_id == "7QBM"
        and monomer_name == "predicted"
    ):
        padding = -0.2

    script = generate_chimerax_native_pred_overlay(
        native,
        model_info,
        png_file,
        color_pal,
        pad=padding,
    )
    with open(cxs_script_dir / f"{png_file.stem}.cxc", "w") as f:
        f.write(script)
    return model_info


def get_case_data(dockq: pd.DataFrame) -> dict[str, pd.DataFrame]:
    oracle_data = get_oracle_per_id_monomer_method(dockq)
    dockq_cases = dockq.copy()
    dockq_cases.loc[:, "pdb_id"] = [id.split("__")[0] for id in list(dockq_cases.id)]
    dockq_cases.loc[:, "pdb_id_monomer"] = (
        dockq_cases.pdb_id + "--" + dockq_cases.monomer_name
    )
    easiest_mean = (
        dockq_cases.sort_values("DockQ", ascending=False)
        .drop_duplicates(["method_name", "id", "monomer_name"], keep="first")
        .groupby(["id", "monomer_name"], as_index=False)
        .agg({"DockQ": "mean"})
        .sort_values("DockQ", ascending=False)
        .drop_duplicates("monomer_name", keep="first")
    )
    easiest_mean.loc[:, "id_monomer_name"] = (
        easiest_mean.id + "--" + easiest_mean.monomer_name
    )
    oracle_easiest_mean = get_case_subset(
        oracle_data, list(easiest_mean.id_monomer_name)
    )

    # Use different PDB IDs for oracle max mean to get larger representative set
    oracle_mean_easiest = (
        dockq_cases.query(f"pdb_id not in {list(oracle_easiest_mean.pdb_id)}")
        .sort_values("DockQ", ascending=False)
        .drop_duplicates(["monomer_name", "method_name", "id"], keep="first")
        .groupby(["id", "monomer_name"], as_index=False)
        .agg({"DockQ": "mean"})
        .sort_values("DockQ", ascending=False)
        .drop_duplicates("monomer_name", keep="first")
    )
    oracle_mean_easiest.loc[:, "id_monomer_name"] = (
        oracle_mean_easiest.id + "--" + oracle_mean_easiest.monomer_name
    )
    oracle_max_mean = get_case_subset(
        oracle_data, list(oracle_mean_easiest.id_monomer_name)
    )

    hardest_mean = (
        dockq.sort_values("DockQ", ascending=False)
        .drop_duplicates(["method_name", "id", "monomer_name"], keep="first")
        .groupby(["id", "monomer_name"], as_index=False)
        .agg({"DockQ": "mean"})
        .sort_values("DockQ", ascending=True)
        .drop_duplicates("monomer_name", keep="first")
    )
    hardest_mean.loc[:, "id_monomer_name"] = (
        hardest_mean.id + "--" + hardest_mean.monomer_name
    )
    oracle_hardest_mean = get_case_subset(
        oracle_data, list(hardest_mean.id_monomer_name)
    )
    oracle_variable = (
        dockq.sort_values("DockQ", ascending=False)
        .drop_duplicates(["monomer_name", "method_name", "id"], keep="first")
        .groupby(["id", "monomer_name"], as_index=False)
        .agg({"DockQ": "std"})
        .sort_values("DockQ", ascending=False)
        .drop_duplicates("monomer_name", keep="first")
    )
    oracle_variable.loc[:, "id_monomer_name"] = (
        oracle_variable.id + "--" + oracle_variable.monomer_name
    )
    oracle_max_var = get_case_subset(oracle_data, list(oracle_variable.id_monomer_name))
    max_delta_lst = []
    for (id, monomer_name), df in oracle_data.groupby(["id", "monomer_name"]):
        df = df.sort_values("DockQ", ascending=False).reset_index(drop=True)
        dq_delta = (
            df[["method_name", "monomer_name", "id", "DockQ"]]
            .DockQ.diff(1)
            .abs()
            .iloc[1]
        )
        dq_max = df.DockQ.max()
        dq_max_method = df.method_name.iloc[0]
        max_delta_lst.append(
            {
                "id": id,
                "monomer_name": monomer_name,
                "best_method": dq_max_method,
                "DockQ": dq_max,
                "DockQ_delta": dq_delta,
            }
        )
    max_delta = pd.DataFrame(max_delta_lst)
    singleton_methods = (
        max_delta.sort_values("DockQ_delta", ascending=False)
        .drop_duplicates("monomer_name", keep="first")
        .reset_index(drop=True)
    )
    singleton_methods.loc[:, "id_monomer_name"] = (
        singleton_methods.id + "--" + singleton_methods.monomer_name
    )
    oracle_singleton = get_case_subset(
        oracle_data, list(singleton_methods.id_monomer_name)
    )

    # Check if we have any duplicate representative systems across cases
    cases: dict[str, pd.DataFrame] = {
        "oracle_singleton": oracle_singleton,
        "oracle_max_var": oracle_max_var,
        "oracle_max_mean": oracle_max_mean,
        "oracle_easiest_mean": oracle_easiest_mean,
        "oracle_hardest_mean": oracle_hardest_mean,
    }
    return cases


def get_method_color_palette(dockq: pd.DataFrame) -> dict[str, tuple[str, str]]:
    # dockq = dockq.query(f'method_name in {list(keep_methods)}').reset_index(drop=True)
    methods = set(dockq.method_name)
    color_pal = generate_color_palette(n_methods=len(methods), lightness_shift=0.55)
    method_pal = {}
    for i, method_name in enumerate(sorted(list(methods))):
        method_label = (
            pc.METHOD_LABELS.get(method_name, method_name)
            .replace("<br>", " ")
            .replace("</br>", "")
        )
        method_pal[method_label] = color_pal[i]

    color_bar = []
    color_bar_map = {}
    for method_name, cgroup in method_pal.items():
        color_bar.append({"Method": f"{method_name}--R", "y": 10})
        color_bar.append({"Method": f"{method_name}--L", "y": 10})
        color_bar_map[f"{method_name}--R"] = cgroup[0]
        color_bar_map[f"{method_name}--L"] = cgroup[1]

    fig = px.bar(
        pd.DataFrame(color_bar),
        x="Method",
        y="y",
        color_discrete_map=color_bar_map,
        color="Method",
        template="simple_white",
        height=650,
    )
    fig.show()
    return method_pal


def generate_case_scripts(
    cases: dict[str, pd.DataFrame],
    color_pal: dict[str, tuple[str, str]],
    gcs_model_root: str,
    fig_dir: Path = get_pinder_location() / "publication_figures",
) -> None:
    cxs_dir = fig_dir / "chimerax_example_structures"
    cxs_dir.mkdir(parents=True, exist_ok=True)
    for group_label, case_data in cases.items():
        for monomer_name, data in case_data.groupby("monomer_name"):
            data.reset_index(drop=True, inplace=True)
            generate_chimera_grid_scripts(
                data,
                cxs_dir,
                group_label,
                color_pal=color_pal,
                gcs_model_root=gcs_model_root,
            )
    scripts = []
    for subdir in cxs_dir.glob("*"):
        if not subdir.is_dir():
            continue
        scripts.extend(list((subdir / "scripts").glob("*.cxc")))
    task_file = ""
    for script in scripts:
        task = f"ChimeraX --script {script} --exit;\n"
        task_file += task

    with open(cxs_dir / "run_scripts.list", "w") as f:
        f.write(task_file)


def generate_case_image_grids(
    cases: dict[str, pd.DataFrame],
    color_pal: dict[str, tuple[str, str]],
    gcs_model_root: str,
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
) -> None:
    cxs_dir = fig_dir / "chimerax_example_structures"
    for group_label, case_data in cases.items():
        for monomer_name, data in case_data.groupby("monomer_name"):
            data.reset_index(drop=True, inplace=True)
            model_info = generate_chimera_grid_scripts(
                data,
                cxs_dir,
                group_label,
                color_pal=color_pal,
                gcs_model_root=gcs_model_root,
            )
            model_info = (
                pd.DataFrame(model_info)
                .sort_values("DockQ", ascending=False)
                .reset_index(drop=True)
            )
            model_info.loc[:, "valid_file"] = [
                m.is_file() for m in list(model_info.png_file)
            ]
            model_info = model_info.query("valid_file").reset_index(drop=True)
            model_images = list(model_info.png_file)
            if model_info.empty:
                continue

            pdb_id, method_n, mono_name = model_images[0].stem.split("--")
            native_png = f"{pdb_id}--native--{mono_name}.png"
            native_png_file = model_images[0].parent / native_png
            images = [native_png_file] + model_images
            labels = ["Ground truth"]
            labels.extend(
                [
                    f'{method_name}\n({label.split(", ")[1]}'
                    for method_name, label in zip(
                        model_info.method_name, model_info.label
                    )
                ]
            )
            if len(labels) == 10:
                per_row = 5
            elif len(labels) == 6:
                per_row = 3
            else:
                per_row = 4
            top_adjust = (
                0.9
                if pdb_id in ["6BYI", "8CZP", "7KAZ", "6XA0", "4Z5Y", "3GPN"]
                else 0.99
            )
            hspace = (
                0.3
                if pdb_id in ["6BYI", "8CZP", "7KAZ", "6XA0", "4Z5Y", "3GPN"]
                else 0.01
            )
            bottom_adjust = 0.4 if pdb_id in ["6BYI", "8CZP", "6XA0", "4Z5Y"] else None
            image_grid(
                images,
                labels,
                pdb_id=pdb_id,
                per_row=per_row,
                top=top_adjust,
                bottom=bottom_adjust,
                hspace=hspace,
                right=0.99,
                wspace=0.01,
                figsize=(22, 10),
                output_png=cxs_dir / f"{group_label}_{monomer_name}_grid.{format}",
            )


def main(
    fig_dir: Path = get_pinder_location() / "publication_figures",
    format: str = "pdf",
    gcs_model_root: str = "gs://vantai-diffppi/pinder-2024-02-sandbox/eval/inference",
) -> None:
    dockq = get_dockq_metrics()
    cases = get_case_data(dockq)
    color_pal = get_method_color_palette(dockq)
    generate_case_scripts(
        cases=cases, color_pal=color_pal, gcs_model_root=gcs_model_root, fig_dir=fig_dir
    )
    # Use mpqueue.py from vantaiqueue to multiprocess tasks in run_scripts.list
    # mpqueue.py -c 10 run_scripts.list
    # Now collate the images into a single png/pdf in a grid
    generate_case_image_grids(
        cases,
        color_pal,
        gcs_model_root=gcs_model_root,
        fig_dir=fig_dir,
        format=format,
    )
