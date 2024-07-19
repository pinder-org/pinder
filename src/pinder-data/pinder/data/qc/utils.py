from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import requests
import shutil
import textwrap

from pinder.data.csv_utils import read_csv_non_default_na
from pinder.core import (
    get_index,
    get_metadata,
    get_pinder_location,
    get_supplementary_data,
    SupplementaryData,
)
from pinder.core.utils import setup_logger
from pinder.core.utils.cloud import gcs_read_dataframe


log = setup_logger(__name__, log_level=logging.WARNING)


def download_pdbfam_db(
    download_dir: Path | str,
    filename: str = "PDBfam.parquet",
    url: str = "http://dunbrack2.fccc.edu/ProtCiD/pfam/PDBfam.txt.gz",
    overwrite: bool = True,
) -> pd.DataFrame:
    if not isinstance(download_dir, Path):
        download_dir = Path(download_dir)
    if not download_dir.is_dir():
        download_dir.mkdir(parents=True)
    parquet_path = download_dir / filename
    download_path = download_dir / "PDBfam.txt.gz"
    if not overwrite and parquet_path.is_file():
        df = pd.read_parquet(parquet_path)
        return df
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(download_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    pfam = read_csv_non_default_na(download_path, sep="\t", dtype={"PdbId": "str"})
    pfam["PdbSeqStart"] = pfam["PdbSeqStart"].astype("str").astype("object")
    pfam["PdbSeqEnd"] = pfam["PdbSeqEnd"].astype("str").astype("object")
    pfam["PdbAlignStart"] = pfam["PdbAlignStart"].astype("str").astype("object")
    pfam["PdbAlignEnd"] = pfam["PdbAlignEnd"].astype("str").astype("object")
    pfam.to_parquet(parquet_path, index=False)
    return pfam


def load_index(index_file: Path | str | None = None) -> pd.DataFrame:
    if index_file:
        if not str(index_file).startswith("gs://"):
            index_file = Path(index_file)
        if str(index_file).endswith(".parquet"):
            pindex = pd.read_parquet(index_file)
        else:
            pindex = read_csv_non_default_na(index_file)
    else:
        pindex = get_index()
    return pindex


def load_metadata(metadata_file: Path | str | None = None) -> pd.DataFrame:
    if metadata_file:
        if not str(metadata_file).startswith("gs://"):
            metadata_file = Path(metadata_file)
        if str(metadata_file).endswith(".parquet"):
            metadata = pd.read_parquet(metadata_file)
        else:
            metadata = read_csv_non_default_na(metadata_file)
    else:
        metadata = get_metadata()
        supp = get_supplementary_data(SupplementaryData["supplementary_metadata"])
        metadata = pd.merge(metadata, supp, how="left")
    return metadata


def load_entity_metadata(
    entity_metadata_file: Path | str | None = None,
) -> pd.DataFrame:
    if entity_metadata_file:
        if not str(entity_metadata_file).startswith("gs://"):
            entity_metadata_file = Path(entity_metadata_file)
        if str(entity_metadata_file).endswith(".parquet"):
            entity_metadata = pd.read_parquet(entity_metadata_file)
        else:
            entity_metadata = read_csv_non_default_na(entity_metadata_file)
    else:
        entity_metadata = get_supplementary_data(SupplementaryData["entity_metadata"])
    return entity_metadata


def load_pfam_db(pfam_file: Path | str | None = None) -> pd.DataFrame:
    if pfam_file:
        if not str(pfam_file).startswith("gs://"):
            pfam_file = Path(pfam_file)
        if str(pfam_file).endswith(".tsv"):
            pfam_data = read_csv_non_default_na(pfam_file, sep="\t")
        else:
            pfam_data = gcs_read_dataframe(pfam_file)
    else:
        download_dir = get_pinder_location() / "data"
        download_dir.mkdir(exist_ok=True, parents=True)
        pfam_data = download_pdbfam_db(download_dir)

    pfam_data.set_index(["PdbID", "AuthChain"], inplace=True)
    return pfam_data


def view_potential_leaks(
    potential_leak_pairs: pd.DataFrame,
    pml_file: Path = Path("./").absolute() / "view_potential_leaks.pml",
    max_scenes: int = 100,
    pdb_dir: Path = get_pinder_location() / "pdbs",
    align_types: list[str] = ["align", "super", "chain"],
) -> None:
    scene_count = 0
    scene_pml = ""
    loaded_objects = set()
    scene_ids = set()
    if "log_pvalue" in potential_leak_pairs.columns:
        potential_leak_pairs = potential_leak_pairs.sort_values(
            "log_pvalue"
        ).reset_index(drop=True)
    elif "alignment_score" in potential_leak_pairs.columns:
        potential_leak_pairs = potential_leak_pairs.sort_values(
            "alignment_score"
        ).reset_index(drop=True)
    for _, r in potential_leak_pairs.iterrows():
        pid = r["query_id"]
        scene_objects = set()
        ref_dimer = pdb_dir / f"{pid}.pdb"
        ref_id = (
            pid.split("__")[0]
            + "_"
            + pid.split("--")[0].split("__")[1].split("_")[0]
            + "_"
            + pid.split("--")[1].split("__")[1].split("_")[0]
        )
        ref_load_id = ref_id
        dupe_prefix = 0
        while ref_load_id in loaded_objects:
            ref_load_id = f"{ref_id}-{dupe_prefix}"
            dupe_prefix += 1

        pml = f"""
        load {ref_dimer}, {ref_load_id};
        color protactinium, {ref_load_id} and chain R;
        color hotpink, {ref_load_id} and chain L;
        set grid_slot, -2, {ref_load_id};
        """
        loaded_objects.add(ref_load_id)
        scene_objects.add(ref_load_id)

        hit_id = r.hit_id
        hit_R, _ = hit_id.split("--")
        hit_dimer = pdb_dir / f"{hit_id}.pdb"
        for align_type in align_types:
            hit_load_id = f"{hit_id}-{align_type}"
            dupe_prefix = 0
            while hit_load_id in loaded_objects:
                hit_load_id = f"{hit_id}-{align_type}-{dupe_prefix}"
                dupe_prefix += 1

            if align_type == "chain":
                if not hasattr(r, "hit_chain"):
                    continue
                if hit_R.startswith(r.hit_pdb_id + "__" + r.hit_chain):
                    hit_chain = "R"
                else:
                    hit_chain = "L"
                ref_R, _ = pid.split("--")
                if ref_R.startswith(r.ref_pdb_id + "__" + r.ref_chain):
                    ref_chain = "R"
                else:
                    ref_chain = "L"
                aln_cmd = f"align {hit_load_id} and chain {hit_chain}, {ref_load_id} and chain {ref_chain}"
            else:
                aln_cmd = f"{align_type} {hit_load_id}, {ref_load_id}"
            pml += f"""
            load {hit_dimer}, {hit_load_id};
            {aln_cmd};
            color palecyan, {hit_load_id} and chain R;
            color lightpink, {hit_load_id} and chain L;
            """
            loaded_objects.add(hit_load_id)
            scene_objects.add(hit_load_id)

        pml += f"""
        set grid_mode, 1;
        orient {ref_load_id};
        zoom visible;
        """
        scene_id = ref_load_id + "-scene"
        scene_ids.add(scene_id)
        for obj in scene_objects:
            pml += f"enable {obj};\n"
        pml += f"scene {scene_id}, store;\n"
        # Newest pymol hangs indefinitely if all objects in a session are disabled
        if (scene_count + 1) == potential_leak_pairs.shape[0] or (
            scene_count + 1
        ) == max_scenes:
            pml += "".join([f"disable {obj};\n" for obj in list(loaded_objects)[0:-1]])
            pml += f"enable {list(loaded_objects)[-1]};\n"
        else:
            pml += "".join([f"disable {obj};\n" for obj in loaded_objects])
        scene_pml += pml
        if (scene_count + 1) == max_scenes:
            log.warning(f"Exceeding max scenes! {scene_count} scenes. Exiting...")
            break
        scene_count += 1

    with open(pml_file, "w") as f:
        f.write(textwrap.dedent(scene_pml))
