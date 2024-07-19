"""Calculate interface alignment scores via IS-align.

Used to handle similarity hit finding following initial alignments via Foldseek or MMSeq2.

For full method details, please see: https://doi.org/10.1093/bioinformatics/btq404
"""

from __future__ import annotations
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from mpire import WorkerPool

from pinder.core import get_pinder_location
from pinder.core.utils import setup_logger
from pinder.data.config import IalignConfig

log = setup_logger(__name__, log_level=logging.WARNING)


def ialign(
    query_id: str,
    query_pdb: Path,
    hit_id: str,
    hit_pdb: Path,
    config: IalignConfig = IalignConfig(),
) -> dict[str, str | float | int] | None:
    ialign_path = shutil.which("ialign.pl")
    if not ialign_path:
        log.error("ialign binary not found on $PATH!")
        return None
    cmd_list = [
        f"perl {ialign_path}",
        f"-a {config.alignment_printout}",
        f"-q {config.speed_mode}",
        f"-minp {config.min_residues}",
        f"-mini {config.min_interface}",
        f"-dc {config.distance_cutoff}",
        f"-w {config.output_prefix}",
        f"{query_pdb} RL {hit_pdb} RL",
    ]
    command = " ".join(cmd_list)
    root_temp_dir = get_pinder_location() / "tmp"
    root_temp_dir.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=root_temp_dir) as temp_dir:
        os.chdir(temp_dir)
        completed_process = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output_lines = completed_process.stdout.splitlines()
        ialign_results: dict[str, str | float | int] = {
            "query_id": query_id,
            "hit_id": hit_id,
        }
        for line in output_lines:
            if line.startswith("IS-score"):
                parts = line.split(",")
                for part in parts:
                    key_value = part.split("=")
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    ialign_results[key] = float(value)
            elif line.startswith("Number of aligned residues"):
                parts = line.split("=")
                ialign_results["Number of aligned residues"] = int(parts[1].strip())
            elif line.startswith("Number of aligned contacts"):
                parts = line.split("=")
                ialign_results["Number of aligned contacts"] = int(parts[1].strip())
            elif line.startswith("RMSD"):
                parts = line.split(",")
                for part in parts:
                    key_value = part.split("=")
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    ialign_results[key] = float(value)
        if completed_process.returncode != 0:
            log.error(f"Error running iAlign: {completed_process.stderr}")
            return None
        os.chdir("..")
    if "RMSD" in ialign_results:
        return ialign_results
    else:
        return None


def ialign_all(
    df: pd.DataFrame,
    pdb_root: Path = get_pinder_location() / "pdbs",
    n_jobs: int = 48,
    config: IalignConfig = IalignConfig(),
) -> pd.DataFrame:
    inputs = []
    for dimer in df[["id", "hit_id"]].itertuples():
        query_id = dimer.id
        hit_id = dimer.hit_id
        query_pdb = pdb_root / query_id
        if not str(query_pdb).endswith(".pdb"):
            query_pdb = query_pdb.with_suffix(".pdb")
        hit_pdb = pdb_root / hit_id
        if not str(hit_pdb).endswith(".pdb"):
            hit_pdb = hit_pdb.with_suffix(".pdb")

        if not Path(query_pdb).exists() or not Path(hit_pdb).exists():
            continue

        inputs.append((query_id, query_pdb, hit_id, hit_pdb, config))

    with WorkerPool(n_jobs=n_jobs) as pool:
        results = pool.map(ialign, inputs, progress_bar=True)

    if results:
        final_df = pd.DataFrame([result for result in results if result is not None])
        return final_df


def process_in_batches(
    df: pd.DataFrame,
    batch_size: int = 1000,
    batch_offset: int = 0,
    overwrite: bool = False,
    cache_dir: Path = get_pinder_location() / "ialign_results",
    n_jobs: int = 48,
    config: IalignConfig = IalignConfig(),
) -> None:
    num_batches = df.shape[0] // batch_size + (1 if df.shape[0] % batch_size else 0)
    for batch in range(num_batches):
        log.info(f"Batch {batch + 1} out of {num_batches}, offset {batch_offset}")
        fname_out = f"{cache_dir}/batch_{(batch + batch_offset)}.parquet"
        if not overwrite and Path(fname_out).exists():
            continue
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        df_slice = df.iloc[start_idx:end_idx]
        batch_results = ialign_all(df_slice, n_jobs=n_jobs, config=config)
        batch_results.to_parquet(fname_out, index=False)
