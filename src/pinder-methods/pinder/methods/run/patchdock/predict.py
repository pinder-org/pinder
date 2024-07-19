from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pinder.core.utils import setup_logger

log = setup_logger(__name__)


def patchdock(
    results_dir: Path,
    n_solutions: int,
    complex_id: str,
    complex_kind: str,
    complex_receptor: Path,
    complex_ligand: Path,
) -> None:
    """
    Run PatchDock on a single receptor-ligand complex and generate `n_solutions` docking solutions.
    Assume that all paths provided are local and all the files exist.
    Assume that `patchdock` binary is available in PATH.
    Save results to the specified `results_dir`. Directory will be created if needed.

    Parameters
    ----------
    results_dir : Path
        Path to the docking dataset.
        resulting path will be: `{results_dir}/patchdock/{id}/{kind}_decoys/model_{i}.pdb`
    n_solutions : int
        Number of solutions `i = {1..n_solutions}` to generate.
    id : str
        PINDER ID of the complex.
    kind : str
        `holo`, `apo` or `predicted`.
    receptor : Path
        All structures are expected to be pre-transformed and ready for docking.
    ligand : Path
        All structures are expected to be pre-transformed and ready for docking.

    """
    log.info(
        f"Running PatchDock ({n_solutions} solutions) on complex {complex_id} {complex_kind} {complex_receptor} {complex_ligand}"
    )

    current_dir = os.getcwd()
    output_dir = Path(tempfile.mkdtemp())
    log.info("OUT", output_dir)
    os.chdir(output_dir)

    dock_path = shutil.which("PatchDock")
    if not dock_path:
        log.error("PatchDock binary not found on $PATH!")
        # restore CWD
        os.chdir(current_dir)
        return

    # file prefices
    R_file_name = complex_receptor.name
    L_file_name = complex_ligand.name

    receptor = R_file_name.split(".")[0]
    ligand = L_file_name.split(".")[0]

    # Define the stages
    stages = [
        (
            "STAGE-1 Generation of parameters",
            f"{dock_path}/buildParams.pl {receptor}.pdb {ligand}.pdb 4.0",
        ),
        (
            "STAGE-2 Creation of the receptor electrostatic potential map",
            f"{dock_path}/patch_dock.Linux params.txt docking.res",
        ),
        (
            f"STAGE-3 Generate top {n_solutions} solutions",
            f"{dock_path}/transOutput.pl docking.res 1 {n_solutions}",
        ),
    ]
    # Execute the stages sequentially
    for stage_name, cmd in stages:
        log.debug(stage_name)
        log.debug(cmd)
        try:
            subprocess.run(cmd, shell=True)
            # optionally redirect stderr to stdout
            # subprocess.run(cmd, shell=True, stderr=subprocess.STDOUT)
        except Exception as e:
            log.error(f"ERROR while running stage {stage_name} on {complex_id}: {e}")
            os.chdir(current_dir)
            return

    # normalize the results by renaming the top solutions
    solutions_exist = False
    for i in range(1, n_solutions + 1):
        if Path(f"docking.res.{i}.pdb").exists():
            os.rename(f"docking.res.{i}.pdb", f"model_{i}.pdb")
            solutions_exist = True

    # delete extra files
    for fname in Path(".").glob("*.pdb"):
        if "model_" not in fname.name:
            fname.unlink()

    if not solutions_exist:
        log.info(f"ERROR: {complex_kind} {complex_id} did not finish successfully")
        os.chdir(current_dir)
        return None

    # change back to original working directory
    os.chdir(current_dir)

    # copy the results
    results_path_complete = (
        results_dir / f"patchdock/{complex_id}/{complex_kind}_decoys/"
    )
    log.info(f"Copying the results to {results_path_complete}")
    results_path_complete.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(output_dir) + "/*.pdb", results_path_complete)


def predict_patchdock(
    complex_config: dict[str, str | Path], params: dict[str, Any]
) -> None:
    """Run PatchDock on a single receptor-ligand complex and generate `n_solutions` docking solutions.
    Assume that all the specified paths are local and the files exist
    Save results to the specified `results_dir`. Directory is created if needed

    Parameters
    ----------
    complex_config : dictionary
        id: str
            PINDER ID of the complex
        kind: str
            `holo`, `apo` or `predicted`
        receptor: Path
            All structures are expected to be pre-transformed and ready for docking
        ligand: Path
            All structures are expected to be pre-transformed and ready for docking
    params : dictionary
        'results_dir' : Path
            Path to the docking dataset
            resulting path will be: `{results_dir}/patchdock/{id}/{kind}_decoys/model_{i}.pdb`
        n_solutions : int
            Number of solutions `i = {1..n_solutions}` to generate
        force: bool Force re-running the docking even if the results already exist
    """
    complex_id = str(complex_config["id"])
    complex_kind = str(complex_config["kind"])
    complex_receptor = Path(complex_config["receptor"])
    complex_ligand = Path(complex_config["ligand"])
    results_dir = Path(params["results_dir"])
    n_solutions = int(params["n_solutions"])
    force = bool(params["force"])

    results_path_complete = (
        results_dir / f"patchdock/{complex_id}/{complex_kind}_decoys/"
    )
    if not force and (results_path_complete / "model_1.pdb").exists():
        log.info(
            f"Skipping {complex_id} {complex_kind} because it already exists in {results_path_complete}"
        )
        return

    patchdock(
        results_dir,
        n_solutions,
        complex_id,
        complex_kind,
        complex_receptor,
        complex_ligand,
    )
