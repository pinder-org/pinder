from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import biotite.structure.io as strucio

from pinder.core.utils import setup_logger

log = setup_logger(__name__)


def frodock(
    results_dir: Path,
    n_solutions: int,
    mpi_np: int,
    complex_id: str,
    complex_kind: str,
    complex_receptor: Path,
    complex_ligand: Path,
) -> None:
    """
    Run FroDock on a single receptor-ligand complex and generate `n_solutions` docking solutions.
    Assume that all paths provided are local and all the files exist.
    Assume that `frodock` binary is available in PATH.
    Save results to the specified `results_dir`. Directory will be created if needed.

    Parameters
    ----------
    results_dir : Path
        Path to the docking dataset.
        resulting path will be: `{results_dir}/frodock/{id}/{kind}_decoys/model_{i}.pdb`
    n_solutions : int
        Number of solutions `i = {1..n_solutions}` to generate
    mpi_np : int
        Number of processors to use for MPI. Default is 8.
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
        f"Running FroDock ({n_solutions} solutions) on complex {complex_id} {complex_kind} {complex_receptor} {complex_ligand}"
    )

    current_dir = os.getcwd()
    output_dir = Path(tempfile.mkdtemp())
    log.info("OUT", output_dir)
    os.chdir(output_dir)

    mpi_path = shutil.which("mpirun")
    dock_path = shutil.which("frodock_mpi_gcc")
    if not dock_path:
        log.error("FroDock binary not found on $PATH!")
        # restore CWD
        os.chdir(current_dir)
        return
    if not mpi_path:
        log.error("mpirun binary not found on $PATH!")
        # restore CWD
        os.chdir(current_dir)
        return

    # file prefices
    R_file_name = complex_receptor.name
    L_file_name = complex_ligand.name

    receptor = R_file_name.split(".")[0]
    ligand = L_file_name.split(".")[0]

    # FRODOCK binaries
    # allow to run MPI as root
    os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    # --use-hwthread-cpus solves "There are not enough slots available in the system ..." error
    # mpi_np = 45
    run = f"mpirun -np {mpi_np} --use-hwthread-cpus "
    frodock_bin_dir = Path(dock_path).parent
    pre = frodock_bin_dir
    suff = "_mpi_gcc"
    suff2 = "_gcc"
    # Define the stages
    stages = [
        (
            "STAGE-1 Creation of receptor vdw potential map",
            f"{run} {pre}frodockgrid{suff} {receptor}.pdb -o {receptor}_W.mrc",
        ),
        (
            "STAGE-2 Creation of the receptor electrostatic potential map",
            f"{run} {pre}frodockgrid{suff} {receptor}.pdb -o {receptor}_E.mrc -m 1 -t A",
        ),
        (
            "STAGE-3 Creation of the receptor desolvation potential map",
            f"{run} {pre}frodockgrid{suff} {receptor}.pdb -o {receptor}_DS.mrc -m 3",
        ),
        (
            "STAGE-4 Creation of the ligand desolvation potential map",
            f"{run} {pre}frodockgrid{suff} {ligand}.pdb -o {ligand}_DS.mrc -m 3",
        ),
        (
            "STAGE-5 Performing the docking",
            f"{run} {pre}frodock{suff} {receptor}_ASA.pdb {ligand}_ASA.pdb -w {receptor}_W.mrc -e {receptor}_E.mrc --th 10 -d {receptor}_DS.mrc,{ligand}_DS.mrc -t A -o dock.dat -s {pre}soap.bin",
        ),
        (
            "STAGE-6 Clustering and visualization of predictions",
            f"{pre}frodockcluster{suff2} dock.dat {ligand}.pdb --nc {n_solutions} -o clust_dock.dat",
        ),
        (
            f"STAGE-7 Visualize the first {n_solutions} solutions",
            f"{pre}frodockview{suff2} clust_dock.dat -r 1-{n_solutions}",
        ),
        (
            f"STAGE-8 Coordinate generation of the {n_solutions} best predicted solutions",
            f"{pre}frodockview{suff2} clust_dock.dat -r 1-{n_solutions} -p {ligand}.pdb",
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
    # docked ligand file exists but we need a complete pose
    receptor_array = strucio.load_structure(f"{receptor}.pdb")
    for i in range(1, n_solutions + 1):
        ligand_file = Path(f"{ligand}_{i}.pdb")
        if ligand_file.exists():
            pose_file = Path(f"{receptor}_{ligand}_{i}.pdb")
            if not pose_file.exists():
                ligand_array = strucio.load_structure(str(ligand_file))
                pose_array = receptor_array + ligand_array
                strucio.save_structure(str(pose_file), pose_array)
            if pose_file.exists():
                solutions_exist = True
                os.rename(pose_file, f"model_{i}.pdb")

    if not solutions_exist:
        log.info(f"ERROR: {complex_kind} {complex_id} did not finish successfully")
        os.chdir(current_dir)
        return

    # delete extra files
    for fname in Path(".").glob("*.pdb"):
        if "model_" not in fname.name:
            fname.unlink()

    # change back to original working directory
    os.chdir(current_dir)

    # copy the results
    results_path_complete = results_dir / f"frodock/{complex_id}/{complex_kind}_decoys/"
    log.info(f"Copying the results to {results_path_complete}")
    results_path_complete.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(output_dir) + "/*.pdb", results_path_complete)


def predict_frodock(
    complex_config: dict[str, str | Path], params: dict[str, Any]
) -> None:
    """Run FroDock on a single receptor-ligand complex and generate `n_solutions` docking solutions.
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
            resulting path will be: `{results_dir}/frodock/{id}/{kind}_decoys/model_{i}.pdb`
        n_solutions : int
            Number of solutions `i = {1..n_solutions}` to generate
        mpi_np : int
            Number of processors to use for MPI. Default is 8.
        force: bool Force re-running the docking even if the results already exist
    """
    complex_id = str(complex_config["id"])
    complex_kind = str(complex_config["kind"])
    complex_receptor = Path(complex_config["receptor"])
    complex_ligand = Path(complex_config["ligand"])
    results_dir = Path(params["results_dir"])
    n_solutions = int(params["n_solutions"])
    mpi_np = int(params.get("mpi_np", 8))
    force = bool(params["force"])

    results_path_complete = results_dir / f"frodock/{complex_id}/{complex_kind}_decoys/"
    if not force and (results_path_complete / "model_1.pdb").exists():
        log.info(
            f"Skipping {complex_id} {complex_kind} because it already exists in {results_path_complete}"
        )
        return

    frodock(
        results_dir,
        n_solutions,
        mpi_np,
        complex_id,
        complex_kind,
        complex_receptor,
        complex_ligand,
    )
