from __future__ import annotations
import json
import multiprocessing
from concurrent import futures
from itertools import repeat
from pathlib import Path
from tqdm import tqdm

import requests
import pandas as pd


from pinder.data.annotation.contact_classification import (
    get_crystal_contact_classification,
)
from pinder.data.annotation.detached_components import get_num_connected_components
from pinder.data.annotation.planarity import get_planarity
from pinder.data.annotation.elongation import calculate_elongation
from pinder.data.annotation.interface_gaps import annotate_interface_gaps
from pinder.data.config import PinderDataGenConfig
from pinder.data.csv_utils import read_csv_non_default_na

from pinder.core.utils import setup_logger


log = setup_logger(__name__)


def annotate_pisalite(path_to_pdb: Path, use_cache: bool = True) -> None:
    """
    Annotate PDB entry with PDBe PISA Lite service.

    This function will make two REST API queries to the PDBe PISA Lite service
    and save the results as JSON files in the directory containing the PDB entry.
    It will create two files in the directory:
    - {pdb_id}-pisa-lite-assembly.json
    - {pdb_id}-pisa-lite-interfaces.json

    Parameters
    ----------
    path_to_pdb : Path
        Path to the directory containing the PDB entry. PDB ID is expected to be encoded in the directory name.
    use_cache : bool
        Whether to skip request if a checkpoint file with name
        `checkpoint-pisa.txt` exists in the `path_to_pdb` directory.

    """
    # two_char_code = path_to_pdb.parent.stem
    pdb_id = path_to_pdb.stem[-4:]

    checkpoint_file = path_to_pdb / "checkpoint-pisa.txt"
    if use_cache and checkpoint_file.is_file():
        log.info(f"Skipping PISA {path_to_pdb.stem}, checkpoint exists")
        return

    # Define the URLs for the REST API queries
    assembly_url = f"https://www.ebi.ac.uk/pdbe/api/pisa/assembly/{pdb_id}/1"
    interfaces_url = f"https://www.ebi.ac.uk/pdbe/api/pisa/interfaces/{pdb_id}/1"

    # Make the REST API queries and handle potential errors
    try:
        assembly_response = requests.get(assembly_url)
        assembly_response.raise_for_status()  # Raises a HTTPError if the response was unsuccessful
        assembly_data = assembly_response.json()
    except (requests.HTTPError, ValueError):
        log.error(f"Failed to fetch or parse assembly data for {pdb_id}")
        assembly_data = None

    try:
        interfaces_response = requests.get(interfaces_url)
        interfaces_response.raise_for_status()  # Raises a HTTPError if the response was unsuccessful
        interfaces_data = interfaces_response.json()
    except (requests.HTTPError, ValueError):
        log.error(f"Failed to fetch or parse interfaces data for {pdb_id}")
        interfaces_data = None

    # Save the results as JSON files if the data was fetched and parsed successfully
    if assembly_data is not None:
        with open(
            f"{path_to_pdb}/{pdb_id}-pisa-lite-assembly.json", "w"
        ) as assembly_file:
            json.dump(assembly_data, assembly_file)

    if interfaces_data is not None:
        with open(
            f"{path_to_pdb}/{pdb_id}-pisa-lite-interfaces.json", "w"
        ) as interfaces_file:
            json.dump(interfaces_data, interfaces_file)
    if use_cache:
        with open(checkpoint_file, "w") as f:
            f.write("complete")


def annotate_complex(
    args: tuple[Path, float],
    use_cache: bool = True,
) -> None:
    """This function annotates a protein complex dimer.

    Parameters
    ----------
    args : Tuple[Path, float]
        The path to the protein complex and the radius for the annotation process.
    use_cache : bool
        Whether to skip calculations if the annotation output tsv exists.

    Returns
    -------
    None
        This function does not return any value. It annotates the protein complex with:
        - crystal contacts
        - number of disconnected components
    """
    # Get number of disconnected components
    pdb_file, radius = args
    log.info(f"Annotating {pdb_file} ...")
    output_tsv = pdb_file.parent / f"{pdb_file.stem}.tsv"
    if use_cache and output_tsv.is_file():
        log.info(f"Skipping dimer annotation {pdb_file}, checkpoint exists")
        return

    try:
        no_of_components1, no_of_components2, _ = get_num_connected_components(
            (pdb_file, radius)
        )
    except Exception as e:
        log.error(
            f"Failed to get number of disconnected components for {pdb_file} due to: {str(e)}"
        )
        no_of_components1, no_of_components2 = -1, -1

    # get the crystal contacts
    try:
        crystal_contacts = get_crystal_contact_classification(pdb_file)
        cols = [
            "path",
            "intermolecular_contacts",
            "charged_charged_contacts",
            "charged_polar_contacts",
            "charged_apolar_contacts",
            "polar_polar_contacts",
            "apolar_polar_contacts",
            "apolar_apolar_contacts",
            "link_density",
            "label",
            "probability",
        ]
        results_df = pd.DataFrame([crystal_contacts], columns=cols)
    except Exception as e:
        log.error(f"Failed to get crystal contacts for {pdb_file} due to: {str(e)}")
        results_df = pd.DataFrame(columns=cols)

    # just keep the file name for the path
    results_df["path"] = pdb_file.name

    # chains are ordered alphabetically, same as in the elongation data frame below
    results_df["number_of_components_1"] = no_of_components1
    results_df["number_of_components_2"] = no_of_components2

    # calculate planarity
    try:
        planarity = get_planarity(pdb_file)
    except Exception as e:
        log.error(f"Failed to calculate planarity for {pdb_file} due to: {str(e)}")
        planarity = -1
    results_df["planarity"] = planarity

    # calculate elongation
    try:
        elongation_tuple = calculate_elongation(pdb_file)
        elongation_cols = [
            "pdb_filename",
            "max_var_1",
            "max_var_2",
            "length1",
            "length2",
            "num_atom_types",
            "chain_id1",
            "chain_id2",
            "chain1_id",
            "chain2_id",
        ]
        elongation_df = pd.DataFrame([elongation_tuple], columns=elongation_cols)
        elongation_df = elongation_df.drop(columns=["pdb_filename"])
    except Exception as e:
        log.error(f"Failed to calculate elongation for {pdb_file} due to: {str(e)}")
        elongation_df = pd.DataFrame(columns=elongation_cols)

    list_of_annotation_df = [results_df, elongation_df]

    # annotate interface gaps
    try:
        df_gaps = annotate_interface_gaps(pdb_file)
        if isinstance(df_gaps, pd.DataFrame):
            df_gaps = df_gaps.drop(columns=["pdb_id", "chain1", "chain2"])
            list_of_annotation_df.append(df_gaps)
    except Exception as e:
        log.error(f"Failed to annotate interface gaps {pdb_file} due to: {str(e)}")
        # pass

    try:
        results_df = pd.concat(list_of_annotation_df, axis=1)
        # save data frame to TSV file
        log.info(f"Saving results to {output_tsv}")
        results_df.to_csv(output_tsv, sep="\t", index=False)
    except Exception as e:
        log.error(f"Failed to save results for {pdb_file} due to: {str(e)}")


def get_pisa_annotations(
    mmcif_list: list[Path],
    parallel: bool = True,
    max_workers: int | None = None,
    config: PinderDataGenConfig = PinderDataGenConfig(),
    use_cache: bool = True,
) -> None:
    """This function fetches PISA annotations for list of PDB entries.

    Parameters
    ----------
    mmcif_list : Path
        The list of mmcif entry files to process in a batch.
    parallel : bool
        If True, files will be processed in parallel.
    max_workers : int | None
        If specified, limits number of processes to spawn in parallel mode.
    config : PinderDataGenConfig
        Configuration parameters for dataset generation.

    Returns
    -------
    None
        This function does not return any value.
        It processes the PDB files in the given directory and saves the results.
    """
    log.info(f"Number of PDB entries: {len(mmcif_list)}")
    entry_dirs = [cif.parent for cif in mmcif_list]
    if len(entry_dirs) > 0:
        if parallel:
            try:
                with futures.ProcessPoolExecutor(
                    mp_context=multiprocessing.get_context("spawn"),
                    max_workers=max_workers,
                ) as exe:
                    exe.map(annotate_pisalite, entry_dirs, repeat(use_cache))

            except Exception as e:
                log.error(
                    f"Failed to annotate PDB entries with PISA-lite in parallel due to: {str(e)}"
                )
        else:
            # process all entries in serial
            for args in tqdm(entry_dirs):
                try:
                    annotate_pisalite(args, use_cache=use_cache)
                except Exception as e:
                    log.error(
                        f"Failed to annotate PDB entries with PISA-lite {args} due to: {str(e)}"
                    )


def get_dimer_annotations(
    dimer_list: list[Path],
    parallel: bool = True,
    max_workers: int | None = None,
    config: PinderDataGenConfig = PinderDataGenConfig(),
    use_cache: bool = True,
) -> None:
    """This function annotates a list of dimer PDB files.

    Parameters
    ----------
    dimer_list : Path
        The list of dimer PDB files to process in a batch.
    max_workers : int | None
        If specified, limits number of processes to spawn in parallel mode.
    config : PinderDataGenConfig
        Configuration parameters for dataset generation.

    Returns
    -------
    None
        This function does not return any value.
        It processes the PDB files in the given directory and saves the results.
    """
    # dimers (could be none or more than one per entry)
    dimer_processing_queue = [
        (pdb_fname, config.connected_component_radius) for pdb_fname in dimer_list
    ]
    log.info(f"Number of dimers: {len(dimer_processing_queue)}")
    if len(dimer_processing_queue) > 0:
        if parallel:
            # process all dimers in parallel
            try:
                with futures.ProcessPoolExecutor(
                    mp_context=multiprocessing.get_context("spawn"),
                    max_workers=max_workers,
                ) as exe:
                    exe.map(annotate_complex, dimer_processing_queue, repeat(use_cache))
            except Exception as e:
                log.error(f"Failed to process dimers in parallel due to: {str(e)}")
        else:
            # process all dimers in serial
            for args in tqdm(dimer_processing_queue):
                try:
                    annotate_complex(args, use_cache=use_cache)
                except Exception as e:
                    log.error(f"Failed to process dimers {args} due to: {str(e)}")


def get_annotations(
    data_dir: Path,
    two_char_code: str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    config: PinderDataGenConfig = PinderDataGenConfig(),
) -> None:
    """This function gets annotations for a given PDB directory.

    Parameters
    ----------
    pdb_dir : Path
        The path to the directory containing the PDB files.
    two_char_code : str, optional
        The two character code for the PDB files, by default None.
    parallel : bool
        If True, files will be processed in parallel.
    max_workers : int | None
        If specified, limits number of processes to spawn in parallel mode.
    config : PinderDataGenConfig
        Configuration parameters for dataset generation.

    Returns
    -------
    None
        This function does not return any value.
        It processes the PDB files in the given directory and saves the results.
    """

    pdb_dir = Path(data_dir)

    # PDB entries
    entry_processing_queue = [
        fname for fname in pdb_dir.glob(f"{two_char_code}/**/*_xyz-enrich.cif.gz")
    ]
    get_pisa_annotations(
        entry_processing_queue, parallel=parallel, max_workers=max_workers
    )

    # dimers (could be none or more than one per entry)
    dimer_list = [
        pdb_fname
        for pdb_fname in pdb_dir.glob(
            f"{two_char_code.lower() if two_char_code else '*'}/**/*__*--*__*.pdb"
        )
    ]
    get_dimer_annotations(
        dimer_list,
        parallel=parallel,
        max_workers=max_workers,
        config=config,
    )


def pisa_json_to_dataframe(json_file: Path) -> pd.DataFrame:
    """Convert the PISA JSON file to a single-row Pandas DataFrame."""
    # json_data is a nested dictionary
    with open(json_file, "r") as file:
        json_data = json.load(file)

    # Flattening the nested dictionary
    flat_dict = {}
    for key, value in json_data.items():
        # Adding the first key as "entry_id"
        flat_dict["entry_id"] = key
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, dict):
                for sub_sub_key, sub_sub_value in sub_value.items():
                    flat_dict[f"{sub_key}_{sub_sub_key}"] = sub_sub_value
            else:
                flat_dict[sub_key] = sub_value

    # Convert the flat dictionary to a single-row DataFrame
    df = pd.DataFrame([flat_dict])
    return df


def collect_metadata(
    pdb_entries: list[Path], include_pisa: bool = False
) -> pd.DataFrame:
    """Collect metadata from PDB entries.

    Parameters
    ----------
    pdb_entries : List[Path]
        List of paths to PDB entries.

    include_pisa : bool
        Whether to include PISA annotations. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata for each PDB entry.
    """
    list_metadata = []
    for path in tqdm(sorted(pdb_entries)):
        # 4 letter PDB code
        entry_id = path.name.split("_")[1][-4:].lower()
        metadata_file = path / f"{entry_id}-metadata.tsv"
        pisa_json_file = path / f"{entry_id}-pisa-lite-assembly.json"

        try:
            _metadata = pd.read_csv(metadata_file, sep="\t")
            _metadata["entry_id"] = entry_id
        except Exception:
            _metadata = pd.DataFrame(
                [
                    {
                        "entry_id": entry_id,
                        "status": "missing",
                        "method": "UNKNOWN",
                        "date": "UNKNOWN",
                        "resolution": -1.0,
                        "assembly": 0.0,
                        "assembly_details": "",
                        "oligomeric_details": "unknown",
                        "oligomeric_count": 0,
                        "biol_details": "",
                        "complex_type": "unknown",
                        "n_chains": 0,
                    }
                ]
            )

        if include_pisa:
            try:
                _pisa = pisa_json_to_dataframe(pisa_json_file)
                # remove the entry_id column or it will be duplicated
                _pisa.drop(columns=["entry_id"], inplace=True)
                if _metadata.shape[0] > 0:
                    _metadata = pd.concat([_metadata, _pisa], axis=1)
            except Exception:
                pass
        list_metadata.append(_metadata)

    if len(list_metadata) > 0:
        df_metadata = pd.concat(list_metadata, ignore_index=True, sort=False)
    else:
        df_metadata = pd.DataFrame()
    return df_metadata


def collect_interacting_chains(pdb_entries: list[Path]) -> pd.DataFrame:
    """Collect interacting chains from PDB entries.

    Parameters
    ----------
    pdb_entries : List[Path]
        List of paths to PDB entries.

    Returns
    -------
    pd.DataFrame
        DataFrame containing interacting chains for each PDB entry.
    """
    list_interacting_chains = []
    for path in tqdm(sorted(pdb_entries)):
        # 4 letter PDB code
        entry_id = path.name.split("_")[1][-4:].lower()
        interacting_chains_file = path / f"{entry_id}-interacting_chains.tsv"
        try:
            _interacting_chains = read_csv_non_default_na(
                interacting_chains_file, sep="\t"
            )
            _interacting_chains["entry_id"] = entry_id
        except Exception:
            _interacting_chains = pd.DataFrame([{"entry_id": entry_id}])
        list_interacting_chains.append(_interacting_chains)
    if len(list_interacting_chains) > 0:
        df_interacting_chains = pd.concat(
            list_interacting_chains, ignore_index=True, sort=False
        )
    else:
        df_interacting_chains = pd.DataFrame()
    return df_interacting_chains


def collect_annotations(pdb_entries: list[Path]) -> pd.DataFrame:
    """Collect annotations from PDB entries.

    Parameters
    ----------
    pdb_entries : List[Path]
        List of paths to PDB entries.

    Returns
    -------
    pd.DataFrame
        DataFrame containing annotations for each PDB entry.
    """
    log.info(f"Collecting annotations")
    list_annotations = []
    for path in tqdm(sorted(pdb_entries)):
        # 4 letter PDB code
        entry_id = path.name.split("_")[1][-4:].lower()
        for fname in path.glob("*__*--*__*.tsv"):
            try:
                _df = read_csv_non_default_na(fname, sep="\t")
                _df["entry_id"] = entry_id
                list_annotations.append(_df)
            except Exception:
                pass

    if len(list_annotations) > 0:
        df_annotations = pd.concat(list_annotations, ignore_index=True, sort=False)
    else:
        df_annotations = pd.DataFrame()
    return df_annotations


def collect(data_dir: Path, pinder_dir: Path) -> None:
    """Collect annotations, metadata and generate index.

    This function is responsible for collecting annotations, metadata and generating an index.

    Parameters
    ----------
    data_dir : Path
        The directory where the data is stored.
    pinder_dir : Path
        The directory where the Pinder data is stored.

    Returns
    -------
    None
    """
    log.info("Generating index...")

    pdb_dir = Path(data_dir)
    pinder_dir = Path(pinder_dir)

    pdb_entries = []
    for two_char_dir in pdb_dir.iterdir():
        if two_char_dir.is_dir():
            pdb_entries.extend(
                [
                    entry
                    for entry in two_char_dir.iterdir()
                    if entry.is_dir() and entry.name.startswith("pdb_")
                ]
            )
    log.info(f"Number of PDB entries: {len(pdb_entries)}")

    metadata_df = collect_metadata(pdb_entries)
    metadata_df.loc[metadata_df.oligomeric_count == "?", "oligomeric_count"] = 0
    metadata_df["oligomeric_count"] = metadata_df["oligomeric_count"].astype(int)
    # Can happen if the entry was a non-protein assembly or entity/assembly generation failed
    metadata_df.loc[metadata_df.n_chains.isna(), "n_chains"] = 0
    metadata_df["n_chains"] = metadata_df["n_chains"].astype(int)
    metadata_df["resolution"] = metadata_df["resolution"].replace(".", 0.0)
    metadata_df["resolution"] = metadata_df["resolution"].astype(float)

    output_metadata_path = pinder_dir / "structural_metadata.parquet"
    metadata_df.to_parquet(output_metadata_path, index=False)
    del metadata_df

    interacting_chains_df = collect_interacting_chains(pdb_entries)
    output_interfaces_path = pinder_dir / "interfaces.parquet"
    interacting_chains_df.to_parquet(output_interfaces_path, index=False)
    del interacting_chains_df

    annotations_df = collect_annotations(pdb_entries)
    output_annotations_path = pinder_dir / "interface_annotations.parquet"
    annotations_df.to_parquet(output_annotations_path, index=False)
    del annotations_df
