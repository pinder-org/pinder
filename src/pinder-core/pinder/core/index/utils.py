from __future__ import annotations
import os
from enum import Enum
from functools import lru_cache, reduce, partial
from pathlib import Path
from zipfile import ZipFile, error as zip_error

import gcsfs
import numpy as np
import pandas as pd
from google.cloud.storage.client import Client
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from fnmatch import fnmatch

from pinder.core.utils import setup_logger
from pinder.core.utils.cloud import Gsutil, gcs_read_dataframe
from pinder.core.utils import constants as pc


log = setup_logger(__name__)


def get_pinder_location() -> Path:
    """
    Determines the base directory location for the Pinder data.

    First, check the environment for PINDER_DATA_DIR. If provided, assume
    it is the full path to the directory containing the index, pdbs/, and mappings/.
    Otherwise, the PINDER_BASE_DIR environment variable is checked, and if unset,
    falls back to the default XDG_DATA_HOME location (~/.local/share), and
    appends the current version of the Pinder release. The Pinder release can be
    controlled via the PINDER_RELEASE environment variable.

    Returns:
        Path: The path to the base directory for Pinder data.
    """
    pinder_data = os.environ.get("PINDER_DATA_DIR", "")
    default_loc = os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")
    if pinder_data:
        pinder_base = Path(pinder_data)
    else:
        base_dir = os.environ.get("PINDER_BASE_DIR", default_loc)
        pinder_release = os.environ.get("PINDER_RELEASE", "2024-02")
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        pinder_base = base_dir.absolute() / "pinder" / pinder_release
    if not pinder_base.is_dir():
        pinder_base.mkdir(exist_ok=True, parents=True)
        log.debug(f"Pinder is located at {pinder_base}")
    return pinder_base


def get_pinder_bucket_root() -> str:
    """Constructs the root bucket path for the Pinder data in Google Cloud Storage
    based on the PINDER_RELEASE environment variable.

    Returns:
        str: The root bucket path as a string.
    """
    pinder_release = os.environ.get("PINDER_RELEASE", "2024-02")
    return f"gs://pinder/{pinder_release}"


def get_index_location(
    csv_name: str = "index.parquet", remote: bool = False
) -> Path | str:
    """Gets the file path for the Pinder index CSV/Parquet file, either locally or remotely.

    Parameters:
        csv_name (str): The name of the CSV/Parquet file. Defaults to "index.parquet".
        remote (bool): A flag to determine if the remote file path should be returned.
                       Defaults to False.

    Returns:
        Union[Path, str]:
            The path to the index CSV/Parquet file, either as a Path object (local)
            or a string (remote).
    """
    if remote:
        return f"{get_pinder_bucket_root()}/{csv_name}"
    return get_pinder_location() / csv_name


@lru_cache
def get_index(csv_name: str = "index.parquet", update: bool = False) -> pd.DataFrame:
    """Retrieves the Pinder index as a pandas DataFrame. If the index is not already
    loaded, it reads from the local CSV/Parquet file, or downloads it if not present.

    Parameters:
        csv_name (str): The name of the CSV/Parquet file to load. Defaults to "index.parquet".
        update (bool): Whether to force update index on disk even if it exists.
            Default is False.

    Returns:
        pd.DataFrame: The Pinder index as a DataFrame.
    """
    if str(csv_name).startswith("gs://") or "/" in str(csv_name):
        # Its a custom index in local filepath or gcs uri
        custom_index = True
        local_index = get_pinder_location() / "custom_indices" / Path(csv_name).name
        if not local_index.parent.is_dir():
            local_index.parent.mkdir(exist_ok=True, parents=True)
    else:
        local_index = Path(get_index_location(csv_name))
        custom_index = False
    if local_index.suffix in [".csv", ".gz"]:
        reader = pd.read_csv
        writer = "to_csv"
        cast_types = True
        # reader_kwargs = {}
    elif local_index.suffix == ".parquet":
        reader = pd.read_parquet
        writer = "to_parquet"
        cast_types = False
        # reader_kwargs = {"engine": "pyarrow"}
    else:
        raise ValueError(
            f"Unsupported file type for index: {local_index.suffix}! Use csv or parquet."
        )
    if local_index.is_file() and not update and not custom_index:
        # Only load cached index when update is False and its not a custom index
        pindex = reader(local_index)
    else:
        # Given the potential duplicate named indices, we force-update any custom index
        if custom_index:
            pindex = gcs_read_dataframe(csv_name)
        else:
            pindex = gcs_read_dataframe(get_index_location(csv_name, remote=True))
        getattr(pindex, writer)(local_index, index=False)

    # for backwards compatibility:
    drop_cols = ["affinity_class", "pKd_pKi_pIC50", "tmp_chain2", "pinder_af2_hard"]
    for c in drop_cols:
        if c in pindex.columns:
            pindex.drop(columns=c, inplace=True)

    pindex = fix_index(
        pindex,
        index_fields=list(IndexEntry.__annotations__.keys()),
        cast_types=cast_types,
    )
    return pindex


@lru_cache
def get_metadata(
    csv_name: str = "metadata.parquet",
    update: bool = False,
    extra_glob: str = "metadata-*.csv.gz",
    extra_data: SupplementaryData | tuple[SupplementaryData] | tuple[()] = (),
) -> pd.DataFrame:
    """Retrieves the Pinder index metadata as a pandas DataFrame.

    If the metadata is not already loaded, it reads from the local CSV/Parquet file,
    or downloads it if not present.

    Also attempts to read extra metadata. We assume that all extra CSV
    files contain an `id` field.

    Parameters:
        csv_name (str): The name of the CSV file to load. Defaults to "metadata.parquet".
        update (bool): Whether to force update index on disk even if it exists.
            Default is False.
        extra_glob (str): The pattern to match extra metadata CSV files. Defaults to
            "metadata-*.csv.gz"

    Returns:
        pd.DataFrame: The Pinder metadata as a DataFrame.
    """
    local_metadata = Path(get_index_location(csv_name))
    if local_metadata.suffix in [".csv", ".gz"]:
        reader = pd.read_csv
        writer = "to_csv"
        cast_types = True
        # reader_kwargs = {}
    elif local_metadata.suffix == ".parquet":
        reader = pd.read_parquet
        writer = "to_parquet"
        cast_types = False
        # reader_kwargs = {"engine": "pyarrow"}
    else:
        raise ValueError(
            f"Unsupported file type for metadata: {local_metadata.suffix}! Use csv or parquet."
        )
    if local_metadata.is_file() and not update:
        metadata = reader(local_metadata)
    else:
        metadata = gcs_read_dataframe(get_index_location(csv_name, remote=True))
        getattr(metadata, writer)(local_metadata, index=False)

    # Backwards compatibility with old column names
    meta_cols = set(metadata.columns)
    if (
        len(
            {"number_of_components_1", "number_of_components_2"}.intersection(meta_cols)
        )
        != 2
    ):
        metadata.rename(
            {
                "number_of_components1": "number_of_components_1",
                "number_of_components2": "number_of_components_2",
            },
            axis=1,
            inplace=True,
        )
    if len({"length_resolved_1", "length_resolved_2"}.intersection(meta_cols)) != 2:
        metadata.loc[:, "length_resolved_1"] = metadata.length1
        metadata.loc[:, "length_resolved_2"] = metadata.length2

    metadata = fix_metadata(
        metadata,
        metadata_fields=list(MetadataEntry.__annotations__.keys()),
        cast_types=cast_types,
    )

    # Try to get extra metadata, if it exists
    if not isinstance(extra_data, tuple):
        extra_data = (extra_data,)
    extra_metadata = get_extra_metadata(
        get_pinder_location(),
        get_pinder_bucket_root(),
        glob_pattern=extra_glob,
        update=update,
        extra_data=extra_data,
    )
    if extra_metadata is not None and not extra_metadata.empty:
        # Only keep non-common columns + id column to merge on to prevent conflicts
        common_cols = set(extra_metadata.columns).intersection(
            set(metadata.columns)
        ) - {"id"}
        if common_cols:
            extra_metadata.drop(columns=list(common_cols), inplace=True)
        metadata = pd.merge(metadata, extra_metadata, on="id", how="left")

    return metadata


@lru_cache
def get_extra_metadata(
    local_location: str,
    remote_location: str,
    glob_pattern: str = "metadata-*.csv.gz",
    extra_data: tuple[SupplementaryData] | tuple[()] = (),
    update: bool = False,
) -> pd.DataFrame | None:
    """Retrieves extra metadata as a pandas DataFrame.

    If the metadata is not already loaded, it reads from local CSV files,
    or downloads them if not present.

    We assume that all CSV files contain an `id` field.

    Parameters:
        local_location (str):
            The filepath to the local directory containing CSV files.
        remote_location (str):
            The filepath to the remote location containing CSV files.
        glob_pattern (str): The pattern to match extra metadata CSV files. Defaults to
            "metadata-*.csv.gz"
        update (bool): Whether to force update index on disk even if it exists.
            Default is False.

    Returns:
        pd.DataFrame: The Pinder metadata as a DataFrame.
    """
    local_metadata = [p for p in Path(local_location).glob(glob_pattern) if p.is_file()]

    metadata_dfs = []
    if local_metadata and not update:
        metadata_dfs = [pd.read_csv(meta) for meta in local_metadata if meta.is_file()]
    else:
        gs = Gsutil()
        fs = gcsfs.GCSFileSystem(token="anon")
        metadata_dfs = [
            gcs_read_dataframe(meta, fs=fs)
            for meta in gs.ls(remote_location, recursive=False)
            if fnmatch(meta.name, glob_pattern)
        ]
    if extra_data:
        for data in extra_data:
            supp = get_supplementary_data(data, update=update)
            metadata_dfs.append(supp)

    if metadata_dfs:
        return reduce(partial(pd.merge, on="id", how="outer"), metadata_dfs)
    else:
        return None


class SupplementaryData(str, Enum):
    sequence_database = "sequence_database.parquet"
    supplementary_metadata = "supplementary_metadata.parquet"
    entity_metadata = "entity_metadata.parquet"
    chain_metadata = "chain_metadata.parquet"
    ecod_metadata = "ecod_metadata.parquet"
    enzyme_classification_metadata = "enzyme_classification_metadata.parquet"
    interface_annotations = "interface_annotations.parquet"
    sabdab_metadata = "sabdab_metadata.parquet"
    monomer_neff = "monomer_neff.parquet"
    paired_neff = "test_split_paired_neffs.parquet"
    transient_interface_metadata = "transient_interface_metadata.parquet"
    ialign_split_similarity_labels = "ialign_split_similarity_labels.parquet"


@lru_cache
def get_supplementary_data(
    supplementary_data: SupplementaryData, update: bool = False
) -> pd.DataFrame:
    """Retrieves supplementary data file exposed in the Pinder dataset as a pandas DataFrame.
    If the data is not already loaded, it reads from the local Parquet file, or downloads it if not present.

    Parameters:
        supplementary_data (SupplementaryData): The name of the Parquet file to load.
        update (bool): Whether to force update data on disk even if it exists.
            Default is False.

    Returns:
        pd.DataFrame: The requested supplementary Pinder data as a DataFrame.
    """
    if not isinstance(supplementary_data, SupplementaryData):
        supplementary_data = SupplementaryData[supplementary_data]
    local_index = Path(get_index_location(supplementary_data.value))
    if local_index.suffix == ".parquet":
        reader = pd.read_parquet
        writer = "to_parquet"
    else:
        raise ValueError(
            f"Unsupported file type for sequence database: {local_index.suffix}! Use parquet."
        )
    if local_index.is_file() and not update:
        data = reader(local_index)
    else:
        data = gcs_read_dataframe(
            get_index_location(supplementary_data.value, remote=True)
        )
        getattr(data, writer)(local_index, index=False)

    return data


@lru_cache
def get_sequence_database(
    pqt_name: str = "sequence_database.parquet", update: bool = False
) -> pd.DataFrame:
    """Retrieves sequences for all PDB files in the Pinder dataset (including dimers, bound and unbound monomers) as a pandas DataFrame.
    If the database is not already loaded, it reads from the local Parquet file, or downloads it if not present.

    Parameters:
        pqt_name (str): The name of the Parquet file to load. Defaults to "sequence_database.parquet".
        update (bool): Whether to force update sequence database on disk even if it exists.
            Default is False.

    Returns:
        pd.DataFrame: The Pinder sequence database as a DataFrame.
    """
    seq_db = get_supplementary_data(SupplementaryData(pqt_name), update=update)
    return seq_db


def download_dataset(skip_inflation: bool = False) -> None:
    """Downloads the Pinder dataset archives (PDBs and mappings) and optionally inflates them.

    Parameters:
        skip_inflation (bool):
            If True, the method will skip inflating (unzipping) the
            downloaded archives. Defaults to False.

    Note:
        Required disk space to download the full Pinder dataset:

        .. code-block:: text

            # compressed
            144G    pdbs.zip
            149M    test_set_pdbs.zip
            6.8G    mappings.zip

            # unpacked
            672G    pdbs
            705M    test_set_pdbs
            25G     mappings

    """
    root = get_pinder_location()
    expected_arch_files: dict[str, int] = {
        "pdbs": pc.PDB_FILE_COUNT,
        "test_set_pdbs": pc.TEST_SET_PDB_FILE_COUNT,
        "mappings": pc.MAPPING_FILE_COUNT,
    }
    remote_archs = []
    local_paths = []
    archive_paths = []
    for arch_name, n_expected in expected_arch_files.items():
        local_dir = root / arch_name
        local_arch = root / f"{arch_name}.zip"
        remote_arch = get_pinder_bucket_root() + f"/{arch_name}.zip"
        if local_dir.is_dir():
            ext = "." + arch_name[0:-1]
            n_existing = len(list(local_dir.glob(ext)))
        else:
            n_existing = 0

        # Only unzip archive if contents are incomplete
        unzip = n_existing < n_expected
        # Only download archive if directory doesn't exist or contents are incomplete
        download = unzip and not local_arch.is_file()
        if download:
            remote_archs.append(remote_arch)
            local_paths.append(local_arch)
        else:
            log.info(f"Archive {local_arch} exists, skipping download...")
        if unzip:
            archive_paths.append(local_arch)
        else:
            log.info(f"Skipping inflation, {local_arch} contents are valid...")

    if remote_archs:
        gs = Gsutil()
        gs.cp_paired_paths(remote_archs, local_paths)

    if skip_inflation:
        return
    for local_arch in archive_paths:
        if local_arch.is_file():
            with ZipFile(local_arch, "r") as archive:
                for member in tqdm(archive.infolist(), desc=f"Extracting {local_arch}"):
                    try:
                        archive.extract(member, local_arch.parent)
                    except zip_error as e:
                        log.error(str(e))
            local_arch.unlink()


def get_arg_parser_args(
    argv: list[str] | None = None,
    title: str = "Download latest pinder dataset to disk",
) -> dict[str, str | bool]:
    """The command-line arg parser for different pinder data downloads and updates.

    It accepts command-line arguments to specify the base directory, release version,
    and whether to skip inflating the compressed archives.

    Parameters:
        argv (Optional[List[str]]): The command-line arguments. If None, sys.argv is used.

    """
    import sys
    from argparse import ArgumentParser

    # Check if XDG_DATA_HOME is defined, if not fallback to expected
    default_loc = os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")
    # Prefer checking PINDER_BASE_DIR, if not set then use default location
    default_base_dir = os.environ.get("PINDER_BASE_DIR", default_loc)
    parser = ArgumentParser(prog=title)
    parser.add_argument(
        "--pinder_base_dir",
        help="specify a non-default pinder base directory",
        required=False,
        default=default_base_dir,
    )
    parser.add_argument(
        "--pinder_release",
        help="specify a pinder dataset version",
        required=False,
        default="2024-02",
    )
    parser.add_argument(
        "--skip_inflation",
        help="if passed, will only download the compressed archives without unpacking",
        action="store_true",
        default=False,
    )
    vargs = vars(parser.parse_args(argv if argv is not None else sys.argv[1:]))
    base_dir = str(Path(vargs["pinder_base_dir"]).absolute())
    release = str(vargs["pinder_release"])
    os.environ["PINDER_BASE_DIR"] = base_dir
    os.environ["PINDER_RELEASE"] = release
    log.info(f"Using {base_dir} as base directory for Pinder {release}")
    return vargs


def download_pinder_cmd(argv: list[str] | None = None) -> None:
    """The command-line interface for downloading the latest Pinder dataset to disk.

    It accepts command-line arguments to specify the base directory, release version,
    and whether to skip inflating the compressed archives.

    Parameters:
        argv (Optional[List[str]]): The command-line arguments. If None, sys.argv is used.

    Note:
        Required disk space to download the full Pinder dataset:

        .. code-block:: text

            # compressed
            144G    pdbs.zip
            149M    test_set_pdbs.zip
            6.8G    mappings.zip

            # unpacked
            672G    pdbs
            705M    test_set_pdbs
            25G     mappings

    """
    vargs = get_arg_parser_args(argv)
    skip_inflation = bool(vargs.pop("skip_inflation", False))
    download_dataset(skip_inflation=skip_inflation)


def update_index_cmd(argv: list[str] | None = None) -> None:
    """The command-line interface for downloading the latest Pinder index to disk.

    It accepts command-line arguments to specify the base directory and release version.

    Parameters:
        argv (Optional[List[str]]): The command-line arguments. If None, sys.argv is used.

    """

    vargs = get_arg_parser_args(argv, "Download latest pinder index to disk")
    get_index(update=True)
    get_metadata(update=True)


def get_missing_blobs(prefix: str) -> tuple[list[str], list[Path]]:
    client = Client.create_anonymous_client()
    bucket = client.bucket("pinder")
    release = get_pinder_bucket_root().split("pinder/")[-1]
    ext = "." + prefix[0:3]

    local_dir = get_pinder_location() / prefix
    local_names = {f.name for f in local_dir.glob(f"*{ext}")}
    missing_blobs = [
        b
        for b in tqdm(bucket.list_blobs(prefix=f"{release}/{prefix}/"))
        if ext in b.name and Path(b.name).name not in local_names
    ]
    remote_paths: list[str] = []
    local_paths: list[Path] = []
    for blob in missing_blobs:
        name = Path(blob.name).name
        remote = get_pinder_bucket_root() + f"/{prefix}/{name}"
        local = local_dir / name
        remote_paths.append(remote)
        local_paths.append(local)
    return remote_paths, local_paths


def sync_pinder_structure_data(argv: list[str] | None = None) -> None:
    """The command-line interface for syncing any structural data files missing on disk.

    It accepts command-line arguments to specify the base directory and release version.

    Parameters:
        argv (Optional[List[str]]): The command-line arguments. If None, sys.argv is used.

    """

    vargs = get_arg_parser_args(
        argv, "Sync missing pinder structural data files to disk"
    )
    remote_paths = []
    local_paths = []
    for prefix in ["pdbs", "test_set_pdbs", "mappings"]:
        remote, local = get_missing_blobs(prefix)
        remote_paths.extend(remote)
        local_paths.extend(local)

    if remote_paths:
        log.info(f"Found {len(remote_paths)} outdated files. Starting download...")
        gs = Gsutil()
        gs.cp_paired_paths(remote_paths, local_paths)
    else:
        log.info("All local files are up to date!")
    return None


class IndexEntry(BaseModel):
    """Pydantic model representing a single entry in the Pinder index.

    Stores all associated metadata for a particular dataset entry as attributes.

    Parameters:
        split (str):
            The type of data split (e.g., 'train', 'test').
        id (str):
            The unique identifier for the dataset entry.
        pdb_id (str):
            The PDB identifier associated with the entry.
        cluster_id (str):
            The cluster identifier associated with the entry.
        cluster_id_R (str):
            The cluster identifier associated with receptor dimer body.
        cluster_id_L (str):
            The cluster identifier associated with ligand dimer body.
        pinder_s (bool):
            Flag indicating if the entry is part of the Pinder-S dataset.
        pinder_xl (bool):
            Flag indicating if the entry is part of the Pinder-XL dataset.
        pinder_af2 (bool):
            Flag indicating if the entry is part of the Pinder-AF2 dataset.
        uniprot_R (str):
            The UniProt identifier for the receptor protein.
        uniprot_L (str):
            The UniProt identifier for the ligand protein.
        holo_R_pdb (str):
            The PDB identifier for the holo form of the receptor protein.
        holo_L_pdb (str):
            The PDB identifier for the holo form of the ligand protein.
        predicted_R_pdb (str):
            The PDB identifier for the predicted structure of the receptor protein.
        predicted_L_pdb (str):
            The PDB identifier for the predicted structure of the ligand protein.
        apo_R_pdb (str):
            The PDB identifier for the apo form of the receptor protein.
        apo_L_pdb (str):
            The PDB identifier for the apo form of the ligand protein.
        apo_R_pdbs (str):
            The PDB identifiers for the apo forms of the receptor protein.
        apo_L_pdbs (str):
            The PDB identifiers for the apo forms of the ligand protein.
        holo_R (bool):
            Flag indicating if the holo form of the receptor protein is available.
        holo_L (bool):
            Flag indicating if the holo form of the ligand protein is available.
        predicted_R (bool):
            Flag indicating if the predicted structure of the receptor protein is available.
        predicted_L (bool):
            Flag indicating if the predicted structure of the ligand protein is available.
        apo_R (bool):
            Flag indicating if the apo form of the receptor protein is available.
        apo_L (bool):
            Flag indicating if the apo form of the ligand protein is available.
        apo_R_quality (str):
            Classification of apo receptor pairing quality. Can be `high, low, ''`.
            All test and val are labeled high. Train split is broken into `high` and `low`,
            depending on whether the pairing was produced with a low-confidence quality/eval metrics
            or `high` if the same metrics were used as for train and val.
            If no pairing exists, it is labeled with an empty string.
        apo_L_quality (str):
            Classification of apo ligand pairing quality. Can be `high, low, ''`.
            All test and val are labeled high. Train split is broken into `high` and `low`,
            depending on whether the pairing was produced with a low-confidence quality/eval metrics
            or `high` if the same metrics were used as for train and val.
            If no pairing exists, it is labeled with an empty string.
        chain1_neff (float):
            The Neff value for the first chain in the protein complex.
        chain2_neff (float):
            The Neff value for the second chain in the protein complex.
        chain_R (str):
            The chain identifier for the receptor protein.
        chain_L (str):
            The chain identifier for the ligand protein.
        contains_antibody (bool):
            Flag indicating if the protein complex contains an antibody as per SAbDab.
        contains_antigen (bool):
            Flag indicating if the protein complex contains an antigen as per SAbDab.
        contains_enzyme (bool):
            Flag indicating if the protein complex contains an enzyme as per EC ID number.
    """

    model_config = ConfigDict(extra="forbid")

    split: str
    id: str
    pdb_id: str
    cluster_id: str
    cluster_id_R: str
    cluster_id_L: str
    pinder_s: bool
    pinder_xl: bool
    pinder_af2: bool
    uniprot_R: str
    uniprot_L: str
    holo_R_pdb: str
    holo_L_pdb: str
    predicted_R_pdb: str
    predicted_L_pdb: str
    apo_R_pdb: str
    apo_L_pdb: str
    apo_R_pdbs: str
    apo_L_pdbs: str
    holo_R: bool
    holo_L: bool
    predicted_R: bool
    predicted_L: bool
    apo_R: bool
    apo_L: bool
    apo_R_quality: str
    apo_L_quality: str
    chain1_neff: float
    chain2_neff: float
    chain_R: str
    chain_L: str
    contains_antibody: bool
    contains_antigen: bool
    contains_enzyme: bool

    def pdb_path(self, pdb_name: str) -> str:
        """Constructs the relative path for a given PDB file within the Pinder dataset.

        Parameters:
            pdb_name (str): The name of the PDB file.

        Returns:
            str: The relative path as a string.
        """
        if pdb_name == "":
            return ""
        if self.test_system and (
            pdb_name.endswith("-R.pdb") or pdb_name.endswith("-L.pdb")
        ):
            pdbs_dir = "test_set_pdbs"
        else:
            pdbs_dir = "pdbs"
        return f"{pdbs_dir}/{pdb_name}"

    def mapping_path(self, pdb_name: str) -> str:
        if pdb_name == "":
            return ""
        return f"mappings/{pdb_name.replace('.pdb', '.parquet')}"

    @property
    def pdb_paths(self) -> dict[str, str | list[str]]:
        return {
            "native": self.pdb_path(self.pinder_pdb),
            "holo_R": self.pdb_path(self.holo_R_pdb),
            "holo_L": self.pdb_path(self.holo_L_pdb),
            "predicted_R": self.pdb_path(self.predicted_R_pdb),
            "predicted_L": self.pdb_path(self.predicted_L_pdb),
            "apo_R": self.pdb_path(self.apo_R_pdb),
            "apo_L": self.pdb_path(self.apo_L_pdb),
            "apo_R_alt": [
                self.pdb_path(p) for p in self.apo_R_alt if isinstance(p, str)
            ],
            "apo_L_alt": [
                self.pdb_path(p) for p in self.apo_L_alt if isinstance(p, str)
            ],
        }

    @property
    def mapping_paths(self) -> dict[str, str | list[str]]:
        return {
            "holo_R": self.mapping_path(self.holo_R_pdb),
            "holo_L": self.mapping_path(self.holo_L_pdb),
            "apo_R": self.mapping_path(self.apo_R_pdb),
            "apo_L": self.mapping_path(self.apo_L_pdb),
            "apo_R_alt": [
                self.mapping_path(p) for p in self.apo_R_alt if isinstance(p, str)
            ],
            "apo_L_alt": [
                self.mapping_path(p) for p in self.apo_L_alt if isinstance(p, str)
            ],
        }

    @property
    def pinder_id(self) -> str:
        return self.id

    @property
    def pinder_pdb(self) -> str:
        return self.id + ".pdb"

    @property
    def apo_R_alt(self) -> list[str] | list[None]:
        return [f for f in self.apo_R_pdbs.split(";") if f != self.apo_R_pdb]

    @property
    def apo_L_alt(self) -> list[str] | list[None]:
        return [f for f in self.apo_L_pdbs.split(";") if f != self.apo_L_pdb]

    @property
    def homodimer(self) -> bool:
        return self.uniprot_R == self.uniprot_L

    @property
    def test_system(self) -> bool:
        return self.split == "test"


class MetadataEntry(BaseModel):
    """Pydantic model representing a single entry in the Pinder metadata.

    Stores detailed metadata for a particular dataset entry as attributes.

    Parameters:
        id (str):
            The unique identifier for the PINDER entry.
            It follows the convention `<Receptor>--<Ligand>`, where `<Receptor>` is `<pdbid>__<chain_1>_<uniprotid>` and
            `<Ligand>` is `<pdbid>__<chain_2><uniprotid>`.
        entry_id (str):
            The RCSB entry identifier associated with the PINDER entry.
        method (str):
            The experimental method for structure determination (XRAY, CRYO-EM, etc.).
        date (str):
            Date of deposition into RCSB PDB.
        release_date (str):
            Date of initial public release in RCSB PDB.
        resolution (float):
            The resolution of the experimental structure.
        label (str):
            Classification of the interface as likely to be biologically-relevant or a crystal contact, annotated using PRODIGY-cryst.
            PRODIGY-cryst uses machine learning to compute bio-relevant/crystal contact propensity based on Intermolecular contact types and Interfacial link density.
        probability (float):
            Probability that the protein complex is a true biological complex.
        chain1_id (str):
            The Receptor chain identifier associated with the dimer entry. Should all be chain 'R'.
        chain2_id (str):
            The Ligand chain identifier associated with the dimer entry. Should all be chain 'L'.
        assembly (int):
            Which bioassembly is used to derive the structure. 1, 2, 3 means first, second, and third assembly, respectively.
            All PINDER dimers are derived from the first biological assembly.
        assembly_details (str):
            How the bioassembly information was derived. Is it author-defined or from another source.
        oligomeric_details (str):
            Description of the oligomeric state of the protein complex.
        oligomeric_count (int):
            The oligomeric count associated with the dataset entry.
        biol_details (str):
            The biological assembly details associated with the dataset entry.
        complex_type (str):
            The type of the complex in the dataset entry (homomer or heteromer).
        chain_1 (str):
            New chain id generated post-bioassembly generation, to reflect the asym_id of the bioassembly and also to ensure that there is no collision of chain ids, for example in homooligomers (receptor chain).
        asym_id_1 (str):
            The first asymmetric identifier (author chain ID)
        chain_2 (str):
            New chain id generated post-bioassembly generation, to reflect the asym_id of the bioassembly and also to ensure that there is no collision of chain ids, for example in homooligomers (ligand chain).
        asym_id_2 (str):
            The second asymmetric identifier (author chain ID)
        length1 (int):
            The number of amino acids in the first (receptor) chain.
        length2 (int):
            The number of amino acids in the second (ligand) chain.
        length_resolved_1 (int):
            The structurally resolved (CA) length of the first (receptor) chain in amino acids.
        length_resolved_2 (int):
            The structurally resolved (CA) length of the second (ligand) chain in amino acids.
        number_of_components_1 (int):
            The number of connected components in the first (receptor) chain (contiguous structural fragments)
        number_of_components_2 (int):
            The number of connected components in the second (receptor) chain (contiguous structural fragments)
        link_density (float):
            Density of contacts at the interface as reported by PRODIGY-cryst.
            Interfacial link density is defined as the number of interfacial contacts normalized by the maximum possible number of pairwise contacts for that interface.
            Values range between 0 and 1, with higher values indicating a denser contact network at the interface.
        planarity (float):
            Defined as the deviation of interfacial Cα atoms from the fitted plane.
            This interface characteristic quantifies interfacial shape complementarity.
            Transient complexes have smaller and more planar interfaces than permanent and structural scaffold complexes.
        max_var_1 (float):
            The maximum variance of coordinates projected onto the largest principal component.
            This allows the detection of long end-to-end stacked complexes, likely to be repetitive with small interfaces (receptor chain).
        max_var_2 (float):
            The maximum variance of coordinates projected onto the largest principal component.
            This allows the detection of long end-to-end stacked complexes, likely to be repetitive with small interfaces (ligand chain).
        num_atom_types (int):
            Number of unique atom types in structure. This is an important annotation to identify complexes with only Cα or backbone atoms.
        n_residue_pairs (int):
            The number of residue pairs at the interface.
        n_residues (int):
            The number of residues at the interface.
        buried_sasa (float):
            The buried solvent accessible surface area upon complex formation.
        intermolecular_contacts (int):
            The total number of intermolecular contacts (pair residues with any atom within a 5Å distance cutoff) at the interface.
            Annotated using PRODIGY-cryst.
        charged_charged_contacts (int):
            Denotes intermolecular contacts between any of the charged amino acids (E, D, H, K).
            Annotated using PRODIGY-cryst.
        charged_polar_contacts (int):
            Denotes intermolecular contacts between charged amino acids (E, D, H, K, R) and polar amino acids (N, Q, S, T).
            Annotated using PRODIGY-cryst.
        charged_apolar_contacts (int):
            Denotes intermolecular contacts between charged amino acids (E, D, H, K) and apolar amino acids (A, C, G, F, I, M, L, P, W, V, Y).
            Annotated using PRODIGY-cryst.
        polar_polar_contacts (int):
            Denotes intermolecular contacts between any of the charged amino acids (N, Q, S, T).
            Annotated using PRODIGY-cryst.
        apolar_polar_contacts (int):
            Denotes intermolecular contacts between apolar amino acids (A, C, G,F, I, M, L, P, W, V, Y) and polar amino acids (N, Q, S, T).
            Annotated using PRODIGY-cryst.
        apolar_apolar_contacts (int):
            Denotes intermolecular contacts between any of the charged amino acids (A, C, G, F, I, M, L, P, W, V, Y).
            Annotated using PRODIGY-cryst.
        interface_atom_gaps_4A (int):
            Number of interface atoms within a 4Å radius of a residue gap.
            A Gap is determined by residue numbering; regions where one or more of the expected residue index is missing is marked as a gap.
        missing_interface_residues_4A (int):
            Number of interface residues within a 4Å radius of a residue gap.
            A Gap is determined by residue numbering; regions where one or more of the expected residue index is missing is marked as a gap.
        interface_atom_gaps_8A (int):
            Number of interface atoms within an 8Å radius of a residue gap.
            A Gap is determined by residue numbering; regions where one or more of the expected residue index is missing is marked as a gap.
        missing_interface_residues_8A (int):
            Number of interface residues within an 8Å radius of a residue gap.
            A Gap is determined by residue numbering; regions where one or more of the expected residue index is missing is marked as a gap.
        entity_id_R (int):
            The RCSB PDB `entity_id` corresponding to the receptor dimer chain.
        entity_id_L (int):
            The RCSB PDB `entity_id` corresponding to the ligand dimer chain.
        pdb_strand_id_R (str):
            The RCSB PDB `pdb_strand_id` (author chain) corresponding to the receptor dimer chain.
        pdb_strand_id_L (str):
            The RCSB PDB `pdb_strand_id` (author chain) corresponding to the ligand dimer chain.
        ECOD_names_R (str):
            The RCSB-derived ECOD domain protein family name(s) corresponding to the receptor dimer chain. If multiple ECOD domain annotations
            were found, the domains are delimited with a comma.
        ECOD_names_L (str):
            The RCSB-derived ECOD domain protein family name(s) corresponding to the ligand dimer chain. If multiple ECOD domain annotations
            were found, the domains are delimited with a comma.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    entry_id: str
    method: str
    date: str
    release_date: str
    resolution: float
    label: str
    probability: float
    chain1_id: str
    chain2_id: str
    assembly: int
    assembly_details: str
    oligomeric_details: str
    oligomeric_count: int
    biol_details: str
    complex_type: str
    chain_1: str
    asym_id_1: str
    chain_2: str
    asym_id_2: str
    length1: int
    length2: int
    length_resolved_1: int
    length_resolved_2: int
    number_of_components_1: int
    number_of_components_2: int
    link_density: float
    planarity: float
    max_var_1: float
    max_var_2: float
    num_atom_types: int
    n_residue_pairs: int
    n_residues: int
    buried_sasa: float
    intermolecular_contacts: int
    charged_charged_contacts: int
    charged_polar_contacts: int
    charged_apolar_contacts: int
    polar_polar_contacts: int
    apolar_polar_contacts: int
    apolar_apolar_contacts: int
    interface_atom_gaps_4A: int
    missing_interface_residues_4A: int
    interface_atom_gaps_8A: int
    missing_interface_residues_8A: int
    entity_id_R: int
    entity_id_L: int
    pdb_strand_id_R: str
    pdb_strand_id_L: str
    ECOD_names_R: str
    ECOD_names_L: str

    @property
    def pinder_id(self) -> str:
        return self.id


def fix_index(
    df_index: pd.DataFrame, index_fields: list[str], cast_types: bool = True
) -> pd.DataFrame:
    """Fix the index dataframe according to the IndexEntry schema"""
    # fix missing values
    for field in index_fields:
        field_type = IndexEntry.__annotations__[field]
        if field in df_index.columns:
            df_index = _fill_missing_values(df_index, field, field_type, cast_types)
        else:
            fill_val = _get_default_value(field_type)
            log.error(
                f"Expected field {field} with type '{field_type}' missing from index! "
                f"Setting default value={fill_val}. Verify your index file!"
            )
            df_index[field] = fill_val
    if not cast_types and "category" not in df_index.dtypes.values:
        df_index = downcast_dtypes(df_index)
    return df_index


def fix_metadata(
    df_metadata: pd.DataFrame, metadata_fields: list[str], cast_types: bool = True
) -> pd.DataFrame:
    """Fix the metadata dataframe according to the MetadataEntry schema"""
    # fix resolution
    df_metadata["resolution"] = df_metadata["resolution"].fillna(0.0)
    df_metadata["resolution"] = df_metadata["resolution"].replace(".", 0.0)
    for field in metadata_fields:
        field_type = MetadataEntry.__annotations__[field]
        if field in df_metadata.columns:
            df_metadata = _fill_missing_values(
                df_metadata, field, field_type, cast_types
            )
        else:
            fill_val = _get_default_value(field_type)
            log.error(
                f"Expected field {field} with type '{field_type}' missing from metadata! "
                f"Setting default value={fill_val}. Verify your metadata file!"
            )
            df_metadata[field] = fill_val
    if not cast_types and "category" not in df_metadata.dtypes.values:
        df_metadata = downcast_dtypes(df_metadata)
    return df_metadata


def _get_default_value(field_type: str) -> int | float | str | bool | None:
    # Add missing columns to the dataframes
    default_vals: dict[str, int | float | str | bool] = {
        "int": 0,
        "float": 0.0,
        "str": "",
        "bool": False,
    }
    fill_val = default_vals.get(field_type)
    return fill_val


def _fill_missing_values(
    df: pd.DataFrame,
    field: str,
    field_type: str,
    cast_types: bool = True,
) -> pd.DataFrame:
    # Fill missing values w/default for the field type and cast column to field_type
    fill_val = _get_default_value(field_type)
    if df[field].isna().any():
        print(
            f"Expected field {field} with type '{field_type}' contains NaN values and will be filled with value={fill_val}. "
            "Verify your index/metadata file!"
        )
        df[field] = df[field].fillna(fill_val)
    if cast_types:
        df[field] = df[field].astype(field_type)
    return df


def downcast_dtypes(
    df: pd.DataFrame,
    str_as_cat: bool = True,
) -> pd.DataFrame:
    str_types = ["O", "str"]
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    np_int_types: list[np.typing.DTypeLike] = [np.int16, np.int32, np.int64]
    np_float_types: list[np.typing.DTypeLike] = [np.float16, np.float32, np.float64]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in str_types and str_as_cat:
            # Only cast to category if the number of categories is < 68% of the data
            # otherwise the memory is going to be higher than object dtype.
            cat_frac = len(set(df[col])) / df.shape[0]
            str_dtype = "object" if cat_frac > 0.68 else "category"
            log.debug(f"Casting {col} from {col_type} to {str_dtype}")
            df[col] = df[col].astype(str_dtype)
        if "int" in str(col_type) or "float" in str(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if "int" in str(col_type):
                np_types = np_int_types
                type_info_fn = np.iinfo
            else:
                np_types = np_float_types
                type_info_fn = np.finfo

            for np_type in np_types:
                type_info = type_info_fn(np_type)
                if c_min > type_info.min and c_max < type_info.max:
                    log.debug(f"Casting {col} from {col_type} to {np_type}")
                    df[col] = df[col].astype(np_type)
                    break
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    log.debug(
        "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
            end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )
    return df


def set_mapping_column_types(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Set the column types for the mapping dataframe to avoid 1.0 2.0 "integers".
    Int64 is supposed to support NaNs. This function requires Pandas 1.0+.

    Parameters
    ----------
    mapping_df : pd.DataFrame
        The dataframe whose column types are to be set.
    """
    # Contain NaN or undefined
    # ['resi_auth', 'one_letter_code_uniprot', 'resi_uniprot', 'uniprot_acc']
    map_dtypes = {
        "entry_id": "category",
        "entity_id": "int16",
        "asym_id": "category",
        "pdb_strand_id": "category",
        "resi": "int32",
        "resi_pdb": "Int32",
        "resi_auth": "object",
        "resn": "category",
        "one_letter_code_can": "category",
        "resolved": "int8",
        "one_letter_code_uniprot": "category",
        "resi_uniprot": "Int32",
        "uniprot_acc": "category",
        "chain": "category",
    }
    try:
        if "resi_uniprot" not in mapping_df.columns:
            mapping_df.loc[:, "resi_uniprot"] = pd.NA
        if "uniprot_acc" not in mapping_df.columns:
            mapping_df.loc[:, "uniprot_acc"] = ""
        if "one_letter_code_uniprot" not in mapping_df.columns:
            mapping_df.loc[:, "one_letter_code_uniprot"] = ""
        mapping_df.loc[mapping_df.resi_uniprot == "", "resi_uniprot"] = pd.NA
        for col, dtype in map_dtypes.items():
            mapping_df[col] = mapping_df[col].astype(dtype)
        mapping_df = mapping_df[list(map_dtypes.keys())].copy()
    except Exception as e:
        log.error(f"Failed to cast mapping column types: {e}")
    return mapping_df
