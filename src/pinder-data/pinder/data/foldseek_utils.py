from __future__ import annotations
from pathlib import Path
from subprocess import check_call
from string import digits

import pandas as pd
from tqdm import tqdm

from pinder.core.index.system import PinderSystem
from pinder.core.loader.structure import Structure
from pinder.core.utils.log import setup_logger
from pinder.core.utils.paths import parallel_copy_files
from pinder.core.utils.process import process_map
from pinder.data.config import GraphConfig, FoldseekConfig, MMSeqsConfig
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.pipeline import scatter


log = setup_logger(__name__)


FOLDSEEK_FIELDS: str = ",".join(
    [
        "query",
        "target",
        "{}",
        "qstart",
        "qend",
        "qlen",
        "tstart",
        "tend",
        "tlen",
        "alnlen",
    ]
)


MMSEQ2_FIELDS: str = ",".join(
    [
        "query",
        "target",
        "{}",
        "qstart",
        "qend",
        "qlen",
        "tstart",
        "tend",
        "tlen",
        "alnlen",
    ]
)


def fasta2dict(fasta_file: Path) -> dict[str, str]:
    fasta_dict: dict[str, str] = {}
    with open(fasta_file, "r") as f:
        fasta_txt = f.read()
    for record in fasta_txt.split(">"):
        if record == "":
            continue
        header, seq = record.strip().split("\n")
        fasta_dict[header] = seq
    return fasta_dict


def create_fasta_from_systems(
    systems: list[PinderSystem], fasta_file: Path | str
) -> None:
    unique_seqs = set()
    for system in systems:
        unique_seqs.add(system.holo_ligand.fasta)
        unique_seqs.add(system.holo_receptor.fasta)
    with open(fasta_file, "w") as f:
        f.write("\n".join(unique_seqs))


def extract_fasta_from_pdb(pdb_file: Path) -> str:
    struct = Structure(pdb_file)
    fasta: str = struct.fasta
    return fasta


def parallel_extract_fasta(
    pdb_files: list[Path],
    max_workers: int | None = None,
    parallel: bool = True,
) -> list[str]:
    """Extract fasta-formatted sequences from a collection of PDB files in parallel.
    Operates in parallel and assumes that source files all exist.

    Parameters:
    pdb_files (list[Path]): List of PDB files to extract fasta strings for.
    max_workers (int, optional): Limit number of parallel processes spawned to `max_workers`.

    """
    fasta_seqs: list[str] = process_map(
        extract_fasta_from_pdb, pdb_files, parallel=parallel, max_workers=max_workers
    )
    return fasta_seqs


def create_fasta_from_foldseek_inputs(
    foldseek_dir: Path,
    fasta_file: Path | str,
    max_workers: int | None = None,
    use_cache: bool = True,
    parallel: bool = True,
) -> None:
    fasta_file = Path(fasta_file)
    if use_cache and fasta_file.is_file():
        return
    foldseek_pdbs = list(foldseek_dir.glob("*.pdb"))
    pdb_seqs = parallel_extract_fasta(
        pdb_files=foldseek_pdbs,
        parallel=parallel,
        max_workers=max_workers,
    )
    all_seqs = "\n".join(pdb_seqs)
    with open(fasta_file, "w") as f:
        f.write(all_seqs)


def create_foldseek_input_dir(
    index: Path | str,
    foldseek_dir: Path | str,
    pdb_dir: Path | str,
    use_cache: bool = True,
    max_workers: int | None = None,
    parallel: bool = True,
) -> None:
    foldseek_dir = Path(foldseek_dir)
    foldseek_dir.mkdir(exist_ok=True)
    pdb_dir = Path(pdb_dir)
    index_df = read_csv_non_default_na(index, dtype={"pdb_id": "str"})
    index_df.loc[:, "foldseek_L_pdb"] = [
        f"{pdb_id}_{ch_L.rstrip(digits)}.pdb"
        for pdb_id, ch_L in zip(*(index_df[c] for c in ["pdb_id", "chain_L"]))
    ]
    index_df.loc[:, "foldseek_R_pdb"] = [
        f"{pdb_id}_{ch_R.rstrip(digits)}.pdb"
        for pdb_id, ch_R in zip(*(index_df[c] for c in ["pdb_id", "chain_R"]))
    ]
    foldseek_R = index_df[["pdb_id", "holo_R_pdb", "foldseek_R_pdb"]].copy()
    foldseek_R.rename(
        {"holo_R_pdb": "holo_pdb", "foldseek_R_pdb": "foldseek_pdb"},
        axis=1,
        inplace=True,
    )
    foldseek_L = index_df[["pdb_id", "holo_L_pdb", "foldseek_L_pdb"]].copy()
    foldseek_L.rename(
        {"holo_L_pdb": "holo_pdb", "foldseek_L_pdb": "foldseek_pdb"},
        axis=1,
        inplace=True,
    )
    foldseek_inputs = (
        pd.concat([foldseek_R, foldseek_L])
        .drop_duplicates("foldseek_pdb")
        .reset_index(drop=True)
    )
    src_files = [pdb_dir / holo_pdb for holo_pdb in list(foldseek_inputs.holo_pdb)]
    dest_files = [
        foldseek_dir / fold_pdb for fold_pdb in list(foldseek_inputs.foldseek_pdb)
    ]
    parallel_copy_files(
        src_files,
        dest_files,
        use_cache=use_cache,
        max_workers=max_workers,
        parallel=parallel,
    )
    log.info(
        f"Created {len(list(foldseek_dir.glob('*.pdb')))} single chain PDB files in {foldseek_dir} for foldseek."
    )


def run_foldseek(
    input_dir: Path,
    output_dir: Path,
    target_db_dir: Path | None = None,
    config: FoldseekConfig = FoldseekConfig(),
) -> None:
    """Run foldseek easy-search on a directory of PDB structures.

    Parameters:
    input_dir (Path): Input directory for foldseek targets.
    output_dir (Path): The output directory to store foldseek alignments.
    target_db_dir (Optional[Path]): Optional target DB input directory for foldseek. If not specified, defaults to input_dir.
    config (FoldseekConfig): The configuration object containing foldseek parameters.

    """
    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    if target_db_dir is None:
        target_db_dir = input_dir
    output_file = Path(out_folder) / config.alignment_filename
    foldseek_command = [
        "foldseek",
        "easy-search",
        str(input_dir),
        str(target_db_dir),
        str(output_file),
        str("tmp"),
        "-s",
        f"{config.sensitivity}",
        "-e",
        f"{config.evalue}",
        "--max-seqs",
        f"{config.max_seqs}",
        "--alignment-type",
        f"{config.alignment_type}",
        "--format-output",
        FOLDSEEK_FIELDS.format(config.score_type),
    ]
    print(" ".join(foldseek_command))
    check_call(foldseek_command, stderr=open(out_folder / "foldseek_error.txt", "w"))


def run_mmseqs(
    input_fasta: Path,
    output_dir: Path,
    target_fasta: Path | None = None,
    use_cache: bool = True,
    config: MMSeqsConfig = MMSeqsConfig(),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_fasta is None:
        mmseqs_dir = output_dir / "mmseqs2"
        mmseqs_dir.mkdir(parents=True, exist_ok=True)
        target_fasta = input_fasta
    else:
        mmseqs_dir = output_dir

    if not input_fasta.is_file():
        log.warning(
            f"Input fasta {input_fasta} not found! "
            f"Attempting to generate from output_dir {output_dir}!"
        )
        create_fasta_from_foldseek_inputs(output_dir, input_fasta)
        log.info(f"Created {input_fasta} for mmseqs2.")

    output_file = mmseqs_dir / config.alignment_filename
    if output_file.is_file() and use_cache:
        return

    log.info(f"Running mmseqs easy-search on {output_dir}")
    command = [
        "mmseqs",
        "easy-search",
        str(input_fasta),
        str(target_fasta),
        str(output_file),
        str("tmp"),
        "-s",
        f"{config.sensitivity}",
        "-e",
        f"{config.evalue}",
        "--min-seq-id",
        f"{config.min_seq_id}",
        "--max-seqs",
        f"{config.max_seqs}",
        "--format-output",
        MMSEQ2_FIELDS.format(config.score_type),
    ]
    print(" ".join(command))
    check_call(command, stderr=open(str(mmseqs_dir / "mmseqs_error.txt"), "w"))


def create_dbs(
    db_root_path: Path | str,
    chains_path: Path | str,
    db_size: int = 50_000,
    max_workers: int | None = None,
    use_cache: bool = True,
    parallel: bool = True,
) -> None:
    db_path = Path(db_root_path)
    db_path.mkdir(exist_ok=True, parents=True)
    chains_path = Path(chains_path)
    chains = list(chains_path.glob("*.pdb"))
    n_chains = len(chains)
    # Split into chunks of max size = db_size
    chain_chunks = [
        chains[i * db_size : (i + 1) * db_size]
        for i in range((n_chains + db_size - 1) // db_size)
    ]
    src_files = []
    dest_files = []
    for i, chain_chunk in enumerate(chain_chunks):
        # Create flat list of src and dest files to copy to their respective db_subdir
        db_subdir = db_path / f"{i:05d}" / "db"
        db_subdir.mkdir(exist_ok=True, parents=True)
        src_files.extend(chain_chunk)
        dest_files.extend([db_subdir / chain.name for chain in chain_chunk])
    parallel_copy_files(
        src_files,
        dest_files,
        use_cache=use_cache,
        max_workers=max_workers,
        parallel=parallel,
    )


def run_db_vs_db(
    db_path: Path,
    i: int,
    j: int,
    config: FoldseekConfig = FoldseekConfig(),
    use_cache: bool = True,
) -> Path:
    subdir1 = db_path / f"{i:05d}" / "db"
    subdir2 = db_path / f"{j:05d}" / "db"
    output_dir = db_path / f"{i:05d}" / f"{j:05d}"
    alignments: Path = output_dir / config.alignment_filename
    if alignments.is_file() and use_cache:
        log.info(f"foldseek pair {i},{j} already present. Skipping...")
    else:
        run_foldseek(
            subdir1,
            output_dir,
            subdir2,
            config=config,
        )
    return alignments


def run_foldseek_db_pair(
    pinder_dir: Path,
    db_indices: tuple[int, int],
    foldseek_config: FoldseekConfig,
    use_cache: bool = True,
) -> None:
    foldseek_dir = pinder_dir / "foldseek"
    db_path = foldseek_dir / "foldseek_dbs"
    if len(db_indices) != 2:
        return
    i, j = db_indices
    log.info(f"Running foldseek DB {i}, {j}")
    al_file = run_db_vs_db(
        db_path,
        i,
        j,
        config=foldseek_config,
        use_cache=use_cache,
    )


def create_dbs_and_run(
    fold_db_path: Path | str,
    chains_path: Path | str,
    db_size: int = 50_000,
    config: FoldseekConfig = FoldseekConfig(),
) -> None:
    fold_db_path = Path(fold_db_path)
    chains_path = Path(chains_path)
    create_dbs(fold_db_path, chains_path, db_size)
    chains = list(chains_path.glob("*.pdb"))
    alignment_files = []
    for i, j in scatter.chunk_all_vs_all_indices(chains, db_size):
        al_file = run_db_vs_db(
            fold_db_path,
            i,
            j,
            config=config,
        )
        alignment_files.append(al_file)

    with open(fold_db_path / config.alignment_filename, "w") as f:
        for al_file in alignment_files:
            with open(al_file, "r") as al:
                f.write(al.read())
            al_file.unlink()


def setup_foldseek_dbs(
    pinder_dir: Path,
    foldseek_db_size: int = 50_000,
    use_cache: bool = True,
) -> None:
    foldseek_dir = pinder_dir / "foldseek"
    foldseek_dir.mkdir(exist_ok=True, parents=True)
    create_foldseek_input_dir(
        index=pinder_dir / "index.1.csv.gz",
        foldseek_dir=foldseek_dir,
        pdb_dir=pinder_dir / "pdbs",
        use_cache=use_cache,
    )
    db_path = foldseek_dir / "foldseek_dbs"
    create_dbs(db_path, foldseek_dir, db_size=foldseek_db_size, use_cache=use_cache)


def _collate_alignment_files(
    pinder_dir: Path,
    output_file: Path,
    db_path: Path,
    db_size: int = 50_000,
    alignment_filename: str = "alignment.txt",
) -> None:
    foldseek_dir = pinder_dir / "foldseek"
    chains = list(foldseek_dir.glob("*.pdb"))
    alignment_files = []
    for i, j in scatter.chunk_all_vs_all_indices(chains, db_size):
        output_dir = db_path / f"{i:05d}" / f"{j:05d}"
        alignment_file = output_dir / alignment_filename
        if alignment_file.is_file():
            alignment_files.append(alignment_file)

    log.info(f"Collating {len(alignment_files)} alignment files")
    with open(output_file, "w") as f:
        for al_file in alignment_files:
            with open(al_file, "r") as al:
                f.write(al.read())


def collate_foldseek_alignments(
    pinder_dir: Path,
    foldseek_db_size: int = 50_000,
    use_cache: bool = True,
    alignment_filename: str = "alignment.txt",
) -> None:
    foldseek_dir = pinder_dir / "foldseek"
    db_path = foldseek_dir / "foldseek_dbs"
    collated_alignment = db_path / alignment_filename
    if collated_alignment.is_file() and use_cache:
        return
    _collate_alignment_files(
        pinder_dir,
        collated_alignment,
        db_path,
        foldseek_db_size,
        alignment_filename=alignment_filename,
    )


def collate_mmseqs_alignments(
    pinder_dir: Path,
    foldseek_db_size: int = 50_000,
    use_cache: bool = True,
    alignment_filename: str = "alignment.txt",
) -> None:
    mmseqs_dir = pinder_dir / "mmseqs2"
    db_path = mmseqs_dir / "mmseqs_dbs"
    collated_alignment = db_path / alignment_filename
    if collated_alignment.is_file() and use_cache:
        return
    _collate_alignment_files(
        pinder_dir,
        collated_alignment,
        db_path,
        foldseek_db_size,
        alignment_filename=alignment_filename,
    )


def aln_to_df(
    alignment_file: Path,
    colnames: list[str],
    score_col: str,
    default_score_val: float = 0.5,
) -> pd.DataFrame:
    df = pd.read_csv(alignment_file, sep="\t", names=colnames)
    df[score_col] = df[score_col].astype(float)
    df[score_col] = df[score_col].fillna(default_score_val)
    return df


def filter_foldseek_edges(
    aln_df: pd.DataFrame, graph_config: GraphConfig = GraphConfig()
) -> pd.DataFrame:
    query_list = [
        "(query != target)",
        f"(alnlen >= {graph_config.min_alignment_length})",
        f"(lddt >= {graph_config.score_threshold})",
        f"(lddt <= {graph_config.upper_threshold})",
    ]
    query_str: str = " and ".join(query_list)
    aln_df = aln_df.query(query_str).reset_index(drop=True)
    return aln_df


def filter_mmseqs_edges(
    aln_df: pd.DataFrame, graph_config: GraphConfig = GraphConfig()
) -> pd.DataFrame:
    query_list = [
        "(query != target)",
        f"(alnlen >= {graph_config.min_alignment_length})",
        f"(pident >= {graph_config.mmseqs_score_threshold})",
        f"(pident <= {graph_config.mmseqs_upper_threshold})",
    ]
    query_str: str = " and ".join(query_list)
    aln_df = aln_df.query(query_str).reset_index(drop=True)
    return aln_df


def alignment_to_parquet(
    alignment_file: Path,
    alignment_type: str,
    foldseek_config: FoldseekConfig = FoldseekConfig(),
    mmseqs_config: MMSeqsConfig = MMSeqsConfig(),
    graph_config: GraphConfig = GraphConfig(),
    use_cache: bool = True,
    remove_original: bool = True,
) -> None:
    alignment_pqt = alignment_file.parent / f"{alignment_file.stem}.parquet"
    filtered_pqt = alignment_file.parent / f"filtered_{alignment_file.stem}.parquet"
    if use_cache and all([f.is_file() for f in (alignment_pqt, filtered_pqt)]):
        log.info(f"Skipping {alignment_type} parquet conversion, files exist...")
        # Remove uncompressed text file
        if alignment_file.is_file() and remove_original:
            alignment_file.unlink()
        return
    foldseek_score_col = foldseek_config.score_type
    mmseqs_score_col = mmseqs_config.score_type
    foldseek_colnames = FOLDSEEK_FIELDS.format(foldseek_score_col).split(",")
    mmseqs_colnames = MMSEQ2_FIELDS.format(mmseqs_score_col).split(",")

    log.info(f"Reading {alignment_type} alignment {alignment_file}")
    if alignment_type == "foldseek":
        colnames = foldseek_colnames
        score_col = foldseek_score_col
    elif alignment_type == "mmseqs":
        colnames = mmseqs_colnames
        score_col = mmseqs_score_col
    else:
        raise ValueError(f"{alignment_type} not recognized or missing schema!")

    aln_df = aln_to_df(alignment_file, colnames=colnames, score_col=score_col)
    log.info(f"Writing {alignment_pqt} with shape {aln_df.shape}")
    aln_df.to_parquet(alignment_pqt, index=False)
    log.info(f"Filtering {alignment_type} alignment...")
    if alignment_type == "foldseek":
        aln_df = filter_foldseek_edges(aln_df, graph_config)
    elif alignment_type == "mmseqs":
        aln_df = filter_mmseqs_edges(aln_df, graph_config)

    log.info(f"Writing {filtered_pqt} with shape {aln_df.shape}")
    aln_df.to_parquet(filtered_pqt, index=False)
    # Remove uncompressed text file
    if remove_original:
        log.debug(f"Removing {alignment_file}...")
        alignment_file.unlink()


def run_foldseek_on_pinder_chains(
    pdb_dir: Path,
    index: str = "index.1.csv.gz",
    foldseek_dir: Path = Path("/tmp/foldseek"),
    config: FoldseekConfig = FoldseekConfig(),
) -> None:
    """Runs foldseek on the PINDER dataset.

    You may need to set your PINDER_DATA_DIR environment variable to the location of the development PINDER dataset.

    Parameters:
        pdb_dir (Path): Input directory containing pinder PDBs to use for populating foldseek inputs.
        index (str): The Pinder index CSV file name.
        foldseek_dir (Path): The directory for storing foldseek input PDBs. Defaults to /tmp/foldseek.
        config (FoldseekConfig): The configuration object containing foldseek parameters.
    """
    if not Path(pdb_dir).is_dir():
        log.error(f"Input PDB directory {pdb_dir} does not exist.")
        return

    create_foldseek_input_dir(index, foldseek_dir, pdb_dir=pdb_dir)
    create_dbs_and_run(
        Path(foldseek_dir) / "foldseek_dbs",
        foldseek_dir,
        config=config,
    )


def run_mmseqs_on_pinder_chains(
    pdb_dir: Path,
    index: str = "index.1.csv.gz",
    output_dir: Path = Path("/tmp/foldseek"),
    use_cache: bool = True,
    config: MMSeqsConfig = MMSeqsConfig(),
) -> None:
    """Runs mmseqs easy-search on the PINDER dataset.

    You may need to set your PINDER_DATA_DIR environment variable to the location of the development PINDER dataset.

    Parameters:
        pdb_dir (Path): Input directory for foldseek
        index (str): The Pinder index CSV file name.
        output_dir (Path): The output directory containing foldseek input PDBs. Defaults to /tmp/foldseek.
        config (MMSeqsConfig): The configuration object containing mmseqs parameters.
    """
    if not Path(pdb_dir).is_dir():
        log.error(f"Input directory {pdb_dir} does not exist.")
        return

    if not output_dir.is_dir():
        log.warning("Foldseek PDB directory does not exist. Attempting to populate...")
        create_foldseek_input_dir(
            index, output_dir, pdb_dir=pdb_dir, use_cache=use_cache
        )

    fasta_file = output_dir / "input.fasta"
    create_fasta_from_foldseek_inputs(output_dir, fasta_file, use_cache=use_cache)
    log.info(f"Created {fasta_file} for mmseqs2.")
    run_mmseqs(fasta_file, output_dir, use_cache=use_cache, config=config)


def setup_mmseqs_dbs(
    pinder_dir: Path,
    mmseqs_db_size: int = 50_000,
    use_cache: bool = True,
) -> None:
    mmseqs_dir = pinder_dir / "mmseqs2"
    mmseqs_dir.mkdir(exist_ok=True, parents=True)
    foldseek_dir = pinder_dir / "foldseek"
    fasta_file = mmseqs_dir / "input.fasta"
    create_fasta_from_foldseek_inputs(foldseek_dir, fasta_file, use_cache=use_cache)
    log.info(f"Created {fasta_file} for mmseqs2.")
    db_path = mmseqs_dir / "mmseqs_dbs"
    fasta_dict = fasta2dict(fasta_file)
    # Split into chunks of fasta files with max size = db_size
    for i, fasta_chunk in scatter.chunk_dict_with_indices(fasta_dict, mmseqs_db_size):
        db_subdir = db_path / f"{i:05d}" / "db"
        db_subdir.mkdir(exist_ok=True, parents=True)
        fasta_str_list: list[str] = []
        for header, seq in fasta_chunk.items():
            fasta_str: str = "\n".join([f">{header}", seq])
            fasta_str_list.append(fasta_str)
        with open(db_subdir / f"input_{i:05d}.fasta", "w") as f:
            f.write("\n".join(fasta_str_list))


def run_mmseqs_db_pair(
    pinder_dir: Path,
    db_indices: tuple[int, int],
    mmseqs_config: MMSeqsConfig = MMSeqsConfig(),
    use_cache: bool = True,
) -> None:
    mmseqs_dir = pinder_dir / "mmseqs2"
    db_path = mmseqs_dir / "mmseqs_dbs"
    if len(db_indices) != 2:
        return
    i, j = db_indices
    log.info(f"Running mmseqs DB {i}, {j}")
    subdir1 = db_path / f"{i:05d}" / "db"
    subdir2 = db_path / f"{j:05d}" / "db"
    output_dir = db_path / f"{i:05d}" / f"{j:05d}"
    alignments = output_dir / mmseqs_config.alignment_filename
    if alignments.is_file() and use_cache:
        log.info(f"mmseqs pair {i},{j} already present. Skipping...")
    else:
        fasta_file = subdir1 / f"input_{i:05d}.fasta"
        target_fasta = subdir2 / f"input_{j:05d}.fasta"
        run_mmseqs(
            fasta_file,
            output_dir,
            target_fasta=target_fasta,
            use_cache=use_cache,
            config=mmseqs_config,
        )
