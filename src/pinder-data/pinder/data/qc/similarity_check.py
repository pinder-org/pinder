from __future__ import annotations
import argparse
import logging
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from itertools import repeat
import pickle as pkl
from tqdm import tqdm
from biotite.sequence import align, ProteinSequence, AlphabetError
from functools import partial

from pinder.core import get_pinder_location
from pinder.core.utils import setup_logger
from pinder.core.utils.process import process_starmap
from pinder.data.config import ClusterConfig, get_config_hash
from pinder.data.qc.utils import load_index, load_metadata, load_entity_metadata

MATRIX = align.SubstitutionMatrix.std_protein_matrix()


log = setup_logger(__name__, log_level=logging.WARNING)


def load_data(
    index_file: Path | str | None = None,
    metadata_file: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"Loading data from {index_file} and {metadata_file}...")
    pindex = load_index(index_file)
    metadata = load_metadata(metadata_file)
    metadata = metadata.set_index("id")
    return pindex, metadata


def process_test_table(test_table_path: Path, pindex: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Processing test table from {test_table_path}...")
    test_table = pd.read_csv(test_table_path)
    test_table = test_table[test_table["split"] == "proto-test"].sort_values(
        "depth_2_hits_with_comm", ascending=True
    )
    test_table = pd.merge(
        test_table, pindex[["id", "pdb_id", "chain_R", "chain_L"]], on="id", how="inner"
    )
    return test_table


def align_sequences(
    input_data: tuple[tuple[str, str], tuple[str, str], str, str],
    output_path: Path,
    chain_overlap_threshold: float = 0.3,
) -> tuple[align.Alignment, Path, float, bool]:
    train_id, test_id, seq1, seq2 = input_data
    align_res = align.align_optimal(
        seq1, seq2, MATRIX, max_number=1, terminal_penalty=False
    )[0]
    train_pdb, train_chain = train_id
    test_pdb, test_chain = test_id
    output_file = output_path / f"{train_pdb}_{train_chain}_{test_pdb}_{test_chain}.pkl"
    max1 = sum([MATRIX.get_score(x, x) for x in seq1 if x != "-"])
    max2 = sum([MATRIX.get_score(x, x) for x in seq2 if x != "-"])
    normed_score = align_res.score / min(max1, max2)
    if (
        (len(seq1) > 40)
        and (len(seq2) > 40)
        and (normed_score > chain_overlap_threshold)
    ):
        return align_res, output_file, normed_score, True
    else:
        return align_res, output_file, normed_score, False


def write_alignment(
    alignment_output: tuple[align.Alignment, Path, float, bool],
) -> None:
    alignment, output_file, score, valid = alignment_output
    if valid:
        with open(output_file, "wb") as f:
            pkl.dump(alignment, f)


def generate_alignments(
    train_sequences: dict[tuple[str, str], str],
    test_sequences: dict[tuple[str, str], str],
    output_path: Path,
    num_cpu: int | None = None,
    parallel: bool = True,
    chain_overlap_threshold: float = 0.3,
    n_chunks: int = 100,
) -> list[tuple[align.Alignment, Path, float, bool]]:
    log.info("Generating/writing alignments...")
    test_vs_train = [
        (train_id, test_id, train_sequences[train_id], test_sequences[test_id])
        for train_id in train_sequences
        for test_id in test_sequences
    ]
    chunk_size = max(1, len(test_vs_train) // n_chunks)
    filtered_alignments = []

    # pass the output path
    for i in tqdm(range(n_chunks)):
        test_vs_train_chunk = test_vs_train[i * chunk_size : (i + 1) * chunk_size]
        alignments = list(
            process_starmap(
                align_sequences,
                zip(
                    test_vs_train_chunk,
                    repeat(output_path),
                    repeat(chain_overlap_threshold),
                ),
                max_workers=num_cpu,
                parallel=parallel,
            )
        )
        valid_alignments = [
            x for x in alignments if x[3] and x[2] > chain_overlap_threshold
        ]
        filtered_alignments.extend(valid_alignments)
        if valid_alignments:
            log.info(f"Found {len(valid_alignments)} valid alignments in chunk-id {i}")
            list(map(write_alignment, valid_alignments))

    return filtered_alignments


def get_processed_alignments(
    alignments_path: Path,
) -> dict[tuple[str, str], list[tuple[align.Alignment, float, tuple[str, str]]]]:
    log.info(f"Loading alignments from {alignments_path}...")
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment_files = list(alignments_path.glob("*.pkl"))
    alignments: dict[
        tuple[str, str], list[tuple[align.Alignment, float, tuple[str, str]]]
    ] = {}
    for alignment_file in tqdm(alignment_files):
        with open(alignment_file, "rb") as f:
            alignment = pkl.load(f)
        train_id, train_chain_id, test_id, test_chain_id = alignment_file.stem.split(
            "_"
        )
        seq1, seq2 = alignment[1].sequences
        max1 = sum([matrix.get_score(x, x) for x in seq1 if x != "-"])
        max2 = sum([matrix.get_score(x, x) for x in seq2 if x != "-"])
        normed_score = alignment[1].score / min([max1, max2])
        if (test_id, test_chain_id) not in alignments:
            alignments[(test_id, test_chain_id)] = [
                (alignment, normed_score, (train_id, train_chain_id))
            ]
        else:
            alignments[(test_id, test_chain_id)].append(
                (alignment, normed_score, (train_id, train_chain_id))
            )

    return alignments


def load_train_system_id_mapping(
    sampled_train: pd.DataFrame,
) -> dict[tuple[tuple[str, str], ...], str]:
    """Generate mapping of system IDs from training data"""
    log.info("Loading train system ID mapping...")
    train_system_id_mapping: dict[tuple[tuple[str, str], ...], str] = {}
    for _, row in sampled_train.iterrows():
        c1: tuple[str, str] = (row["pdb_id"], row["chain_R"])
        c2: tuple[str, str] = (row["pdb_id"], row["chain_L"])
        c1_c2: tuple[tuple[str, str], ...] = tuple(sorted([c1, c2]))
        train_system_id_mapping[c1_c2] = row["id"]
    return train_system_id_mapping


def get_pdb_chain_uni(monomer_id: str) -> tuple[str, str, str]:
    """Extract pdb_id, chain, and uni from a combined monomer_id string"""
    pdb_id, chain_uni = monomer_id.split("__")
    chain, uni = chain_uni.split("_")
    return pdb_id, chain, uni


def system_id_to_alignment(
    sys_id: str,
    alignments: dict[
        tuple[str, str], list[tuple[align.Alignment, float, tuple[str, str]]]
    ],
    train_system_id_mapping: dict[tuple[tuple[str, str], ...], str],
) -> (
    tuple[
        str,
        dict[
            tuple[tuple[str, str], ...],
            tuple[
                tuple[align.Alignment, float, tuple[str, str]],
                tuple[align.Alignment, float, tuple[str, str]],
            ],
        ]
        | None,
    ]
    | None
):
    """Map system IDs to alignments if applicable"""
    train_systems = set(train_system_id_mapping.keys())
    R, L = sys_id.split("--")
    R_pdb_id, R_chain, _ = get_pdb_chain_uni(R)
    L_pdb_id, L_chain, _ = get_pdb_chain_uni(L)
    if (R_pdb_id, R_chain) in alignments and (L_pdb_id, L_chain) in alignments:
        dual = alignments_to_dual_alignment(
            alignments[(R_pdb_id, R_chain)],
            alignments[(L_pdb_id, L_chain)],
            train_systems,
        )
        return sys_id, dual if dual is not None else None
    return None


def alignments_to_dual_alignment(
    r_alignments: list[tuple[align.Alignment, float, tuple[str, str]]],
    l_alignments: list[tuple[align.Alignment, float, tuple[str, str]]],
    train_systems: set[tuple[tuple[str, str], ...]],
) -> (
    dict[
        tuple[tuple[str, str], ...],
        tuple[
            tuple[align.Alignment, float, tuple[str, str]],
            tuple[align.Alignment, float, tuple[str, str]],
        ],
    ]
    | None
):
    """Check for dual alignments across systems and return them if valid"""
    dual_alignments: dict[
        tuple[tuple[str, str], ...],
        tuple[
            tuple[align.Alignment, float, tuple[str, str]],
            tuple[align.Alignment, float, tuple[str, str]],
        ],
    ] = {}
    r_pdbs = {x[-1][0] for x in r_alignments}
    l_pdbs = {x[-1][0] for x in l_alignments}
    if len(r_pdbs.intersection(l_pdbs)) == 0:
        return None
    for r_alignment in r_alignments:
        for l_alignment in l_alignments:
            # If the pdbs are the same
            if r_alignment[-1][0] == l_alignment[-1][0]:
                system = tuple(sorted([r_alignment[-1], l_alignment[-1]]))
                if system in train_systems:
                    dual_alignments[system] = (r_alignment, l_alignment)
    return dual_alignments if dual_alignments else None


def get_aligned_indices(
    alignment_trace: list[tuple[int, int]],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Extract aligned indices from an alignment trace"""
    train_indices, test_indices = [], []
    for i, (a, b) in enumerate(alignment_trace):
        if a != -1 and b != -1:
            train_indices.append(a)
            test_indices.append(b)
    return np.array(train_indices), np.array(test_indices)


def get_aligned_interface_indices(
    alignment: align.Alignment,
    test_system: str,
    train_system: str,
    test_chain: str,
    train_chain: str,
    metadata: pd.DataFrame,
) -> tuple[float, float]:
    """Calculate the percentage of interface residues that are aligned"""
    trace = alignment.trace
    test_interface_residues = (
        metadata.loc[train_system]["chain_1_residues"]
        if test_chain == "R"
        else metadata.loc[train_system]["chain_2_residues"]
    )
    train_interface_residues = (
        metadata.loc[test_system]["chain_1_residues"]
        if train_chain == "R"
        else metadata.loc[test_system]["chain_2_residues"]
    )
    test_interface_residues = {int(x) for x in test_interface_residues.split(",")}
    train_interface_residues = {int(x) for x in train_interface_residues.split(",")}
    train_aligned_indices, test_aligned_indices = get_aligned_indices(trace)
    return (
        len(set(train_aligned_indices).intersection(train_interface_residues))
        / len(train_interface_residues),
        len(set(test_aligned_indices).intersection(test_interface_residues))
        / len(test_interface_residues),
    )


def analyze_leakage(
    test_table: pd.DataFrame,
    alignments: dict[
        tuple[str, str], list[tuple[align.Alignment, float, tuple[str, str]]]
    ],
    train_system_id_mapping: dict[tuple[tuple[str, str], ...], str],
    metadata: pd.DataFrame,
    chain_overlap_threshold: float,
) -> None:
    """Analyze leakage across systems based on alignments and mapping"""
    test_ids = test_table["id"].values.tolist()

    system_map_fn = partial(
        system_id_to_alignment,
        alignments=alignments,
        train_system_id_mapping=train_system_id_mapping,
    )

    all_matched: dict[
        str,
        dict[
            tuple[tuple[str, str], ...],
            tuple[
                tuple[align.Alignment, float, tuple[str, str]],
                tuple[align.Alignment, float, tuple[str, str]],
            ],
        ],
    ] = dict(
        filter(
            lambda x: x is not None and None not in x,  # type: ignore
            map(system_map_fn, test_ids),
        )
    )
    log.info(f"Found {len(all_matched)} matched alignments")

    total_leakage = 0
    for test_id, matched in all_matched.items():
        test_system = test_id
        _, test_chain_r, _ = get_pdb_chain_uni(test_id.split("--")[0])
        _, test_chain_l, _ = get_pdb_chain_uni(test_id.split("--")[1])
        for train_id, (r_al, l_al) in matched.items():
            train_system = train_system_id_mapping[tuple(sorted(list(train_id)))]
            train_chain_r, train_chain_l = train_id[0][1], train_id[1][1]
            chain_r_overlap_train, chain_r_overlap_test = get_aligned_interface_indices(
                r_al[0],
                test_system,
                train_system,
                test_chain_r,
                train_chain_r,
                metadata,
            )
            chain_l_overlap_train, chain_l_overlap_test = get_aligned_interface_indices(
                l_al[0],
                test_system,
                train_system,
                test_chain_l,
                train_chain_l,
                metadata,
            )
            if (
                chain_r_overlap_train > chain_overlap_threshold
                and chain_l_overlap_train > chain_overlap_threshold
                and chain_r_overlap_test > chain_overlap_threshold
                and chain_l_overlap_test > chain_overlap_threshold
            ):
                log.info(
                    f"Leakage detected between {test_system} and {train_system}: Chain R and L overlaps are significant."
                )
                total_leakage += 1
    log.info(
        f"Total leakage ratio: {total_leakage / len(test_table)} ({total_leakage} / {len(test_table)})"
    )


def subsample_train(
    pindex: pd.DataFrame,
    cluster_size_cutoff: int,
    num_to_sample: int,
    random_state: int,
    cache_path: Path,
    n_chunks: int,
) -> pd.DataFrame:
    multi_clusters = (
        pindex.groupby("cluster_id", observed=True)
        .filter(lambda x: len(x) > cluster_size_cutoff)["cluster_id"]
        .unique()
    )
    subsample = (
        pindex[
            (pindex["split"] == "train") & (pindex["cluster_id"].isin(multi_clusters))
        ]
        .groupby("cluster_id", observed=True)
        .sample(n=num_to_sample, random_state=random_state)
    )

    num_train = pindex[pindex["split"] == "train"].shape[0]
    num_subsample = subsample.shape[0]

    log.info(
        (
            f"Subsampled {num_subsample} / {num_train} "
            f"({n_chunks*num_subsample/num_train:.2f}%) of training set."
        )
    )
    subsample.to_csv(cache_path / "subsampled_systems.csv", index=False)
    return subsample


def sequence_leakage_main(
    index_file: str | None = None,
    metadata_file: str | None = None,
    test_table_file: str | Path | None = None,
    entity_metadata_file: str | None = None,
    cache_path: str | Path = get_pinder_location() / "data/similarity-cache",
    cluster_size_cutoff: int = 20,
    chain_overlap_threshold: float = 0.3,
    num_to_sample: int = 1,
    n_chunks: int = 100,
    random_state: int = 42,
    use_cache: bool = True,
    num_cpu: int | None = None,
    pinder_dir: Path | None = None,
    config: ClusterConfig = ClusterConfig(),
) -> None:
    """Extract sequence similarity / leakage by subsampling members in train split.

    Parameters:
        index_file (str | None): Path to custom/intermediate index file, if not provided will use get_index().
        metadata_file (str | None): Path to custom/intermediate metadata file, if not provided will use get_metadata().
        test_table_file (str | None): Path to custom/intermediate test systems table, if not provided must provide pinder ingest directory via pinder_dir.
        entity_metadata_file (str | None): Path to custom/intermediate entity metadata file, if not provided will use get_supplementary_data().
        cache_path (str | Path): Directory to store cached alignments. Defaults to get_pinder_location() / 'data/similarity-cache'.
        cluster_size_cutoff (int): The minimum size of a train set cluster for sampling. Default is 20.
        chain_overlap_threshold (float): Threshold for chain overlap in interface residues. Default is 0.3.
        num_to_sample (int): The number of cluster elements to sample. Default is 1.
        n_chunks (int): The number of sequence pair chunks to evaluate. Default is 100.
        random_state (int): Random state for train set subsampling. Default is 42.
        use_cache (bool): Whether to use cached alignments. Default is True.
        num_cpu (int | None): Limit number of CPU used in multiprocessing. Default is None (use all).
        pinder_dir (Path | None): Directory to pinder dataset generation directory. If not provided, will assume it is being run on local files / post-data gen.
        config (ClusterConfig): The config object used to generate the clustering. Used to infer location of test_table_file if pinder_dir is provided.

    Returns:
        None.

    """
    # Load absolutely necessary data
    pindex, metadata = load_data(index_file, metadata_file)

    # Make sure that we only keep valid entries
    pindex_valid = pindex[
        (pindex["split"].isin(["train", "test", "val"]))
        & ~(pindex["cluster_id"].str.contains("-1", regex=False))
        & ~(pindex["cluster_id"].str.contains("p_p", regex=False))
    ]

    if not test_table_file and pinder_dir is None:
        raise ValueError(f"Must provide one of test_table_file or pinder_dir!")
    if pinder_dir and test_table_file is None:
        if not isinstance(pinder_dir, Path):
            pinder_dir = Path(pinder_dir)
        chk_dir = pinder_dir / "cluster" / get_config_hash(config)
        test_table_file = chk_dir / "test_subset.csv"
        cache_path = chk_dir / "data/similarity-cache"
    assert (
        test_table_file is not None
    ), "test_table_file not provided or could not automatically be located"
    test_table_file = Path(test_table_file)

    test_table = process_test_table(Path(test_table_file), pindex_valid)

    cache_path = Path(cache_path)
    cache_path.mkdir(exist_ok=True, parents=True)

    # subsample the train set
    sampled_train = subsample_train(
        pindex_valid,
        cluster_size_cutoff,
        num_to_sample,
        random_state,
        cache_path,
        n_chunks,
    )

    # Try to get alignments
    alignments_path = cache_path / "alignments"
    alignments = None
    if use_cache and alignments_path.exists():
        log.info(f"Attempting to use cached alignments from {alignments_path}")
        alignments = get_processed_alignments(alignments_path)
        if len(alignments) == 0:
            log.info("No cached alignments found. Creating alignments...")

    if not use_cache or alignments is None:
        log.info(f"Creating alignments in {alignments_path}...")

        log.info(f"Loading entities from {entity_metadata_file}...")
        entities = load_entity_metadata(entity_metadata_file)

        # Prepare sets of PDB chains from the sampled data
        sampled_train_pdb_chain = {
            (row["pdb_id"], row["chain_R"]) for _, row in sampled_train.iterrows()
        }
        sampled_train_pdb_chain.update(
            (row["pdb_id"], row["chain_L"]) for _, row in sampled_train.iterrows()
        )
        sampled_test_pdb_chain = {
            (row["pdb_id"], row["chain_R"]) for _, row in test_table.iterrows()
        }
        sampled_test_pdb_chain.update(
            (row["pdb_id"], row["chain_L"]) for _, row in test_table.iterrows()
        )

        # Filter entities based on the sampled train and test PDB chains
        relevant_entry_ids = {
            x[0] for x in sampled_train_pdb_chain | sampled_test_pdb_chain
        }
        entities = entities[entities["entry_id"].isin(relevant_entry_ids)]

        # Prepare sequences for alignment
        all_sequences = {
            (x1, x2): seq
            for (x1, x2, seq) in zip(
                entities["entry_id"], entities["chain"], entities["sequence"]
            )
        }
        train_sequences = {}
        train_skipped_count = 0
        for x in sampled_train_pdb_chain:
            try:
                train_sequences[x] = ProteinSequence(all_sequences[x])
            except AlphabetError as e:
                log.warning(f"{x} has a non-standard amino acid! Skipping")
                train_skipped_count += 1

        log.warning(
            f"Skipped {train_skipped_count} / {len(sampled_train_pdb_chain)} train sequences"
        )

        test_sequences = {}
        test_skipped_count = 0
        for x in sampled_test_pdb_chain:
            try:
                test_sequences[x] = ProteinSequence(all_sequences[x])
            except AlphabetError as e:
                log.warning(f"{x} has a non-standard amino acid! Skipping")
                test_skipped_count += 1

        log.warning(
            f"Skipped {test_skipped_count} / {len(sampled_test_pdb_chain)} test sequences"
        )

        # Process alignments
        alignments_path.mkdir(exist_ok=True)
        gen_alignments = generate_alignments(
            train_sequences,
            test_sequences,
            alignments_path,
            num_cpu,
            chain_overlap_threshold=chain_overlap_threshold,
            n_chunks=n_chunks,
        )
        log.info(f"Total valid alignments processed: {len(gen_alignments)}")

        # Further processing and setup for analysis...
        log.info(
            f"Processed data for {len(pindex_valid)} indexed entries and {len(test_table)} test entries."
        )
        alignments = get_processed_alignments(alignments_path)

    train_system_id_mapping = load_train_system_id_mapping(sampled_train)
    # train_systems = set(train_system_id_mapping.keys())

    analyze_leakage(
        test_table,
        alignments,
        train_system_id_mapping,
        metadata,
        chain_overlap_threshold,
    )
