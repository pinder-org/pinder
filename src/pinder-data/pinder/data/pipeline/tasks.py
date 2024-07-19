from __future__ import annotations
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Generator, Sized, Union
import pandas as pd

from pinder.core.utils.log import setup_logger
from pinder.data import get_apo, get_dimers, rcsb_rsync
from pinder.data.pipeline.scatter import chunk_collection


log = setup_logger(__name__)

EMPTY_TASKS: list[list[Any]] = [[]]
INPUT_TYPES = Union[
    list[str],
    list[Path],
    list[tuple[str, bool]],
    list[Union[tuple[Path, Path], pd.DataFrame]],
]


def cif_glob(data_dir: Path) -> list[Path]:
    return list(data_dir.glob("*/*/*enrich.cif.gz"))


def dimer_glob(data_dir: Path) -> list[Path]:
    return list(data_dir.glob("*/*/*__*--*__*.pdb"))


def entry_glob(data_dir: Path) -> list[Path]:
    entry_dirs: list[Path] = get_dimers.get_pdb_entry_dirs(data_dir)
    return entry_dirs


def two_char_code_rsync(data_dir: Path) -> list[str]:
    two_char_codes: list[str] = rcsb_rsync.get_rsync_directories()
    two_char_codes = rcsb_rsync.get_two_char_codes_not_downloaded(
        data_dir, two_char_codes
    )
    return two_char_codes


def pdb_id_glob(data_dir: Path) -> list[str]:
    entry_dirs = entry_glob(data_dir)
    pdb_ids = [entry.stem.split("_")[1][-4:].lower() for entry in entry_dirs]
    return pdb_ids


def foldseek_pdb_glob(data_dir: Path) -> list[Path]:
    return list(data_dir.glob("*.pdb"))


def plm_embedding_glob(data_dir: Path) -> list[Path]:
    return list(data_dir.glob("embeddings_*.npz"))


def plm_embedding_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    npz_files = plm_embedding_glob(data_dir)
    npz_pairs = []
    for i in range(len(npz_files)):
        for j in range(i + 1, len(npz_files)):
            chunk1 = npz_files[i]
            chunk2 = npz_files[j]
            output_dir = chunk1.parent / "similarities"
            output_file = output_dir / f"similarities_{chunk1.stem}_{chunk2.stem}.txt"
            if output_file.is_file():
                continue
            npz_pairs.append((chunk1, chunk2))
    return npz_pairs


def plm_embedding_fasta(data_dir: Path) -> list[Path]:
    fasta_file = list(data_dir.glob("*.fasta"))
    return fasta_file


def graph_type_glob(data_dir: Path) -> list[tuple[str, bool]]:
    graph_specs = []
    for g in data_dir.glob("cleaned_*_alignment_graph.pkl"):
        graph_type = g.stem.split("cleaned_")[1].split("_alignment_graph")[0]
        if graph_type == "foldseek":
            # Create two foldseek graph specs: af2_transitive_hits=False + af2_transitive_hits=True
            graph_specs.append((graph_type, False))
            graph_specs.append((graph_type, True))
        else:
            # AF2 alternative transitive hits only applies to foldseek graph_type
            graph_specs.append((graph_type, False))
    return graph_specs


def putative_apo_pairings(data_dir: Path) -> pd.DataFrame:
    RL = get_apo.get_putative_pairings(
        pinder_dir=data_dir, use_cache=True, remove_chain_copies=False
    )
    RL["pairing_id"] = RL["id"] + ":" + RL["apo_monomer_id"] + ":" + RL["body"]

    # Remove pairing IDs that already have metrics present
    metric_dir = data_dir / "apo_metrics/pair_eval"
    existing_ids: set[str] = set()
    for pqt in metric_dir.glob("metrics_*.parquet"):
        exist = pd.read_parquet(pqt, columns=["id", "apo_monomer_id", "unbound_body"])
        exist["pairing_id"] = (
            exist["id"] + ":" + exist["apo_monomer_id"] + ":" + exist["unbound_body"]
        )
        existing_ids = existing_ids.union(set(exist.pairing_id))
    RL = RL[~RL.pairing_id.isin(set(existing_ids))].reset_index(drop=True)
    return RL


DATA_GLOBS: dict[str, Callable[[Path], INPUT_TYPES]] = {
    "cif": cif_glob,
    "dimers": dimer_glob,
    "entries": entry_glob,
    "two_char_codes": two_char_code_rsync,
    "pdb_ids": pdb_id_glob,
    "foldseek": foldseek_pdb_glob,
    "plm_fasta": plm_embedding_fasta,
    "plm_embeddings": plm_embedding_glob,
    "plm_embedding_pairs": plm_embedding_pairs,
    "graph_types": graph_type_glob,
    "putative_apo_pairings": putative_apo_pairings,
}


def get_stage_inputs(
    data_dir: Path,
    input_type: str,
) -> INPUT_TYPES:
    glob_func: Callable[[Path], INPUT_TYPES] = DATA_GLOBS[input_type]
    inputs: INPUT_TYPES = glob_func(data_dir)
    return inputs


def get_cache_delta(
    cache_func: Callable[..., INPUT_TYPES],
    batches: INPUT_TYPES,
    **kwargs: str | Path | bool,
) -> INPUT_TYPES:
    remaining = cache_func(batches, **kwargs)
    return remaining


def get_stage_tasks(
    data_dir: Path,
    input_type: str,
    batch_size: int,
    cache_func: Callable[..., list[str] | list[Path]] | None = None,
    cache_kwargs: dict[str, str | Path | bool] = {},
    scatter_method: Callable[..., Generator[Any, Any, None]] = chunk_collection,
    scatter_kwargs: dict[str, str | Path | bool] = {},
) -> list[INPUT_TYPES]:
    batches = get_stage_inputs(data_dir, input_type=input_type)
    tasks = EMPTY_TASKS
    if batches is None or len(batches) == 0:
        return tasks

    if cache_func is not None:
        # This is only available after calling download_rcsb_files
        batches = get_cache_delta(cache_func, batches, **cache_kwargs)
    if len(batches):
        tasks = list(scatter_method(batches, batch_size=batch_size, **scatter_kwargs))

    log.info(f"{input_type}: {len(batches)} inputs -> {len(tasks)} task batches")
    return tasks


def run_task(
    task_func: Callable[[Any], Any],
    task_input: dict[
        str,
        Path
        | list[Path]
        | list[str]
        | tuple[Path, Path]
        | tuple[int, int]
        | tuple[int, dict[str, str]],
    ],
    iterable_kwarg: str | None = None,
) -> None:
    # Inspect task_func signature to ensure task_input covers all required arguments
    sig = signature(task_func)
    missing_args = [
        param.name
        for param in sig.parameters.values()
        if param.name not in task_input and param.default is param.empty
    ]
    task_name: str = task_func.__name__
    if missing_args:
        log.error(f"Missing required arguments for {task_name}: {missing_args}")
        return

    if iterable_kwarg and iterable_kwarg in task_input:
        # Check if the value supports the `len` operation by ensuring it's a `Sized` instance.
        input_value = task_input[iterable_kwarg]
        if isinstance(input_value, Sized):
            arg_msg = f"{len(input_value)} {iterable_kwarg} inputs"
            if not len(input_value):
                log.info(f"Skipping {task_name}: {iterable_kwarg} inputs are empty...")
                return
        else:
            # Handle the case where the value does not support `len`
            log.debug(f"{iterable_kwarg} does not support length operation.")
    else:
        arg_msg = ", ".join(task_input.keys())
    log.info(f"Running {task_name}: {arg_msg}")
    # Should probably just pipe **kwargs defined in run_task signature to task_func
    task_func(**task_input)  # type: ignore
