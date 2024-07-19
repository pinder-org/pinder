from __future__ import annotations
from itertools import islice
from math import ceil
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd


def chunk_collection(
    array: list[Any], batch_size: int | None = None, num_batches: int | None = None
) -> Generator[list[Any], list[Any], None]:
    """Simple chunking algorithm with two possible constraints:
        1. the maximum number of items within a chunk
        2. the maximum total number of chunks

    Note
    ----
    Does not guarantee even packing across individual chunks

    Parameters
    ----------
    array : List
        the iterable to be chunked
    batch_size : Optional[int], default=None
        the maximum number of items in a given chunk
    num_batches : Optional[int], default=None
        the maximum number of chunks to be produced

    Returns
    -------
    chunks : Generator[List[Any], List[Any], None]
    """

    num_items = len(array)
    if batch_size is None and num_batches is None:
        raise ValueError("Either batch_size or num_batches must be provided.")

    if batch_size is None and num_batches is not None:
        max_chunk_size = ceil(num_items / num_batches)
        max_chunks = num_batches

    elif batch_size is not None and num_batches is None:
        max_chunks = ceil(num_items / batch_size)
        max_chunk_size = batch_size

    elif batch_size is not None and num_batches is not None:
        max_chunk_size = batch_size
        max_chunks = num_batches

    chunk_size = max(max_chunk_size, ceil(num_items / max_chunks))
    if chunk_size == max_chunk_size:
        max_chunks = ceil(num_items / chunk_size)
        chunk_size = ceil(num_items / max_chunks)

    for i in range((num_items + chunk_size - 1) // chunk_size):
        yield array[i * chunk_size : (i + 1) * chunk_size]


def chunk_dict(
    data: dict[str, str],
    batch_size: int,
) -> Generator[dict[str, str], dict[str, str], None]:
    it = iter(data)
    for i in range(0, len(data), batch_size):
        yield {k: data[k] for k in islice(it, batch_size)}


def chunk_all_vs_all_indices(
    array: list[Any],
    batch_size: int,
) -> Generator[tuple[int, int], tuple[int, int], None]:
    n_items = len(array)
    array_chunks = [
        array[i * batch_size : (i + 1) * batch_size]
        for i in range((n_items + batch_size - 1) // batch_size)
    ]
    for i in range(len(array_chunks)):
        for j in range(len(array_chunks)):
            if j < i:
                continue
            yield (i, j)


def chunk_dict_with_indices(
    data: dict[str, str],
    batch_size: int,
) -> Generator[tuple[int, dict[str, str]], tuple[int, dict[str, str]], None]:
    for i, dict_chunk in enumerate(chunk_dict(data, batch_size=batch_size)):
        yield (i, dict_chunk)


def chunk_dataframe(
    data: pd.DataFrame,
    batch_size: int,
) -> Generator[tuple[int, pd.DataFrame], tuple[int, pd.DataFrame], None]:
    df_chunks = np.array_split(data, batch_size)
    for i, df in enumerate(df_chunks):
        yield (i, df)


def chunk_apo_pairing_ids(
    data: pd.DataFrame,
    batch_size: int,
    pinder_dir: Path,
) -> Generator[tuple[list[str], Path], tuple[list[str], Path], None]:
    metric_dir = pinder_dir / "apo_metrics/pair_eval"
    existing_idx = [
        int(pqt.stem.split("metrics_")[1])
        for pqt in metric_dir.glob("metrics_*.parquet")
    ]
    if len(existing_idx):
        current_idx = max(existing_idx) + 1
    else:
        current_idx = 0

    data["pairing_id"] = data["id"] + ":" + data["apo_monomer_id"] + ":" + data["body"]
    id_chunks = chunk_collection(list(data.pairing_id), batch_size=batch_size)
    for i, id_chunk in enumerate(id_chunks):
        output_idx = current_idx + i
        output_pqt = metric_dir / f"metrics_{output_idx}.parquet"
        yield (id_chunk, output_pqt)
