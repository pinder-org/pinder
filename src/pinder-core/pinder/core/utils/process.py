"""Helper functions for toggling serial or parallel processing via map and starmap."""

from __future__ import annotations
import multiprocessing
from typing import Any, Callable, Iterable

from tqdm import tqdm


def process_starmap(
    func: Callable[..., Any],
    args: Iterable[Iterable[Any]],
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[Any]:
    if parallel:
        with multiprocessing.get_context("spawn").Pool(processes=max_workers) as pool:
            outputs = pool.starmap(func, args)
    else:
        outputs = []
        for args_i in tqdm(args):
            outputs.append(func(*args_i))
    return outputs


def process_map(
    func: Callable[..., Any],
    args: Iterable[Iterable[Any]],
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[Any]:
    if parallel:
        with multiprocessing.get_context("spawn").Pool(processes=max_workers) as pool:
            outputs = pool.map(func, args)
    else:
        outputs = []
        for args_i in tqdm(args):
            outputs.append(func(args_i))
    return outputs
