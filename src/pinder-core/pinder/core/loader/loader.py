from __future__ import annotations
import multiprocessing
from collections.abc import Generator
from itertools import chain, islice
from typing import Iterable

from tqdm import tqdm
from dataclasses import dataclass, field
from pinder.core.index.utils import get_index
from pinder.core.index.system import PinderSystem
from pinder.core.loader.filters import PinderFilterSubBase, PinderFilterBase
from pinder.core.loader.writer import PinderWriterBase
from pinder.core.utils.dataclass import stringify_dataclass
from pinder.core.utils.log import setup_logger
from pinder.core.structure.models import DatasetName


log = setup_logger(__name__)


def chunked_generator(
    iterable: Iterable[PinderSystem], chunk_size: int = 10
) -> Generator[chain[PinderSystem], chain[PinderSystem], None]:
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, chunk_size - 1))


def get_systems(systems: list[str]) -> Generator[PinderSystem, PinderSystem, None]:
    for system in tqdm(systems):
        try:
            system = PinderSystem(system)
            yield system
        except Exception as e:
            log.error(str(e))
            continue


@dataclass
class PinderLoader:
    dimers: Iterable[PinderSystem] | None = None
    base_filters: list[PinderFilterBase | None] = field(default_factory=list)
    sub_filters: list[PinderFilterSubBase | None] = field(default_factory=list)
    writer: PinderWriterBase | None = None

    def load_systems(self, systems: list[str]) -> None:
        self.systems = get_systems(systems)
        self.dimers = self.apply_filters_and_transforms()
        # self.load()

    def load_split(
        self, split: str = "train", subset: DatasetName | None = None
    ) -> None:
        pindex = get_index()
        if subset:
            if not isinstance(subset, DatasetName):
                subset = DatasetName(subset)
            ids = list(pindex.query(f"split == '{split}' and {subset.value}").id)
        else:
            ids = list(pindex.query(f'split == "{split}"').id)
        if not ids:
            raise ValueError(f"No systems found matching {split} {subset}")
        self.load_systems(ids)

    def apply_filters_and_transforms(self) -> Iterable[list[PinderSystem]]:
        for dimer in self.systems:
            dimer = self.apply_dimer_filters(dimer, self.base_filters, self.sub_filters)
            if dimer:
                yield dimer

    @staticmethod
    def apply_dimer_filters(
        dimer: PinderSystem,
        base_filters: list[PinderFilterBase] | list[None] = [],
        sub_filters: list[PinderFilterSubBase] | list[None] = [],
    ) -> PinderSystem | bool:
        for sub_filter in sub_filters:
            if isinstance(sub_filter, PinderFilterSubBase):
                dimer = sub_filter(dimer)
        for base_filter in base_filters:
            if isinstance(base_filter, PinderFilterBase):
                if not base_filter(dimer):
                    return False
        return dimer

    def load(
        self, n_cpu: int = 8, batch_size: int = 10
    ) -> Iterable[list[PinderSystem]]:
        assert self.dimers is not None
        if n_cpu < 2 or self.writer is None:
            return self._load_serial(batch_size)
        else:
            return self._load_parallel(n_cpu, batch_size)

    def _load_parallel(
        self, n_cpu: int, batch_size: int
    ) -> Iterable[list[PinderSystem]]:
        assert self.writer is not None
        assert self.dimers is not None
        if batch_size < 2:
            with multiprocessing.get_context("spawn").Pool(n_cpu) as p:
                for dimer in p.imap_unordered(self.writer, self.dimers, chunksize=10):
                    yield self.writer(dimer)
        else:
            with multiprocessing.get_context("spawn").Pool(n_cpu) as p:
                chunked_generator(
                    (p.imap_unordered(self.writer, self.dimers, chunksize=10)),
                    chunk_size=batch_size,
                )

    def _load_serial(self, batch_size: int) -> Iterable[list[PinderSystem]]:
        assert self.dimers is not None
        if batch_size < 2:
            for dimer in self.dimers:
                if self.writer is not None:
                    yield self.writer(dimer)
                else:
                    yield dimer
        else:
            for outputs in tqdm(chunked_generator(self.dimers, chunk_size=batch_size)):
                if self.writer is not None:
                    outputs = list(outputs)
                    for dimer in outputs:
                        yield self.writer(dimer)
                yield outputs

    def __repr__(self) -> str:
        class_str: str = stringify_dataclass(self, 4)
        return class_str
