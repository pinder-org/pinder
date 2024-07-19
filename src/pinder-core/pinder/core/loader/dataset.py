from __future__ import annotations
import multiprocessing
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from pinder.core.index.system import PinderSystem
from pinder.core.index.utils import get_index, get_pinder_location
from pinder.core.loader.loader import get_systems, PinderLoader
from pinder.core.loader.geodata import NodeRepresentation, PairedPDB
from pinder.core.loader.filters import PinderFilterSubBase, PinderFilterBase
from pinder.core.utils.log import setup_logger

log = setup_logger(__name__)


class PPIDataset(Dataset):  # type: ignore
    def __init__(
        self,
        node_types: set[NodeRepresentation],
        split: str = "train",
        monomer1: str = "holo_receptor",
        monomer2: str = "holo_ligand",
        base_filters: list[PinderFilterBase] = [],
        sub_filters: list[PinderFilterSubBase] = [],
        root: Path = get_pinder_location(),
        transform: Callable[[PinderSystem], PinderSystem] | None = None,
        pre_transform: Callable[[PinderSystem], PinderSystem] | None = None,
        pre_filter: Callable[[PinderSystem], PinderSystem | bool] | None = None,
        limit_by: int | None = None,
        force_reload: bool = False,
        filenames_dir: Path | str | None = None,
        repeat: int = 1,
        use_cache: bool = False,
        ids: list[str] | None = None,
        add_edges: bool = True,
        k: int = 10,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        self.node_types = node_types
        self.split = split
        self.use_cache = use_cache
        self.repeat = repeat
        self.force_reload = force_reload
        self.filenames: list[str] | set[str] = []
        self.limit_by = limit_by
        self.base_filters = base_filters
        self.sub_filters = sub_filters
        self.pindex = get_index()
        self.monomer1 = monomer1
        self.monomer2 = monomer2
        self.add_edges = add_edges
        self.k = k
        self.parallel = parallel
        self.max_workers = max_workers
        default_file_dir = Path(root) / "filenames"
        self.filenames_dir = default_file_dir
        if filenames_dir:
            self.filenames_dir = Path(filenames_dir)
        self.filenames_dir.mkdir(exist_ok=True)
        # pre_filter = self.loader.apply_filters_and_transforms
        self.ids = ids or list(self.pindex.query(f'split == "{self.split}"').id)
        if force_reload:
            self.ids_to_process = self.ids
        else:
            # Get pre-cached ids to remove from generator
            processed_dir = Path(root) / "processed"
            processed_ids = {x.stem for x in Path(processed_dir).glob("*.pt")}
            self.ids_to_process = list(set(self.ids) - processed_ids)

        self.loader = PinderLoader(
            dimers=get_systems(self.ids_to_process),
            base_filters=base_filters,
            sub_filters=sub_filters,
        )
        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        # self.process()

    @property
    def raw_file_names(self) -> list[str]:
        return sorted(self.ids) if self.limit_by is None else self.ids[: self.limit_by]

    @property
    def processed_file_names(self) -> list[Path]:
        processed_files = {x.stem: x for x in Path(self.processed_dir).glob("*.pt")}
        return sorted(
            [processed_files[x] for x in self.filenames if x in processed_files]
        )

    @staticmethod
    def process_single_file(
        system: PinderSystem,
        node_types: set[NodeRepresentation],
        monomer1: str,
        monomer2: str,
        output_file: Path,
        add_edges: bool = True,
        k: int = 10,
    ) -> bool:
        try:
            data = PairedPDB.from_pinder_system(
                system=system,
                monomer1=monomer1,
                monomer2=monomer2,
                node_types=node_types,
                add_edges=add_edges,
                k=k,
            )
            torch.save(data, output_file)
            return True
        except Exception as e:
            log.error(f"Unable to process {system.entry.id}: {e}")
            return False

    @staticmethod
    def process_single_file_parallel(
        args: tuple[
            PinderSystem,
            set[NodeRepresentation],
            Path,
            list[PinderFilterBase],
            list[PinderFilterSubBase],
            str,
            str,
            bool,
            int,
            bool,
        ],
    ) -> str | None:
        (
            system,
            node_types,
            processed_dir,
            base_filters,
            sub_filters,
            monomer1,
            monomer2,
            add_edges,
            k,
            force_reload,
        ) = args
        if not hasattr(system, "entry"):
            return None
        pinder_id: str = system.entry.id
        to_write = Path(processed_dir) / f"{pinder_id}.pt"
        if to_write.exists() and force_reload:
            to_write.unlink()
        if to_write.exists():
            return pinder_id
        try:
            system = PinderLoader.apply_dimer_filters(system, base_filters, sub_filters)
        except Exception as e:
            log.error(str(e))
        if not system:
            return None
        processed = PPIDataset.process_single_file(
            system,
            node_types,
            monomer1,
            monomer2,
            to_write,
            add_edges,
            k,
        )
        if processed:
            return pinder_id
        return None

    def process_parallel(self) -> None:
        with multiprocessing.get_context("spawn").Pool(processes=self.max_workers) as p:
            args_gen = (
                (
                    dimer,
                    self.node_types,
                    self.processed_dir,
                    self.loader.base_filters,
                    self.loader.sub_filters,
                    self.monomer1,
                    self.monomer2,
                    self.add_edges,
                    self.k,
                    self.force_reload,
                )
                for dimer in self.loader.dimers
            )
            processed_ids = set()
            for processed_id in p.imap_unordered(
                PPIDataset.process_single_file_parallel, args_gen
            ):
                if isinstance(processed_id, str):
                    processed_ids.add(processed_id)

        self.filenames = set(self.filenames).union(processed_ids)
        self.filenames = list(self.filenames)
        self.filenames = [fname for fname in self.filenames for _ in range(self.repeat)]
        with open(Path(self.filenames_dir) / "filenames.txt", "w") as f:
            f.write("\n".join(self.filenames))

    def process(self) -> None:
        self.filenames = set(self.filenames)
        if not self.use_cache:
            if self.parallel:
                return self.process_parallel()
            for system in self.loader.dimers:
                if self.limit_by and len(self.filenames) >= self.limit_by:
                    print(f"Finished processing, only {self.limit_by} systems")
                    break
                try:
                    system = self.loader.apply_dimer_filters(
                        system, self.loader.base_filters, self.loader.sub_filters
                    )
                except Exception as e:
                    log.error(str(e))
                    continue
                if not system:
                    continue
                pinder_id = system.entry.id
                to_write = Path(self.processed_dir) / f"{pinder_id}.pt"
                if to_write.exists() and self.force_reload:
                    to_write.unlink()
                if to_write.exists():
                    self.filenames.add(pinder_id)
                    continue
                processed = PPIDataset.process_single_file(
                    system,
                    self.node_types,
                    self.monomer1,
                    self.monomer2,
                    to_write,
                    self.add_edges,
                    self.k,
                )
                if processed:
                    self.filenames.add(pinder_id)
            self.filenames = list(self.filenames)
            self.filenames = [
                fname for fname in self.filenames for _ in range(self.repeat)
            ]
            with open(Path(self.filenames_dir) / "filenames.txt", "w") as f:
                f.write("\n".join(self.filenames))
        else:
            with open(Path(self.filenames_dir) / "filenames.txt", "r") as f:
                self.filenames = {x.strip() for x in f.readlines()}
            self.filenames = [x.stem for x in self.processed_file_names]
            self.filenames = [
                fname for fname in self.filenames for _ in range(self.repeat)
            ]

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> PairedPDB:
        assert isinstance(self.filenames, list)
        filename = self.filenames[idx]
        pt_file = Path(self.processed_dir) / f"{filename}.pt"
        data = self.load_filename(pt_file, idx)
        return data

    def get_filename(self, filename: str) -> PairedPDB:
        idx = self.get_filename_idx(filename)
        return self.get(idx)

    def get_filename_idx(self, key: str) -> int:
        idx = [i for i, file in enumerate(self.filenames) if file == key]
        if not idx:
            raise KeyError(f"{key} not in {self.filenames}")
        file_idx = idx[0]
        return file_idx

    @staticmethod
    def load_filename(filename: Path, idx: int) -> PairedPDB:
        data = torch.load(filename)
        data["pdb"].id = torch.tensor([idx]).type(torch.int32)
        data["pdb"].num_nodes = 1
        return data


def get_geo_loader(
    dataset: PPIDataset, batch_size: int = 2, num_workers: int = 1
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
