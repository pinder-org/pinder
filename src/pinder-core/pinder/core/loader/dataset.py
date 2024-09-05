"""Construct torch datasets and dataloaders from pinder systems.

This module provides two example implementations of how to integrate the pinder dataset into a torch-based machine learning pipeline.

1. PinderDataset: A torch Dataset that can be used with torch DataLoaders.
2. PPIDataset: A torch-geometric Dataset that can be used with torch-geometric DataLoaders. This class is designed to be used with the torch_geometric package.

Together, the two datasets provide an example implementation of how to abstract away the complexity of loading and processing multiple structures associated with each `PinderSystem` by leveraging the following utilities from pinder:

* `pinder.core.PinderLoader`
* `pinder.core.loader.filters`
* `pinder.core.loader.transforms`

The examples cover two different batch data item structures to illustrate two different use-cases:

* PinderDataset: A batch of `(target_complex, feature_complex)` pairs, where `target_complex` and `feature_complex` are `torch.Tensor` objects representing the atomic coordinates and atom types of the holo and sampled (decoy, holo/apo/pred) complexes, respectively.
* PPIDataset: A batch of `PairedPDB` objects, where the receptor and ligand are encoded separately in a heterogeneous graph, via `torch_geometric.data.HeteroData`, holding multiple node and/or edge types in disjunct storage objects.

"""

from __future__ import annotations
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Dataset as TorchGeoDataset
from torch_geometric.loader import DataLoader as TorchGeoDataLoader

from pinder.core.index.system import PinderSystem
from pinder.core.index.utils import get_index, get_pinder_location
from pinder.core.loader.filters import (
    PinderFilterSubBase,
    PinderFilterBase,
    StructureFilter,
)
from pinder.core.loader.geodata import NodeRepresentation, PairedPDB, structure2tensor
from pinder.core.loader.loader import PinderLoader
from pinder.core.loader.structure import Structure
from pinder.core.loader.transforms import StructureTransform
from pinder.core.utils.log import setup_logger

RES_IDX_PAD_VALUE = -99
COORDS_PAD_VALUE = -100
ATOM_TYPE_PAD_VALUE = -1

log = setup_logger(__name__)


def structure2tensor_transform(structure: Structure) -> dict[str, torch.Tensor]:
    props: dict[str, torch.Tensor] = structure2tensor(
        atom_coordinates=structure.coords,
        atom_types=structure.atom_array.element,
        residue_coordinates=structure.coords,
        residue_types=structure.atom_array.res_name,
        residue_ids=structure.atom_array.res_id,
    )
    return props


def pad_to_max_length(
    mat: Tensor,
    max_length: int | Sequence[int] | Tensor,
    dims: Sequence[int],
    value: int | float | None = None,
) -> Tensor:
    """Takes a tensor and pads it to maximum length with right padding on the specified dimensions.

    Parameters:
        mat (Tensor): The tensor to pad. Can be of any shape
        max_length (int | Sequence[int] | Tensor): The size of the tensor along specified dimensions after padding.
        dims (Sequence[int]): The dimensions to pad. Must have the same number of elements as `max_length`.
        value (int, optional): The value to pad with, by default None

    Returns:
        Tensor : The padded tensor. Below are examples of input and output shapes
            Example 1:
                input: (2, 3, 4), max_length: 5, dims: [0, 2]
                output: (5, 3, 5)
            Example 2:
                input: (2, 3, 4), max_length: 5, dims: [0]
                output: (5, 3, 4)
            Example 3:
                input: (2, 3, 4), max_length: [5, 7], dims: [0, 2]
                output: (5, 3, 7)

    """
    if not isinstance(max_length, int):
        assert len(dims) == len(max_length)

    num_dims = len(mat.shape)
    pad_idxs = [(num_dims - i) * 2 - 1 for i in dims]

    if isinstance(max_length, int):
        pad_sizes = [
            max_length - mat.shape[int(-(i + 1) / 2 + num_dims)] if i in pad_idxs else 0
            for i in range(num_dims * 2)
        ]
    else:
        max_length_list = (
            list(max_length) if not isinstance(max_length, list) else max_length
        )
        pad_sizes = [
            max_length_list[int(-(i + 1) / 2 + num_dims)]
            - mat.shape[int(-(i + 1) / 2 + num_dims)]
            if i in pad_idxs
            else 0
            for i in range(num_dims * 2)
        ]

    return torch.nn.functional.pad(input=mat, pad=tuple(pad_sizes), value=value)


def pad_and_stack(
    tensors: list[Tensor],
    dim: int = 0,
    dims_to_pad: list[int] | None = None,
    value: int | float | None = None,
) -> Tensor:
    """Pads a list of tensors to the maximum length observed along each dimension and then stacks them along a new dimension (given by `dim`).

    Parameters:
        tensors (list[Tensor]): A list of tensors to pad and stack
        dim (int): The new dimension to stack along.
        dims_to_pad (list[int] | None): The dimensions to pad
        value (int | float | None, optional): The value to pad with, by default None

    Returns:
        Tensor: The padded and stacked tensor. Below are examples of input and output shapes
            Example 1: Sequence features (although redundant with torch.rnn.utils.pad_sequence)
                input: [(2,), (7,)], dim: 0
                output: (2, 7)
            Example 2: Pair features (e.g., pairwise coordinates)
                input: [(4, 4, 3), (7, 7, 3)], dim: 0
                output: (2, 7, 7, 3)

    """
    assert (
        len({t.ndim for t in tensors}) == 1
    ), f"All `tensors` must have the same number of dimensions."

    # Pad all dims if none are specified
    if dims_to_pad is None:
        dims_to_pad = list(range(tensors[0].ndim))

    # Find the max length of the dims_to_pad
    shapes = torch.tensor([t.shape for t in tensors])
    envelope = shapes.max(dim=0).values
    max_length = envelope[dims_to_pad]

    padded_matrices = [
        pad_to_max_length(t, max_length, dims_to_pad, value) for t in tensors
    ]
    return torch.stack(padded_matrices, dim=dim)


def collate_complex(
    structures: list[dict[str, Tensor]],
    coords_pad_value: int = COORDS_PAD_VALUE,
    atom_type_pad_value: int = ATOM_TYPE_PAD_VALUE,
    residue_id_pad_value: int = RES_IDX_PAD_VALUE,
) -> dict[str, Tensor]:
    atom_types = []
    residue_types = []
    atom_coordinates = []
    residue_coordinates = []
    residue_ids = []
    for x in structures:
        atom_types.append(x["atom_types"])
        residue_types.append(x["residue_types"])
        atom_coordinates.append(x["atom_coordinates"])
        residue_coordinates.append(x["residue_coordinates"])
        residue_ids.append(x["residue_ids"])
    return {
        "atom_types": pad_and_stack(atom_types, dim=0, value=atom_type_pad_value),
        "residue_types": pad_and_stack(residue_types, dim=0, value=atom_type_pad_value),
        "atom_coordinates": pad_and_stack(
            atom_coordinates, dim=0, value=coords_pad_value
        ),
        "residue_coordinates": pad_and_stack(
            residue_coordinates, dim=0, value=coords_pad_value
        ),
        "residue_ids": pad_and_stack(residue_ids, dim=0, value=residue_id_pad_value),
    }


def collate_batch(
    batch: list[dict[str, dict[str, Tensor] | str]],
) -> dict[str, dict[str, Tensor] | list[str]]:
    """Collate a batch of PinderDataset items into a merged mini-batch of Tensors.

    Used as the default collate_fn for the torch DataLoader consuming PinderDataset.

    Parameters:
        batch (list[dict[str, dict[str, Tensor] | str]]): A list of dictionaries containing the data for each item in the batch.

    Returns:
        dict[str, dict[str, Tensor] | list[str]]: A dictionary containing the merged Tensors for the batch.

    """
    ids: list[str] = []
    sample_ids: list[str] = []
    target_ids: list[str] = []
    target_structures: list[dict[str, Tensor]] = []
    feature_structures: list[dict[str, Tensor]] = []
    for x in batch:
        assert isinstance(x["id"], str)
        assert isinstance(x["sample_id"], str)
        assert isinstance(x["target_id"], str)
        assert isinstance(x["target_complex"], dict)
        assert isinstance(x["feature_complex"], dict)
        ids.append(x["id"])
        sample_ids.append(x["sample_id"])
        target_ids.append(x["target_id"])
        target_structures.append(x["target_complex"])
        feature_structures.append(x["feature_complex"])

    collated_batch: dict[str, dict[str, Tensor] | list[str]] = {
        "target_complex": collate_complex(target_structures),
        "feature_complex": collate_complex(feature_structures),
        "id": ids,
        "sample_id": sample_ids,
        "target_id": target_ids,
    }
    return collated_batch


class PinderDataset(Dataset):  # type: ignore
    def __init__(
        self,
        split: str | None = None,
        index: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        monomer_priority: str = "holo",
        base_filters: list[PinderFilterBase] = [],
        sub_filters: list[PinderFilterSubBase] = [],
        structure_filters: list[StructureFilter] = [],
        structure_transforms: list[StructureTransform] = [],
        transform: Callable[
            [Structure], torch.Tensor | dict[str, torch.Tensor]
        ] = structure2tensor_transform,
        target_transform: Callable[
            [Structure], torch.Tensor | dict[str, torch.Tensor]
        ] = structure2tensor_transform,
        ids: list[str] | None = None,
        fallback_to_holo: bool = True,
        use_canonical_apo: bool = True,
        crop_equal_monomer_shapes: bool = True,
        index_query: str | None = None,
        metadata_query: str | None = None,
        pre_specified_monomers: dict[str, str] | pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        self.loader = PinderLoader(
            split=split,
            ids=ids,
            index=index,
            metadata=metadata,
            base_filters=base_filters,
            sub_filters=sub_filters,
            structure_filters=structure_filters,
            structure_transforms=structure_transforms,
            index_query=index_query,
            metadata_query=metadata_query,
            monomer_priority=monomer_priority,
            fallback_to_holo=fallback_to_holo,
            use_canonical_apo=use_canonical_apo,
            crop_equal_monomer_shapes=crop_equal_monomer_shapes,
            pre_specified_monomers=pre_specified_monomers,
            **kwargs,
        )
        # Optional transform and target transform to apply
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(
        self, idx: int
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        system, feature_complex, target_complex = self.loader[idx]
        sample_id = feature_complex.pinder_id
        target_id = target_complex.pinder_id
        if self.transform is not None:
            feature_complex = self.transform(feature_complex)
        if self.target_transform is not None:
            target_complex = self.target_transform(target_complex)
        data: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            "target_complex": target_complex,
            "feature_complex": feature_complex,
            # for convenience, track original pinder ID and the IDs of the selected monomers in the complex
            "id": system.entry.id,
            "sample_id": sample_id,
            "target_id": target_id,
        }
        return data


class PPIDataset(TorchGeoDataset):  # type: ignore
    def __init__(
        self,
        node_types: set[NodeRepresentation],
        split: str = "train",
        monomer1: str = "holo_receptor",
        monomer2: str = "holo_ligand",
        base_filters: list[PinderFilterBase] = [],
        sub_filters: list[PinderFilterSubBase] = [],
        structure_filters: list[StructureFilter] = [],
        root: Path = get_pinder_location(),
        transform: Callable[[PairedPDB], PairedPDB] | None = None,
        pre_transform: Callable[[PairedPDB], PairedPDB] | None = None,
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
        fallback_to_holo: bool = True,
        crop_equal_monomer_shapes: bool = True,
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
        if split:
            self.pindex = self.pindex.query(f'split == "{self.split}"').reset_index(
                drop=True
            )
        self.monomer1 = monomer1
        self.monomer2 = monomer2
        self.add_edges = add_edges
        self.k = k
        self.parallel = parallel
        self.max_workers = max_workers
        self.fallback_to_holo = fallback_to_holo
        default_file_dir = Path(root) / "filenames"
        self.filenames_dir = default_file_dir
        if filenames_dir:
            self.filenames_dir = Path(filenames_dir)
        self.filenames_dir.mkdir(exist_ok=True)
        self.ids = ids or list(self.pindex.id)
        if force_reload:
            self.ids_to_process = self.ids
        else:
            # Get pre-cached ids to remove from generator
            processed_dir = Path(root) / "processed"
            processed_ids = {x.stem for x in Path(processed_dir).glob("*.pt")}
            self.ids_to_process = list(set(self.ids) - processed_ids)

        R_monomer = monomer1.split("_receptor")[0]
        L_monomer = monomer2.split("_receptor")[0]
        if R_monomer != L_monomer:
            monomer_priority = "random_mixed"
        else:
            monomer_priority = R_monomer
        self.loader = PinderLoader(
            split=split,
            ids=self.ids_to_process,
            index=self.pindex,
            base_filters=base_filters,
            sub_filters=sub_filters,
            structure_filters=structure_filters,
            monomer_priority=monomer_priority,
            fallback_to_holo=fallback_to_holo,
            crop_equal_monomer_shapes=crop_equal_monomer_shapes,
        )
        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

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
        fallback_to_holo: bool = True,
        pre_transform: Callable[[PairedPDB], PairedPDB] | None = None,
    ) -> bool:
        try:
            data = PairedPDB.from_pinder_system(
                system=system,
                monomer1=monomer1,
                monomer2=monomer2,
                node_types=node_types,
                add_edges=add_edges,
                k=k,
                fallback_to_holo=fallback_to_holo,
            )
            if pre_transform is not None:
                data = pre_transform(data)

            torch.save(data, output_file)
            return True
        except Exception as e:
            log.error(f"Unable to process {system.entry.id}: {e}")
            return False

    @staticmethod
    def process_single_file_parallel(
        args: tuple[
            int,
            PinderLoader,
            set[NodeRepresentation],
            Path,
            str,
            str,
            bool,
            int,
            bool,
            bool,
            Callable[[PinderSystem], PinderSystem | bool] | None,
            Callable[[PairedPDB], PairedPDB] | None,
        ],
    ) -> str | None:
        (
            system_idx,
            loader,
            node_types,
            processed_dir,
            monomer1,
            monomer2,
            add_edges,
            k,
            force_reload,
            fallback_to_holo,
            pre_filter,
            pre_transform,
        ) = args

        try:
            system, _, _ = loader[system_idx]
        except Exception as e:
            log.error(str(e))
            system = None

        if not hasattr(system, "entry") or system is None:
            return None
        if pre_filter is not None and not pre_filter(system):
            return None
        pinder_id: str = system.entry.id
        to_write = Path(processed_dir) / f"{pinder_id}.pt"
        if to_write.exists() and force_reload:
            to_write.unlink()
        if to_write.exists():
            return pinder_id
        processed = PPIDataset.process_single_file(
            system,
            node_types,
            monomer1,
            monomer2,
            to_write,
            add_edges,
            k,
            fallback_to_holo,
            pre_transform,
        )
        if processed:
            return pinder_id
        return None

    def process_parallel(self) -> None:
        with multiprocessing.get_context("spawn").Pool(processes=self.max_workers) as p:
            args_gen = (
                (
                    system_index,
                    self.loader,
                    self.node_types,
                    self.processed_dir,
                    self.monomer1,
                    self.monomer2,
                    self.add_edges,
                    self.k,
                    self.force_reload,
                    self.fallback_to_holo,
                    self.pre_filter,
                    self.pre_transform,
                )
                for system_index in range(len(self.loader))
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
            for system_idx in range(len(self.loader)):
                if self.limit_by and len(self.filenames) >= self.limit_by:
                    log.info(f"Finished processing, only {self.limit_by} systems")
                    break
                try:
                    system, _, _ = self.loader[system_idx]
                except Exception as e:
                    log.error(str(e))
                    continue
                if not system:
                    continue
                if self.pre_filter is not None and not self.pre_filter(system):
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
                    self.fallback_to_holo,
                    self.pre_transform,
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
) -> TorchGeoDataLoader:
    return TorchGeoDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )


def get_torch_loader(
    dataset: PinderDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    sampler: "Sampler[PinderDataset]" | None = None,
    num_workers: int = 1,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = collate_batch,
    **kwargs: Any,
) -> "DataLoader[PinderDataset]":
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        **kwargs,
    )
