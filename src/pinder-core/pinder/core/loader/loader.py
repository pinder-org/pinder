from __future__ import annotations
import random
from collections.abc import Generator
from itertools import chain, islice
from typing import Iterable

import pandas as pd
from tqdm import tqdm
from pinder.core.index.utils import get_index, get_metadata
from pinder.core.index.system import _align_monomers_with_mask, PinderSystem
from pinder.core.loader.filters import (
    PinderFilterSubBase,
    PinderFilterBase,
    StructureFilter,
)
from pinder.core.loader.transforms import StructureTransform
from pinder.core.loader.writer import PinderWriterBase
from pinder.core.utils.log import setup_logger
from pinder.core.structure.models import DatasetName
from pinder.core.loader.structure import Structure


ALLOWED_MONOMER_PRIORITIES = ["apo", "holo", "pred", "random", "random_mixed"]
log = setup_logger(__name__)


def get_systems(systems: list[str]) -> Generator[PinderSystem, PinderSystem, None]:
    for system in tqdm(systems):
        try:
            system = PinderSystem(system)
            yield system
        except Exception as e:
            log.error(str(e))
            continue


def get_available_monomers(row: pd.Series) -> dict[str, list[str]]:
    """Get the available monomers for a given row of the index.

    Parameters:
        row (pd.Series): A row of the pinder index representing a pinder system.

    Returns:
        dict[str, list[str]]: A dictionary mapping dimer body (R or L) to a list of available monomer types (apo, predicted, or holo).

    """
    available_monomers: dict[str, list[str]] = {"R": ["holo"], "L": ["holo"]}
    for side in ["R", "L"]:
        for monomer in ["apo", "predicted"]:
            if row[f"{monomer}_{side}"]:
                monomer_key = "pred" if monomer == "predicted" else monomer
                available_monomers[side].append(monomer_key)
    return available_monomers


def get_alternate_apo_codes(row: pd.Series, side: str) -> list[str]:
    """Get the list of non-canonical (alternate) apo PDB codes for the specified dimer side (R or L).

    Parameters:
        row (pd.Series): A row of the pinder index representing a pinder system.
        side (str): The dimer side, R or L, representing receptor or ligand, respectively.

    Returns:
        list[str]: A list of 4-letter PDB codes for all alternate apo monomers (when available).
            The codes can be used to select an alternate apo monomer when working with `PinderSystem` objects.
            When no alternate apo monomers exist, returns an empty list.

    """
    alternate_apo = []
    if row[f"apo_{side}"]:
        alternate_apo.extend(
            [
                pdb.split("__")[0]
                for pdb in row[f"apo_{side}_pdbs"].split(";")
                if pdb not in ["", row[f"apo_{side}_pdb"]]
            ]
        )
    return alternate_apo


def select_monomer(
    row: pd.Series,
    monomer_priority: str = "holo",
    fallback_to_holo: bool = True,
    canonical_apo: bool = True,
) -> dict[str, str]:
    """Select a monomer type to use for the receptor and ligand in a given pinder dimer system.

    Parameters:
        row (pd.Series): A row of the pinder index representing a pinder system.
        monomer_priority (str, optional): The monomer priority to use. Defaults to "holo".
            Allowed values are "apo", "holo", "pred", "random" or "random_mixed"..
            See note about the random and random_mixed options.
        fallback_to_holo (bool, optional): Whether to fallback to the holo monomer when no other monomer is available. Defaults to True.
        canonical_apo (bool, optional): Whether to use the canonical apo monomer when the apo monomer type is available and selected.
            Defaults to True. To sample non-canonical apo monomers, set this value to False.

    Returns:
        dict[str, str]: A dictionary mapping dimer body (R or L) to the selected monomer type (apo, predicted, or holo).
            If non-canonical apo monomers are selected, the dictionary values will point to the apo PDB code to load.
            See `PinderSystem` for more details on how the apo PDB code is used.
    Note:
        The allowed values for `monomer_priority` are "apo", "holo", "pred", "random" or "random_mixed".

        When `monomer_priority` is set to one of the available monomer types (holo, apo, pred), the same monomer type will be selected for both receptor and ligand.

        When the monomer priority is "random", a random monomer type will be selected from the set of monomer types available for both the receptor and ligand. This option ensures the same type of monomer is used for the receptor and ligand.

        When the monomer priority is "random_mixed", a random monomer type will be selected for each of receptor and ligand, separately.

        Enabling the `fallback_to_holo` option (default) will enable silent fallback to holo when the `monomer_priority` is set to one of apo or pred, but the corresponding monomer is not available for the dimer.
        This is useful when only one of receptor or ligand has an unbound monomer, but you wish to include apo or predicted structures in your workflow.
        If `fallback_to_holo` is disabled, an error will be raised when the `monomer_priority` is set to one of apo or pred, but the corresponding monomer is not available for the dimer.

    """
    available_monomers = get_available_monomers(row)
    paired_monomers = set(available_monomers["R"]).intersection(
        set(available_monomers["L"])
    )
    if monomer_priority == "random_mixed":
        R_selection = random.choice(available_monomers["R"])
        L_selection = random.choice(available_monomers["L"])
    elif monomer_priority == "random":
        R_selection = L_selection = random.choice(list(paired_monomers))
    elif monomer_priority in paired_monomers:
        R_selection = monomer_priority
        L_selection = monomer_priority
    elif fallback_to_holo:
        R_selection = L_selection = "holo"
    else:
        raise ValueError(
            f"Unable to find suitable monomers for {row.id} with monomer_priority={monomer_priority} and fallback_to_holo was False!"
        )

    # If an alternate is available, select a random one, otherwise use canonical
    if R_selection == "apo" and not canonical_apo:
        alt_receptor = get_alternate_apo_codes(row, "R")
        if alt_receptor:
            R_selection = random.choice(alt_receptor)
    if L_selection == "apo" and not canonical_apo:
        alt_ligand = get_alternate_apo_codes(row, "L")
        if alt_ligand:
            L_selection = random.choice(alt_ligand)

    selected_monomers = {
        "R": R_selection,
        "L": L_selection,
    }
    return selected_monomers


def _create_target_feature_complex(
    system: PinderSystem,
    selected_monomers: dict[str, str],
    crop_equal_monomer_shapes: bool = True,
    fallback_to_holo: bool = True,
) -> tuple[Structure, Structure]:
    holo_R = system.aligned_holo_R
    holo_L = system.aligned_holo_L
    if selected_monomers["R"] not in ["holo", "apo", "pred"]:
        # Alternative apo monomer is available and requested
        system.apo_receptor_pdb_code = selected_monomers["R"]
        selected_monomers["R"] = "apo"
    if selected_monomers["L"] not in ["holo", "apo", "pred"]:
        # Alternative apo monomer is available and requested
        system.apo_ligand_pdb_code = selected_monomers["L"]
        selected_monomers["L"] = "apo"

    decoy_R = getattr(system, selected_monomers["R"] + "_receptor")
    decoy_L = getattr(system, selected_monomers["L"] + "_ligand")
    # Check to ensure that the monomer wasn't filtered out by a PinderFilterSubBase
    if decoy_R is None and fallback_to_holo:
        selected_monomers["R"] = "holo"
    if decoy_L is None and fallback_to_holo:
        selected_monomers["L"] = "holo"
        decoy_L = getattr(system, selected_monomers["L"] + "_ligand")
    if decoy_R is None or decoy_L is None:
        raise ValueError(
            f"No valid monomers found for {system.entry.id} with selected_monomers={selected_monomers}"
        )
    both_holo = all([selection == "holo" for selection in selected_monomers.values()])
    # No need to crop since our target and feature complex is identical
    if crop_equal_monomer_shapes and not both_holo:
        holo_R, decoy_R = _align_monomers_with_mask(holo_R, decoy_R)
        holo_L, decoy_L = _align_monomers_with_mask(holo_L, decoy_L)
    target_complex = holo_R + holo_L
    feature_complex = decoy_R + decoy_L
    return target_complex, feature_complex


class PinderLoader:
    def __init__(
        self,
        split: str | None = None,
        ids: list[str] | None = None,
        index: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        subset: DatasetName | None = None,
        base_filters: list[PinderFilterBase] = [],
        sub_filters: list[PinderFilterSubBase] = [],
        structure_filters: list[StructureFilter] = [],
        structure_transforms: list[StructureTransform] = [],
        index_query: str | None = None,
        metadata_query: str | None = None,
        writer: PinderWriterBase | None = None,
        monomer_priority: str = "holo",
        fallback_to_holo: bool = True,
        use_canonical_apo: bool = True,
        crop_equal_monomer_shapes: bool = True,
        max_load_attempts: int = 10,
        pre_specified_monomers: dict[str, str] | pd.DataFrame | None = None,
    ) -> None:
        """Initialize a PinderLoader instance.

        Parameters:
            split (str, optional): The split of the pinder dataset to load. Defaults to None.
            ids (list[str], optional): A list of specific pinder system IDs to load. Defaults to None.
            index (pd.DataFrame, optional): The pinder index to load. Defaults to None and retrieved via `get_index` if not provided.
            metadata (pd.DataFrame, optional): The pinder metadata to load. Defaults to None and retrieved via `get_metadata` if not provided.
            subset (DatasetName, optional): An optional test subset (pinder_xl/pinder_s/pinder_af2) to load. Defaults to None.
            base_filters (list[PinderFilterBase], optional): A list of system-level filters (PinderFilterBase) to apply to the pinder system.
                Defaults to [].
            sub_filters (list[PinderFilterSubBase], optional): A list of substructure filters (PinderFilterSubBase) to apply to the pinder system.
                Defaults to [].
            structure_filters (list[StructureFilter], optional): A list of structure filters to apply to the target and feature complexes created from the pinder system.
                Defaults to [].
            structure_transforms (list[StructureTransform], optional): A list of structure transforms to apply to the target and feature complexes created from the pinder system.
                Defaults to [].
            index_query (str, optional): A query string to apply to the pinder index. Defaults to None.
            metadata_query (str, optional): A query string to apply to the pinder metadata. Defaults to None.
            writer (PinderWriterBase, optional): A writer to use to write the pinder system. Defaults to None.
            monomer_priority (str, optional): The monomer priority to use. Defaults to "holo".
                Allowed values are "apo", "holo", "pred", "random" or "random_mixed"..
                See note about the random and random_mixed options.
            fallback_to_holo (bool, optional): Whether to fallback to the holo monomer when no other monomer is available. Defaults to True.
            use_canonical_apo (bool, optional): Whether to use the canonical apo monomer when the apo monomer type is available and selected.
                Defaults to True. To sample non-canonical apo monomers, set this value to False.
            crop_equal_monomer_shapes (bool, optional): Whether to crop the holo and feature (decoy, potentially with apo/pred monomers) complexes to the same shape. Defaults to True.
                See the tutorial on cropped superposition for more details: https://pinder-org.github.io/pinder/superposition.html.
            max_load_attempts (int): Maximum number of times to try loading an item from the index in the __getitem__ call before raising an IndexError.
                Default is 10. When a system is removed by one of the filters, the default behavior is to try another index selected at random.
                This is done to prevent exceptions or NoneType objects being returned to the data loader.
            pre_specified_monomers (dict[str, str], pd.DataFrame, optional): Optional pre-specified monomers to use for each pinder system ID.
                This argument can either be a dictionary or a pandas DataFrame, specifying a mapping of system IDs to monomer types.
                If a dictionary is provided, it should be in the format {"system_id": "monomer_type"}.
                If a DataFrame is provided, it should have a "id" column and a "monomer" column.
                The "id" column should contain the system IDs, and the "monomer" column should contain the monomer types (either "holo", "apo", or "pred").
                Default is None, indicating that monomer_priority will be used to select monomers for each system.

        Note:
            The allowed values for `monomer_priority` are "apo", "holo", "pred", "random" or "random_mixed".

            When `monomer_priority` is set to one of the available monomer types (holo, apo, pred), the same monomer type will be selected for both receptor and ligand.

            When the monomer priority is "random", a random monomer type will be selected from the set of monomer types available for both the receptor and ligand. This option ensures the same type of monomer is used for the receptor and ligand.

            When the monomer priority is "random_mixed", a random monomer type will be selected for each of receptor and ligand, separately.

            Enabling the `fallback_to_holo` option (default) will enable silent fallback to holo when the `monomer_priority` is set to one of apo or pred, but the corresponding monomer is not available for the dimer.
            This is useful when only one of receptor or ligand has an unbound monomer, but you wish to include apo or predicted structures in your workflow.
            If `fallback_to_holo` is disabled, an error will be raised when the `monomer_priority` is set to one of apo or pred, but the corresponding monomer is not available for the dimer.

        """
        if all([x is None for x in [split, ids, index, metadata, subset]]):
            raise ValueError(
                "Must provide at least one of split, ids, index, metadata, subset"
            )
        assert (
            monomer_priority in ALLOWED_MONOMER_PRIORITIES
        ), f"Invalid monomer_priority={monomer_priority}. Allowed values are {ALLOWED_MONOMER_PRIORITIES}"
        # Which split/mode we are using for the dataset instance
        self.split = split
        self.monomer_priority = monomer_priority
        self.fallback_to_holo = fallback_to_holo
        self.use_canonical_apo = use_canonical_apo
        self.crop_equal_monomer_shapes = crop_equal_monomer_shapes
        self.max_load_attempts = max_load_attempts

        # Optional structure filters to apply
        self.base_filters = base_filters
        self.sub_filters = sub_filters
        self.structure_filters = structure_filters
        # Optional structure transforms to apply
        self.structure_transforms = structure_transforms
        self.index_query = index_query
        self.metadata_query = metadata_query
        self.writer = writer
        self.subset = subset

        if index is None:
            index = get_index()
        if metadata is None:
            metadata = get_metadata()

        # Define the subset of the pinder index and metadata corresponding to the split of our dataset instance
        if split:
            index = index.query(f'split == "{split}"')
        if subset:
            if not isinstance(subset, DatasetName):
                subset = DatasetName(subset)
            index = index.query(subset.value)
        if ids:
            index = index[index["id"].isin(set(ids))].reset_index(drop=True)
        if index_query:
            try:
                index = index.query(index_query).reset_index(drop=True)
            except Exception as e:
                log.error(f"Failed to apply index_query={index_query}: {e}")

        if metadata_query:
            try:
                metadata = metadata.query(metadata_query).reset_index(drop=True)
            except Exception as e:
                log.error(f"Failed to apply metadata_query={metadata_query}: {e}")

        final_ids = set(index.id).intersection(set(metadata.id))
        self.index = index[index["id"].isin(final_ids)].reset_index(drop=True)
        self.metadata = metadata[metadata["id"].isin(final_ids)].reset_index(drop=True)
        if len(self.index) == 0:
            raise ValueError(
                f"No systems found matching split={split}, ids={ids}, subset={subset}"
            )

        if isinstance(pre_specified_monomers, pd.DataFrame):
            err_msg = f"pre_specified_monomers DataFrame must have columns 'id' and 'monomer', got {pre_specified_monomers.columns}"
            assert {"id", "monomer"}.issubset(
                set(pre_specified_monomers.columns)
            ), err_msg
            pre_specified_monomers = {
                pid: monomer
                for pid, monomer in zip(
                    pre_specified_monomers["id"], pre_specified_monomers["monomer"]
                )
            }
        if isinstance(pre_specified_monomers, dict):
            monomer_names = set(pre_specified_monomers.values())
            pre_specified_ids = set(pre_specified_monomers.keys())
            err_msg = f"pre_specified_monomers dict must only contain values: 'holo', 'apo', or 'pred', got {monomer_names}"
            assert monomer_names.issubset({"holo", "apo", "pred"}), err_msg
            missing = len(set(self.index.id).difference(pre_specified_ids))
            err_msg = f"pre_specified_monomers dict must map all IDs to be loaded to monomer types, missing {missing} IDs"
            assert pre_specified_ids.issuperset(set(self.index.id)), err_msg
        self.pre_specified_monomers = pre_specified_monomers

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(
        self,
    ) -> Generator[tuple[PinderSystem, Structure, Structure], None, None]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> tuple[PinderSystem, Structure, Structure]:
        valid_idx = False
        attempts = 0
        while not valid_idx and attempts < self.max_load_attempts:
            attempts += 1
            row = self.index.iloc[idx]
            system = PinderSystem(row.id)
            system = self.apply_dimer_filters(
                system, self.base_filters, self.sub_filters
            )
            if not isinstance(system, PinderSystem):
                continue

            if self.pre_specified_monomers is not None:
                pre_selected = self.pre_specified_monomers.get(
                    system.entry.id,
                    "holo" if self.fallback_to_holo else self.monomer_priority,
                )
                selected_monomers = {"R": pre_selected, "L": pre_selected}
            else:
                selected_monomers = select_monomer(
                    row,
                    self.monomer_priority,
                    self.fallback_to_holo,
                    self.use_canonical_apo,
                )
            target_complex, feature_complex = _create_target_feature_complex(
                system,
                selected_monomers,
                self.crop_equal_monomer_shapes,
                self.fallback_to_holo,
            )
            valid_idx = self.apply_structure_filters(target_complex)
            if not valid_idx:
                idx = random.choice(list(range(len(self))))

        if not valid_idx:
            raise IndexError(
                f"Unable to find a valid item in the dataset satisfying filters at {idx} after {attempts} attempts!"
            )
        for transform in self.structure_transforms:
            target_complex = transform(target_complex)
            feature_complex = transform(feature_complex)
        if self.writer:
            self.writer.write(system)
        return system, feature_complex, target_complex

    def apply_structure_filters(self, structure: Structure) -> bool:
        pass_filters = True
        for structure_filter in self.structure_filters:
            if not structure_filter(structure):
                pass_filters = False
                break
        return pass_filters

    @staticmethod
    def apply_dimer_filters(
        dimer: PinderSystem,
        base_filters: list[PinderFilterBase] | list[None] = [],
        sub_filters: list[PinderFilterSubBase] | list[None] = [],
    ) -> PinderSystem | bool:
        for sub_filter in sub_filters:
            if isinstance(sub_filter, PinderFilterSubBase):
                try:
                    dimer = sub_filter(dimer)
                except Exception as e:
                    # Likely due to previous filter removing a holo Structure from the system
                    log.error(
                        f"Failed to apply sub_filter={sub_filter} on {dimer.entry.id}: {e}"
                    )
                    return False
        for base_filter in base_filters:
            if isinstance(base_filter, PinderFilterBase):
                if not base_filter(dimer):
                    return False
        return dimer

    def __repr__(self) -> str:
        return f"PinderLoader(split={self.split}, monomers={self.monomer_priority}, systems={len(self)})"
