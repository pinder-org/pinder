from __future__ import annotations
from pinder.core.utils import setup_logger
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure.atoms import AtomArray, AtomArrayStack, stack
from numpy.typing import NDArray


from pinder.core.utils.timer import timeit
from pinder.core.utils.process import process_starmap
from pinder.core.structure import surgery
from pinder.core.structure.atoms import (
    apply_mask,
    assign_receptor_ligand,
    atom_array_from_pdb_file,
    get_seq_aligned_structures,
    get_per_chain_seq_alignments,
    invert_chain_seq_map,
    standardize_atom_array,
)
from pinder.core.structure.contacts import pairwise_contacts
from pinder.core.structure.models import BackboneDefinition, ChainConfig
from pinder.eval.dockq import metrics


log = setup_logger(__name__)


def get_irmsd_interface(
    native: Path,
    chain1: list[str],
    chain2: list[str],
    backbone_only: bool = False,
    heavy_only: bool = False,
    interface_cutoff: float = 10.0,
) -> dict[str, list[int]]:
    """
    Get the residues part of the interface by chain, using a more permissive
    contact distance cutofff definition of 10A.
    """
    contact_pairs = pairwise_contacts(
        native,
        chain1,
        chain2,
        radius=interface_cutoff,
        heavy_only=heavy_only,
        backbone_only=backbone_only,
    )
    chain_map: dict[str, list[int]] = {ch: [] for ch in set(chain1).union(set(chain2))}
    for cp in contact_pairs:
        c1, c2, r1, r2 = cp
        chain_map[c1].append(r1)
        chain_map[c2].append(r2)

    return chain_map


class DecoyDockQ:
    def __init__(
        self,
        native: AtomArray,
        decoy: AtomArray | AtomArrayStack,
        chain_config: ChainConfig,
        backbone_definition: BackboneDefinition,
        native_contacts: set[tuple[str, str, int, int]],
        native_interface: dict[str, list[int]],
    ) -> None:
        self.native = native
        self.decoy_stack = decoy
        self.chain_config = chain_config
        self.backbone_definition = backbone_definition
        self.native_contacts = native_contacts
        self.native_interface = native_interface

    def get_metrics(self) -> pd.DataFrame | dict[str, float | str | int]:
        self.decoy2native_seq = get_per_chain_seq_alignments(
            self.native, self.decoy_stack
        )
        self.native2decoy_seq = invert_chain_seq_map(self.decoy2native_seq)

        decoy_contacts = self.get_decoy_contacts()
        fnat = metrics.calc_fnat(
            decoy_contacts, self.native_contacts, self.chain_config
        )
        irms = self.get_irmsd()
        lrms = self.get_lrmsd()
        if isinstance(self.decoy_stack, AtomArray):
            fnat = fnat[0]
        dockq = metrics.get_dockq_score(irms, lrms, fnat)
        if isinstance(self.decoy_stack, AtomArrayStack):
            assert isinstance(irms, np.ndarray)
            assert isinstance(lrms, np.ndarray)
            df = pd.DataFrame(
                [
                    {
                        "iRMS": ir,
                        "LRMS": lr,
                        "Fnat": fn,
                        "DockQ": dq,
                        "CAPRI": metrics.capri_class(fn, ir, lr),
                        "decoy_contacts": len(decoy_contacts[i]),
                        "native_contacts": len(self.native_contacts),
                    }
                    for i, (ir, lr, fn, dq) in enumerate(zip(irms, lrms, fnat, dockq))
                ]
            )
        else:
            df = {
                "iRMS": irms,
                "LRMS": lrms,
                "Fnat": fnat,
                "DockQ": dockq,
                "CAPRI": metrics.capri_class(fnat, irms, lrms),
                "decoy_contacts": len(decoy_contacts[0]),
                "native_contacts": len(self.native_contacts),
            }
        return df

    @staticmethod
    def _get_ref_interface_in_model_numbering(
        decoy: AtomArrayStack | AtomArray,
        ref_interface: dict[str, list[int]],
        native2decoy_seq: dict[str, dict[int, int]],
    ) -> dict[str, list[int]] | list[dict[str, list[int]]]:
        model2ref_interface: dict[str, list[int]] = {ch: [] for ch in ref_interface}
        assert isinstance(native2decoy_seq, dict)
        for ch, rlist in ref_interface.items():
            model2ref_interface[ch] = [native2decoy_seq[ch].get(r, r) for r in rlist]
        return model2ref_interface

    def get_lrmsd(self) -> NDArray[np.double] | float:
        try:
            lrms = self._calc_lrmsd()
        except Exception as e:
            self.native, self.decoy_stack = get_seq_aligned_structures(
                self.native, self.decoy_stack
            )
            self.decoy2native_seq = get_per_chain_seq_alignments(
                self.native, self.decoy_stack
            )
            lrms = self._calc_lrmsd()
        return lrms

    def get_irmsd(self) -> NDArray[np.double] | float:
        self.model2ref_interface = self._get_ref_interface_in_model_numbering(
            self.decoy_stack, self.native_interface, self.native2decoy_seq
        )
        try:
            irms = self._calc_irmsd(decoy_renumbered=False)
        except Exception as e:
            # Should now use self.native_interface numbering
            self.native, self.decoy_stack = get_seq_aligned_structures(
                self.native, self.decoy_stack
            )
            self.decoy2native_seq = get_per_chain_seq_alignments(
                self.native, self.decoy_stack
            )
            irms = self._calc_irmsd(decoy_renumbered=True)
        return irms

    def _calc_irmsd(self, decoy_renumbered: bool = False) -> NDArray[np.double] | float:
        assert isinstance(self.model2ref_interface, dict)
        assert isinstance(self.decoy2native_seq, dict)
        irms: NDArray[np.double] | float = metrics.calc_irmsd(
            self.decoy_stack,
            self.native,
            self.native_interface if decoy_renumbered else self.model2ref_interface,
            self.native_interface,
            self.decoy2native_seq,
            self.chain_config,
            backbone_definition=self.backbone_definition,
        )
        return irms

    def _calc_lrmsd(self) -> NDArray[np.double] | float:
        assert isinstance(self.decoy2native_seq, dict)
        lrms: NDArray[np.double] | float = metrics.calc_lrmsd(
            self.decoy_stack,
            self.native,
            self.decoy2native_seq,
            self.chain_config,
            backbone_definition=self.backbone_definition,
        )
        return lrms

    @timeit
    def get_decoy_contacts(self) -> list[set[tuple[str, str, int, int]]]:
        decoy_contacts = pairwise_contacts(
            self.decoy_stack,
            self.chain_config.decoy_receptor,
            self.chain_config.decoy_ligand,
        )
        if not isinstance(decoy_contacts, list):
            decoy_contacts = [decoy_contacts]
        renumbered_contacts = []
        for i, pose_contacts in enumerate(decoy_contacts):
            pose_set = set()
            for cp in pose_contacts:
                c1, c2, r1, r2 = cp
                r1_renum = self.decoy2native_seq[c1].get(r1, r1)
                r2_renum = self.decoy2native_seq[c2].get(r2, r2)
                pose_set.add((c1, c2, r1_renum, r2_renum))
            renumbered_contacts.append(pose_set)
        return renumbered_contacts


class BiotiteDockQ:
    """Biotite interface for fast calculation of DockQ and CAPRI metrics.

    Takes as input one native and all it's decoys
    from arbitrary number of methods to compare.
    """

    def __init__(
        self,
        native: Path,
        decoys: list[Path] | Path,
        native_receptor_chain: list[str] | None = None,
        native_ligand_chain: list[str] | None = None,
        decoy_receptor_chain: list[str] | None = None,
        decoy_ligand_chain: list[str] | None = None,
        pdb_engine: str = "fastpdb",
        backbone_definition: BackboneDefinition = "dockq",
        parallel_io: bool = True,
        max_workers: int | None = None,
    ) -> None:
        if not isinstance(decoys, list):
            decoys = [decoys]

        self.native_pdb = native
        self.decoy_pdbs = decoys
        self.pdb_engine = pdb_engine
        self.backbone_definition = backbone_definition
        self.parallel_io = parallel_io
        self.max_workers = max_workers
        self.native = atom_array_from_pdb_file(self.native_pdb)
        self.raw_native_shape = self.native.shape[0]
        # Im puzzled here, since the default DockQ keeps heavy atoms
        # but does nothing about the decoys missing the heavy atoms
        # This would penalize decoys without hydrogens when calculating fnat
        # and iRMSD
        # self.native = filter_atoms(native, calpha_only=False, backbone_only=False, heavy_only=True)
        if not (native_receptor_chain and native_ligand_chain):
            native_receptor_chain, native_ligand_chain = assign_receptor_ligand(
                self.native, set(self.native.chain_id)
            )
        self.native_rec_chain = native_receptor_chain
        self.native_lig_chain = native_ligand_chain
        self.model_rec_chain = decoy_receptor_chain
        self.model_lig_chain = decoy_ligand_chain

    def calculate(self) -> pd.DataFrame:
        self.prepare_inputs()
        if isinstance(self.decoy_stack, AtomArrayStack):
            ddq = DecoyDockQ(
                native=self.native,
                decoy=self.decoy_stack,
                chain_config=self.chain_config,
                backbone_definition=self.backbone_definition,
                native_contacts=self.native_contacts,
                native_interface=self.native_interface,
            )
            dockq_metrics = ddq.get_metrics()
            assert isinstance(dockq_metrics, pd.DataFrame)
            final_decoy_shapes = [arr.shape[0] for arr in ddq.decoy_stack]
            dockq_metrics.loc[:, "initial_decoy_shape"] = self.raw_decoy_shapes
            dockq_metrics.loc[:, "final_decoy_shape"] = final_decoy_shapes
            dockq_metrics.loc[:, "initial_native_shape"] = self.raw_native_shape
            dockq_metrics.loc[:, "final_native_shape"] = ddq.native.shape[0]
        else:
            dockq_metrics = []
            for i, pose in enumerate(self.decoy_stack):
                ddq = DecoyDockQ(
                    native=self.native,
                    decoy=pose,
                    chain_config=self.chain_config,
                    backbone_definition=self.backbone_definition,
                    native_contacts=self.native_contacts,
                    native_interface=self.native_interface,
                )
                pose_metrics = ddq.get_metrics()
                pose_metrics["initial_decoy_shape"] = self.raw_decoy_shapes[i]
                pose_metrics["final_decoy_shape"] = ddq.decoy_stack.shape[0]
                pose_metrics["initial_native_shape"] = self.raw_native_shape
                pose_metrics["final_native_shape"] = ddq.native.shape[0]
                dockq_metrics.append(pose_metrics)
            dockq_metrics = pd.DataFrame(dockq_metrics)

        assert isinstance(dockq_metrics, pd.DataFrame)
        dockq_metrics.loc[:, "model_name"] = [decoy.stem for decoy in self.decoy_pdbs]
        dockq_metrics.loc[:, "native_name"] = self.native_pdb.stem
        dockq_metrics.loc[:, "system"] = self.native_pdb.stem
        dockq_metrics.loc[:, "method"] = self.native_pdb.parent.parent.stem
        dockq_metrics.loc[:, "model_folder"] = self.decoy_pdbs[0].parent.stem
        col_order = [
            "model_name",
            "native_name",
            "system",
            "method",
            "model_folder",
            "iRMS",
            "LRMS",
            "Fnat",
            "DockQ",
            "CAPRI",
            "decoy_contacts",
            "native_contacts",
            "initial_decoy_shape",
            "final_decoy_shape",
            "initial_native_shape",
            "final_native_shape",
        ]
        self.metrics = dockq_metrics[col_order].copy()

        return self.metrics

    def get_native_contacts(self) -> None:
        # Used to define interface for iRMSD
        assert isinstance(self.native_rec_chain, list)
        assert isinstance(self.native_lig_chain, list)
        self.native_interface = get_irmsd_interface(
            self.native,
            self.native_rec_chain,
            self.native_lig_chain,
            backbone_only=False,
        )
        # Used to define Fnat
        self.native_contacts = pairwise_contacts(
            self.native,
            self.native_rec_chain,
            self.native_lig_chain,
        )

    def prepare_inputs(self) -> None:
        arr_list, decoy_pdbs = self.read_decoys(
            self.decoy_pdbs,
            self.pdb_engine,
            self.parallel_io,
            self.max_workers,
        )
        self.decoy_pdbs = decoy_pdbs
        if not (self.model_rec_chain and self.model_lig_chain):
            # Determine chain based on atom count
            self.model_rec_chain, self.model_lig_chain = assign_receptor_ligand(
                arr_list[0], set(arr_list[0].chain_id)
            )

        assert isinstance(self.model_rec_chain, list)
        assert isinstance(self.model_lig_chain, list)
        assert isinstance(self.native_rec_chain, list)
        assert isinstance(self.native_lig_chain, list)

        chain_remap = {
            existing: new
            for existing, new in zip(self.model_rec_chain, self.native_rec_chain)
        }
        for existing, new in zip(self.model_lig_chain, self.native_lig_chain):
            chain_remap[existing] = new

        # Re-name chains to match native.
        # filter_intersect causes differing chains to cause annotation mismatch
        self.chain_config = ChainConfig(
            decoy_receptor=self.native_rec_chain,
            decoy_ligand=self.native_lig_chain,
            native_receptor=self.native_rec_chain,
            native_ligand=self.native_lig_chain,
        )
        all_chains = self.native_rec_chain + self.native_lig_chain
        self.native = self.native[np.isin(self.native.chain_id, all_chains)]
        log.debug(f"Will use the following chain pairings:\n{self.chain_config}")

        self.raw_decoy_shapes = [arr.shape[0] for arr in arr_list]
        self.decoy_stack, self.native = self.create_decoy_stack(
            arr_list,
            self.native,
            R_chain=self.model_rec_chain,
            L_chain=self.model_lig_chain,
        )
        # Contacts need to be calculated here in case native renumbered above
        self.get_native_contacts()
        if isinstance(self.decoy_stack, AtomArrayStack):
            self.decoy_stack.chain_id = np.array(
                [chain_remap.get(ch, ch) for ch in self.decoy_stack.chain_id]
            )
            chain_mask = np.isin(self.decoy_stack.chain_id, all_chains)
            self.decoy_stack = apply_mask(self.decoy_stack, chain_mask)
        else:
            for i, arr in enumerate(self.decoy_stack):
                arr.chain_id = np.array(
                    [chain_remap.get(ch, ch) for ch in arr.chain_id]
                )
                arr = apply_mask(arr, np.isin(arr.chain_id, all_chains))
                self.decoy_stack[i] = arr.copy()
        self.set_common()

    @staticmethod
    @timeit
    def read_decoys(
        decoys: list[Path],
        pdb_engine: str,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> tuple[list[AtomArray], list[Path]]:
        from itertools import repeat

        arr_list = process_starmap(
            _atom_array_from_pdb_file,
            zip(decoys, repeat(pdb_engine)),
            parallel=parallel,
            max_workers=max_workers,
        )
        valid_arr = []
        valid_decoys = []
        for arr, decoy in zip(arr_list, decoys):
            if arr is not None:
                if len(set(arr.chain_id)) > 1:
                    valid_arr.append(arr)
                    valid_decoys.append(decoy)

        arr_ordered = [(i, arr) for i, arr in enumerate(valid_arr)]
        arr_ordered = sorted(arr_ordered, key=lambda x: x[1].shape[0])
        ordered_decoys = []
        ordered_arrays = []
        for i, arr in arr_ordered:
            ordered_decoys.append(valid_decoys[i])
            ordered_arrays.append(arr)
        return ordered_arrays, ordered_decoys

    @staticmethod
    def create_decoy_stack(
        arr_list: list[AtomArray],
        native: AtomArray,
        R_chain: list[str],
        L_chain: list[str],
    ) -> tuple[AtomArrayStack, AtomArray]:
        try:
            # All annotations are equal and atom counts identical
            decoy_stack = stack(arr_list)
        except Exception:
            try:
                log.info(
                    "Couldnt create stack. Attempting chain and atom re-ordering..."
                )
                # First check if standardizing chain + atom order can make annotation arrays equal
                for i, arr in enumerate(arr_list):
                    arr_R = arr[np.isin(arr.chain_id, R_chain)].copy()
                    arr_L = arr[np.isin(arr.chain_id, L_chain)].copy()
                    arr_RL = arr_R + arr_L
                    arr_ordered = standardize_atom_array(arr_RL)
                    arr_list[i] = arr_ordered.copy()
                decoy_stack = stack(arr_list)
                log.info("Successfully created stack after standardizing order")
            except Exception as e:
                log.warning(
                    "Models have unequal annotations and/or shapes, not using vectorized AtomArrayStack"
                )
                decoy_stack = arr_list
                # # Requires slower curation of intersecting atoms before stacking
                # decoy_stack, standardize_order = stack_filter_intersect(arr_list)
                # nat = native.copy()
                # if hasattr(nat, "atom_id"):
                #     nat.set_annotation("atom_id", np.repeat(0, nat.shape[0]))
                # native = standardize_atom_array(nat, standardize_order)
        return decoy_stack, native

    @timeit
    def set_common(self) -> None:
        # Ensure chain order is same between native and decoys
        # This still doesn't guarantee chain order is same for multi-chain case
        self.native = surgery.set_canonical_chain_order(
            self.native, self.chain_config, "native"
        )
        self.decoy_stack = surgery.set_canonical_chain_order(
            self.decoy_stack, self.chain_config, "decoy"
        )


def _atom_array_from_pdb_file(
    structure: Path | AtomArray, backend: str
) -> AtomArray | None:
    """Like `atom_array_from_pdb_file()` but an exception leads to `None` as return value."""
    try:
        return atom_array_from_pdb_file(structure, backend)
    except Exception as e:
        return None
