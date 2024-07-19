from __future__ import annotations

import biotite.structure as struc
import numpy as np
from biotite.structure.atoms import AtomArray, AtomArrayStack
from numpy.typing import NDArray
from pinder.core.structure import surgery
from pinder.core.structure.atoms import apply_mask, get_backbone_atom_masks
from pinder.core.structure.models import BackboneDefinition, ChainConfig


def get_interface_mask(
    structure: AtomArray | AtomArrayStack,
    interface: dict[str, list[int]],
    chains: ChainConfig,
    subject: str,
) -> NDArray[np.bool_]:
    # Create set of residues in interface split into receptor and ligand
    # Conflict between residue numbers in different chains is handled by
    # logical mask on array.chain_id and array.res_id
    lig_chains = getattr(chains, f"{subject}_ligand")
    rec_chains = getattr(chains, f"{subject}_receptor")
    lig_res: set[int] = set()
    for ch in lig_chains:
        lig_res = lig_res.union(interface[ch])

    rec_res: set[int] = set()
    for ch in rec_chains:
        rec_res = rec_res.union(interface[ch])

    # Cast to list, otherwise numpy.isin will convert non-sequence collections
    # to one-element object arrays
    rec_res_lst: list[int] = list(rec_res)
    lig_res_lst: list[int] = list(lig_res)
    interface_mask = (
        np.isin(structure.chain_id, lig_chains) & np.isin(structure.res_id, lig_res_lst)
    ) | (
        np.isin(structure.chain_id, rec_chains) & np.isin(structure.res_id, rec_res_lst)
    )
    return interface_mask


def calc_irmsd(
    decoy_stack: AtomArrayStack,
    native: AtomArray,
    decoy_interface: dict[str, list[int]],
    native_interface: dict[str, list[int]],
    decoy2native_res: dict[str, dict[int, int]],
    chains: ChainConfig,
    backbone_only: bool = True,
    calpha_only: bool = False,
    backbone_definition: BackboneDefinition = "default",
) -> NDArray[np.double] | float:
    """Get interface RMSD after superposition of interface backbone atoms."""
    ref_interface_mask = get_interface_mask(native, native_interface, chains, "native")
    mod_interface_mask = get_interface_mask(
        decoy_stack, decoy_interface, chains, "decoy"
    )
    model_common = apply_mask(decoy_stack, mod_interface_mask)
    native_common = native[ref_interface_mask]

    # superimpose
    native_mask, model_mask = get_backbone_atom_masks(
        native_common, model_common, backbone_only, calpha_only, backbone_definition
    )

    model_common = apply_mask(model_common, model_mask)
    native_common = native_common[native_mask]

    # Even if atom counts are identical, annotation categories must be the same
    native_common, model_common = surgery.fix_annotation_mismatch(
        native_common, model_common
    )

    # Ensure numbering is the same
    # For non-overlapping, use -99999 as residue to ensure
    # it doesnt match native when filtering for intersect
    ref_model = (
        model_common[0] if isinstance(model_common, AtomArrayStack) else model_common
    )
    new_res_ids = np.array(
        [decoy2native_res[at.chain_id].get(at.res_id, -99999) for at in ref_model]
    )
    model_common.res_id = new_res_ids

    # There still may be mismatch in atom counts, filter
    native_common = native_common[
        struc.filter_intersection(native_common, model_common)
    ]

    model_mask = struc.filter_intersection(model_common, native_common)
    model_common = apply_mask(model_common, model_mask)
    fitted, transformation = struc.superimpose(native_common, model_common)

    # Make sure the arrays have atoms
    # If assertion is raised, will be retried after sequence alignment
    assert native_common.shape[0]
    shape_idx = int(isinstance(fitted, AtomArrayStack))
    assert fitted.shape[shape_idx]

    irms: NDArray[np.double] | float = struc.rmsd(
        native_common,
        fitted,
    )
    return irms


def get_receptor_masks(
    native: AtomArray,
    decoy_stack: AtomArrayStack,
    chains: ChainConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    decoy_R_mask = np.isin(decoy_stack.chain_id, chains.decoy_receptor)
    native_R_mask = np.isin(native.chain_id, chains.native_receptor)
    return native_R_mask, decoy_R_mask


def calc_lrmsd(
    decoy_stack: AtomArrayStack | AtomArray,
    native: AtomArray,
    decoy2native_res: dict[str, dict[int, int]],
    chains: ChainConfig,
    backbone_only: bool = True,
    calpha_only: bool = False,
    backbone_definition: BackboneDefinition = "dockq",
) -> NDArray[np.double] | float:
    """Get ligand body Calpha or backbone RMSD."""

    # Even if atom counts are identical, annotation categories must be the same
    native, decoy_stack = surgery.fix_annotation_mismatch(native, decoy_stack)

    # Ensure numbering is the same
    # For non-overlapping, use -99999 as residue to ensure
    # it doesnt match native when filtering for intersect
    if isinstance(decoy_stack, AtomArrayStack):
        ref_model = decoy_stack[0]
    else:
        ref_model = decoy_stack

    new_res_ids = np.array(
        [decoy2native_res[at.chain_id].get(at.res_id, -99999) for at in ref_model]
    )
    decoy_stack.res_id = new_res_ids

    # There still may be mismatch in atom counts, filter
    native_common = native[struc.filter_intersection(native, decoy_stack)]

    model_mask = struc.filter_intersection(decoy_stack, native_common)
    model_common = apply_mask(decoy_stack, model_mask)
    # Make sure the arrays have atoms
    # If assertion is raised, will be retried after sequence alignment
    assert native_common.shape[0]
    shape_idx = int(isinstance(model_common, AtomArrayStack))
    assert model_common.shape[shape_idx]

    native_atom_mask, decoy_atom_mask = get_backbone_atom_masks(
        native_common, model_common, backbone_only, calpha_only, backbone_definition
    )
    native_R_mask, decoy_R_mask = get_receptor_masks(
        native_common, model_common, chains
    )
    # superimpose receptor using
    fitted, transformation = struc.superimpose(
        native_common[native_R_mask & native_atom_mask],
        apply_mask(model_common, (decoy_R_mask & decoy_atom_mask)),
    )
    mobile_at = model_common.copy()
    mobile_at.coord = transformation.apply(mobile_at).coord

    # Perform RMSD calculation on receptor-superimposed stack with inverse
    # chain mask for ligand chain(s)
    ref_L = native_common[(~native_R_mask) & native_atom_mask]
    mobile_L = apply_mask(mobile_at, ((~decoy_R_mask) & decoy_atom_mask))
    assert ref_L.shape[0]
    assert mobile_L.shape[shape_idx]
    lrms: NDArray[np.double] | float = struc.rmsd(
        ref_L,
        mobile_L,
    )
    return lrms


def calc_fnat(
    decoy_contacts: set[tuple[str, str, int, int]],
    native_contacts: set[tuple[str, str, int, int]],
    chains: ChainConfig,
) -> NDArray[np.double]:
    """Get fraction of native contacts.
    Parameters
    ----------
    decoy_stack : AtomArrayStack
        Stack of AtomArray containing decoys (models)
    native_contacts : set[tuple[str, str, int, int]]
        Residue contact pairs in native. See dockq.contacts.pairwise_contacts
    chains : ChainConfig
        Chain config object defining receptor and ligand chain pairings.

    Returns
    -------
    ndarray
         Numpy array of fnat values for each decoy in the stack.
    """
    n_nat = [len(native_contacts.intersection(decoy_c)) for decoy_c in decoy_contacts]
    n_tot = len(native_contacts)
    return np.array(n_nat) / n_tot


def get_dockq_score(
    i_rmsd: NDArray[np.double] | float,
    l_rmsd: NDArray[np.double] | float,
    fnat: NDArray[np.double] | float,
) -> NDArray[np.double] | float:
    """Get DockQ scores
    Parameters
    ----------
    i_rmsd : ndarray | float
        interface rmsd
    l_rmsd : ndarray | float
        ligand rmsd
    fnat : ndarray | float
        fraction of native contact
    Returns
    -------
    ndarray | float
        DockQ score
    """
    return (
        fnat
        + 1 / (1 + (i_rmsd / 1.5) * (i_rmsd / 1.5))
        + 1 / (1 + (l_rmsd / 8.5) * (l_rmsd / 8.5))
    ) / 3


def capri_class(
    fnat: float, iRMS: float, LRMS: float, capri_peptide: bool = False
) -> str:
    if capri_peptide:
        if fnat < 0.2 or (LRMS > 5.0 and iRMS > 2.0):
            return "Incorrect"
        elif (
            (fnat >= 0.2 and fnat < 0.5)
            and (LRMS <= 5.0 or iRMS <= 2.0)
            or (fnat >= 0.5 and LRMS > 2.0 and iRMS > 1.0)
        ):
            return "Acceptable"
        elif (
            (fnat >= 0.5 and fnat < 0.8)
            and (LRMS <= 2.0 or iRMS <= 1.0)
            or (fnat >= 0.8 and LRMS > 1.0 and iRMS > 0.5)
        ):
            return "Medium"
        elif fnat >= 0.8 and (LRMS <= 1.0 or iRMS <= 0.5):
            return "High"
        else:
            return "Undef"
    else:
        if fnat < 0.1 or (LRMS > 10.0 and iRMS > 4.0):
            return "Incorrect"
        elif (
            (fnat >= 0.1 and fnat < 0.3)
            and (LRMS <= 10.0 or iRMS <= 4.0)
            or (fnat >= 0.3 and LRMS > 5.0 and iRMS > 2.0)
        ):
            return "Acceptable"
        elif (
            (fnat >= 0.3 and fnat < 0.5)
            and (LRMS <= 5.0 or iRMS <= 2.0)
            or (fnat >= 0.5 and LRMS > 1.0 and iRMS > 1.0)
        ):
            return "Medium"
        elif fnat >= 0.5 and (LRMS <= 1.0 or iRMS <= 1.0):
            return "High"
        else:
            return "Undef"
