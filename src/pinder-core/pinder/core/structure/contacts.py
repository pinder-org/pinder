from __future__ import annotations
from pathlib import Path
from itertools import combinations
from typing import List, NamedTuple, Set, Tuple, Union
import pandas as pd
import numpy as np
import biotite.structure as struc
from biotite.structure.atoms import AtomArray, AtomArrayStack, stack
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from pinder.core.structure.atoms import (
    atom_array_from_pdb_file,
    atom_vdw_radius,
    filter_atoms,
)
from pinder.core.utils import setup_logger


_Contacts = Set[Tuple[str, str, int, int]]
_StackContacts = List[_Contacts]
_AtomResContacts = Union[Tuple[_Contacts, _Contacts], _Contacts]
_StackAtomResContacts = Union[Tuple[_StackContacts, _StackContacts], _StackContacts]


class ContactPairs(NamedTuple):
    residue_contacts: _Contacts
    atom_contacts: _Contacts


log = setup_logger(__name__)


def get_atoms_within_coordinate(
    atom_array: AtomArray | NDArray[np.double],
    coords: NDArray[np.double],
    radius: float,
    cell_size: float,
    as_mask: bool = False,
) -> NDArray[np.int_]:
    """
    Find atom indices (contact matrix) within distance of coords.

    Parameters
    ----------
    atom_array : AtomArray | NDArray[np.double]
        The array of atoms representing the protein structure.
    coords : NDArray[np.double]
        The coordinates to find atoms within.
    radius : float
        The radius within which to consider atoms as being in contact.
    cell_size : float
        The size of the cell for the cell list.
    as_mask : bool, optional
        Whether to return the contacts as a mask, by default False.

    Returns
    -------
    NDArray[np.int_]
        The contact indices.
    """
    cell_list = struc.CellList(atom_array, cell_size=cell_size)
    contacts: NDArray[np.int_] = cell_list.get_atoms(
        coords.reshape(-1, coords.shape[-1]), radius=radius, as_mask=False
    )
    return contacts


def get_stack_contacts(
    receptor: AtomArray,
    ligand_poses: AtomArrayStack,
    threshold: float = 4.0,
    cell_size: float | None = None,
    as_mask: bool = False,
) -> NDArray[np.int_]:
    """This function concatenates all poses in the stack into a long list of atoms.
    It then creates a cell list of the atom array for efficient measurement of adjacency.

    Parameters
    ----------
    receptor : AtomArray
        The receptor atoms.
    ligand_poses : AtomArrayStack
        The ligand poses.
    threshold : float, optional
        The threshold for contact distance, by default 4.0.
    cell_size : float or None, optional
        The cell size for the cell list, by default None. If None, the threshold is used as the cell size.
    as_mask : bool, optional
        Whether to return the contacts as a mask, by default False.

    Returns
    -------
    NDArray[np.int_]
        The contact indices.
    """
    if not cell_size:
        cell_size = threshold
    contacts = get_atoms_within_coordinate(
        atom_array=receptor,
        coords=ligand_poses.coord,
        radius=threshold,
        cell_size=cell_size,
        as_mask=as_mask,
    )
    # reshape back to stack separated by poses
    return contacts.reshape(
        ligand_poses.shape[0], ligand_poses.shape[1], contacts.shape[-1]
    )


def get_atom_neighbors(
    atom_array: AtomArray,
    query_array: AtomArray,
    radius: float,
    cell_size: float | None = None,
    as_mask: bool = False,
) -> AtomArray:
    """Find atoms within a specified distance of coordinates.

    Parameters
    ----------
    atom_array : AtomArray
        The array of atoms.
    query_array : AtomArray
        The array of query atoms.
    radius : float
        The radius within which to find neighboring atoms.
    cell_size : float or None, optional
        The cell size for the cell list, by default None. If None, the radius is used as the cell size.
    as_mask : bool, optional
        Whether to return the contacts as a mask, by default False.

    Returns
    -------
    AtomArray
        The array of neighboring atoms.
    """
    if not cell_size:
        cell_size = radius
    contacts = get_atoms_within_coordinate(
        atom_array=query_array,
        coords=atom_array.coord,
        radius=radius,
        cell_size=cell_size,
        as_mask=as_mask,
    )
    neighbor_atoms = set()
    for at_idx, j in np.argwhere(contacts >= 0.0):
        neighbor_atoms.add(at_idx)

    return atom_array[list(neighbor_atoms)]


def _get_stack_contact_pairs(
    atoms: AtomArrayStack,
    chain1: list[str],
    chain2: list[str],
    radius: float,
    cell_size: float,
    atom_and_residue_level: bool,
) -> list[ContactPairs]:
    stack_contacts = []
    for i in range(atoms.shape[0]):
        prot1 = atoms[i, np.isin(atoms.chain_id, chain1)]
        prot2 = atoms[i, np.isin(atoms.chain_id, chain2)]
        contacts = get_atoms_within_coordinate(
            atom_array=prot1,
            coords=prot2.coord,
            radius=radius,
            cell_size=cell_size,
            as_mask=False,
        )
        contact_pairs = set()
        at_contact_pairs = set()
        for p2_idx, p1_idx in np.argwhere(contacts != -1):
            # chain1 residue
            p1_at = prot1[contacts[p2_idx, p1_idx]]
            p1_res = p1_at.res_id
            p1_ch = p1_at.chain_id
            # chain2 residue
            p2_at = prot2[p2_idx]
            p2_res = p2_at.res_id
            p2_ch = p2_at.chain_id
            # residue IDs in contact
            # store chains in case of multiple chains
            contact_pairs.add((p1_ch, p2_ch, p1_res, p2_res))
            if atom_and_residue_level:
                p1_at_idx = contacts[p2_idx, p1_idx]
                # Atom IDs in contact
                at_contact_pairs.add((p1_ch, p2_ch, p1_at_idx, p2_idx))
        cp = ContactPairs(
            residue_contacts=contact_pairs, atom_contacts=at_contact_pairs
        )
        stack_contacts.append(cp)
    return stack_contacts


def pairwise_contacts(
    pdb_file: Path | AtomArray | AtomArrayStack,
    chain1: str | list[str],
    chain2: str | list[str],
    radius: float = 5.0,
    cell_size: float | None = None,
    calpha_only: bool = False,
    backbone_only: bool = False,
    heavy_only: bool = True,
    atom_and_residue_level: bool = False,
) -> _AtomResContacts | _StackAtomResContacts:
    """Calculate pairwise contacts between two chains.

    Parameters
    ----------
    pdb_file : Path | AtomArray | AtomArrayStack
        The pdb file or atom array.
    chain1 : str | list[str]
        The first chain or list of chains.
    chain2 : str | list[str]
        The second chain or list of chains.
    radius : float, optional
        The radius for contact distance, by default 5.0.
    cell_size : float or None, optional
        The cell size for the cell list, by default None. If None, the radius is used as the cell size.
    calpha_only : bool, optional
        Whether to consider only alpha carbons, by default False.
    backbone_only : bool, optional
        Whether to consider only backbone atoms, by default False.
    heavy_only : bool, optional
        Whether to consider only heavy atoms, by default True.
    atom_and_residue_level : bool, optional
        Whether to return atomic and residue-level contact pairs.
        Default is False (only residue-level).

    Returns
    -------
    _AtomResContacts | _StackAtomResContacts
        The contact pairs. If atom_and_residue_level, returns a tuple of
        residue contacts and atom contacts, respectively. If the input
        structure contains multiple models, the returned value is a list of
        contact pairs in the same order that the input stack was given.

    Examples
    --------
    >>> pairwise_contacts(pdb_file='1abc.pdb', chain1='A', chain2='B') # doctest: +SKIP
    """
    if not cell_size:
        cell_size = radius

    atoms = atom_array_from_pdb_file(pdb_file)
    atoms = filter_atoms(atoms, calpha_only, backbone_only, heavy_only)
    is_stack = isinstance(atoms, AtomArrayStack)
    if not is_stack:
        atoms = stack([atoms])

    if not isinstance(chain1, list):
        chain1 = [chain1]
    if not isinstance(chain2, list):
        chain2 = [chain2]

    stack_contacts = _get_stack_contact_pairs(
        atoms, chain1, chain2, radius, cell_size, atom_and_residue_level
    )
    if is_stack:
        res_contacts = []
        atom_contacts = []
        for cp in stack_contacts:
            res_contacts.append(cp.residue_contacts)
            atom_contacts.append(cp.atom_contacts)
        if atom_and_residue_level:
            return res_contacts, atom_contacts
        else:
            return res_contacts

    pose_contacts = stack_contacts[0]
    if atom_and_residue_level:
        return pose_contacts.residue_contacts, pose_contacts.atom_contacts
    else:
        return pose_contacts.residue_contacts


def pairwise_chain_contacts(
    atom_array: AtomArray, radius: float = 4.5, cell_size: float | None = None
) -> pd.DataFrame | None:
    """Calculate pairwise contacts between chains in a protein structure.

    Parameters
    ----------
    atom_array : AtomArray
        The array of atoms representing the protein structure.
    radius : float, optional
        The radius within which to consider atoms as being in contact, by default 4.5.
    cell_size : float | None, optional
        The size of the cell used for the contact calculation. If None, it is set to the value of radius.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame containing the chain pairs and their contacts, or None if no contacts are found.

    Examples
    --------
    >>> from pinder.core.structure.atoms import atom_array_from_pdb_file
    >>> atom_array = atom_array_from_pdb_file('1abc.pdb') # doctest: +SKIP
    >>> pairwise_chain_contacts(atom_array) # doctest: +SKIP
    """
    if not cell_size:
        cell_size = radius
    chains = sorted(set(atom_array.chain_id))
    ch_combos = list(combinations(chains, 2))
    all_contacts = []
    for ch1, ch2 in ch_combos:
        prot1 = atom_array[atom_array.chain_id == ch1]
        prot2 = atom_array[atom_array.chain_id == ch2]
        contacts = get_atoms_within_coordinate(
            atom_array=prot1,
            coords=prot2.coord,
            radius=radius,
            cell_size=cell_size,
            as_mask=False,
        )
        contact_pairs = set()
        for p2_idx, p1_idx in np.argwhere(contacts != -1):
            # chain1 residue
            p1_at = prot1.res_id[contacts[p2_idx, p1_idx]]
            # chain2 residue
            p2_at = prot2.res_id[p2_idx]
            # residue IDs in contact
            contact_pairs.add((p1_at, p2_at))
        if len(contact_pairs):
            all_contacts.append(
                {
                    "chain1": ch1,
                    "chain2": ch2,
                    "contacts": contact_pairs,
                }
            )

    if all_contacts:
        all_contacts = pd.DataFrame(all_contacts)
        return all_contacts

    return None


def pairwise_clashes(
    pdb_file: Path | AtomArray,
    chain1: str | None = None,
    chain2: str | None = None,
    radius: float = 1.2,
    cell_size: float | None = None,
    calpha_only: bool = False,
    backbone_only: bool = False,
    heavy_only: bool = False,
) -> dict[str, int | float]:
    if not cell_size:
        cell_size = radius

    # Load from PDB and pass atoms to pairwise_contacts to avoid duplicate I/O
    atoms = atom_array_from_pdb_file(pdb_file)
    if not all([chain1, chain2]):
        chain1, chain2 = sorted(list(set(atoms.chain_id)))

    assert isinstance(chain1, str)
    assert isinstance(chain2, str)

    res_contacts, at_contacts = pairwise_contacts(
        atoms,
        chain1=chain1,
        chain2=chain2,
        radius=radius,
        calpha_only=calpha_only,
        backbone_only=backbone_only,
        heavy_only=heavy_only,
        atom_and_residue_level=True,
    )
    prot1 = atoms[np.isin(atoms.chain_id, chain1)]
    prot2 = atoms[np.isin(atoms.chain_id, chain2)]
    # Get residues in each chain that have atoms within 5A
    contacts_5A = get_atoms_within_coordinate(
        atom_array=prot1,
        coords=prot2.coord,
        radius=5.0,
        cell_size=5.0,
        as_mask=False,
    )
    ch1_res_ids = set()
    ch2_res_ids = set()
    for p2_idx, p1_idx in np.argwhere(contacts_5A != -1):
        # chain1 residue
        p1_at = prot1[contacts_5A[p2_idx, p1_idx]]
        p1_res = p1_at.res_id
        # chain2 residue
        p2_at = prot2[p2_idx]
        p2_res = p2_at.res_id
        ch1_res_ids.add(p1_res)
        ch2_res_ids.add(p2_res)

    # Find two closest atoms in each chain
    prot1_contact = prot1[np.isin(prot1.res_id, list(ch1_res_ids))]
    prot2_contact = prot2[np.isin(prot2.res_id, list(ch2_res_ids))]
    if prot1_contact.shape[0]:
        point_dists = cdist(prot1_contact.coord, prot2_contact.coord)
        min_dist = np.min(point_dists)
        atom_indices = np.argwhere(point_dists == min_dist)
        p1_idx, p2_idx = atom_indices[0].flatten()
        p1_at = prot1_contact[p1_idx]
        p2_at = prot2_contact[p2_idx]
        vdw_sum = atom_vdw_radius(p1_at) + atom_vdw_radius(p2_at)
    else:
        # No atoms are within 5A of each other
        min_dist = 5.0
        vdw_sum = 1.0

    clash_info: dict[str, int | float] = {
        "atom_clashes": len(at_contacts),
        "residue_clashes": len(res_contacts),
        "min_dist": min_dist,
        "min_dist_vdw_ratio": min_dist / vdw_sum,
        "vdw_sum": vdw_sum,
        "radius": radius,
    }
    return clash_info


def label_structure_gaps(atom_array: AtomArray) -> pd.DataFrame | None:
    """Find gaps in residue numbering between C-alpha atoms.

    Parameters
    ----------
    atom_array : AtomArray
        The array of atoms for which to find gaps.

    Returns
    -------
    chains : list
        A sorted list of unique chain IDs from the atom array.
    """
    chains = sorted(set(atom_array.chain_id))
    gaps = []
    for ch in chains:
        prot = atom_array[atom_array.chain_id == ch]
        prot_ca = prot[prot.atom_name == "CA"]
        prot_ca = prot_ca[prot_ca.element != "CA"]
        for i in range(prot_ca.shape[0] - 1):
            if prot_ca[i + 1].res_id != (prot_ca[i].res_id + 1):
                if prot_ca[i + 1].res_id == (prot_ca[i].res_id):
                    log.warning(
                        f"found two Calpha atoms for residue {prot_ca[i].res_id}"
                    )
                    continue
                gaps.append(
                    {
                        "chain": ch,
                        "gap_start": prot_ca[i].res_id,
                        "gap_end": prot_ca[i + 1].res_id,
                    }
                )
    if gaps:
        gaps = pd.DataFrame(gaps)
        return gaps
    return None
