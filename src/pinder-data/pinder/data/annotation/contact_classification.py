from __future__ import annotations
from subprocess import Popen, PIPE
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from numpy.typing import NDArray

from pinder.core.utils import setup_logger, constants as pc

log = setup_logger(__name__)


def get_crystal_contact_classification(pdb_path: Path) -> list[Path | str | None]:
    """
    Get crystal contact classification using prodigy_cryst
    In addition to the contacts, the link density and label are also returned
    The label is the classification of the crystal contact as either
    "biological" or "crystal"
    """
    label: list[str] | list[None] = [None, None]
    contacts: list[str] | list[None] = [None, None, None, None, None, None, None]
    link_density: list[str] | list[None] = [None]
    classification: list[str | None | Path] = [pdb_path]

    try:
        cmd = f"prodigy_cryst {pdb_path}"
        proc = Popen(cmd, shell=True, stderr=PIPE, stdout=PIPE)
        stdout, stderr = proc.communicate()
        result = stdout.decode().strip().split("\n")
        if proc.returncode != 0:
            for ln in stderr.decode().splitlines():
                log.error(ln.strip())
        assert result != [""]
        contacts = result[1:-2]
        contacts = [i.split()[-1] for i in contacts]
        link_density = [result[-2].split()[-1]]
        label = result[-1].split()[2:4]
    except Exception as e:
        pass

    classification.extend(contacts)
    classification.extend(link_density)
    classification.extend(label)
    return classification


def detect_disulfide_bonds(
    structure: AtomArray,
    distance: float = 2.05,
    distance_tol: float = 0.05,
    dihedral: float = 90.0,
    dihedral_tol: float = 10.0,
) -> NDArray[np.int_]:
    """Detect potential disulfide bonds.

    This function is used to detects disulfide bridges in protein structures.

    The employed criteria for disulfide bonds are quite simple in this case:
    the :math:`S_\gamma` atoms of two cystein residues must be in a vicinity
    of :math:`2.05 \pm 0.05` Å and the dihedral angle of
    :math:`C_\beta - S_\gamma - S^\prime_\gamma - C^\prime_\beta` must be
    :math:`90 \pm 10 ^{\circ}`.

    Note:
        Code source: Patrick Kunzmann
        License: BSD 3 clause
        Originally implemented in https://www.biotite-python.org/examples/gallery/structure/disulfide_bonds.html#sphx-glr-examples-gallery-structure-disulfide-bonds-py

    """
    # Array where detected disulfide bonds are stored
    disulfide_bonds = []
    # All 3-letter codes for CYS
    cys_resnames = list(
        filter(
            lambda k: pc.three_to_one_noncanonical_mapping[k] == "C",
            pc.three_to_one_noncanonical_mapping,
        )
    )
    cys_resnames += ["CYS"]
    # A mask that selects only S-gamma atoms of cystein or modified cystein.
    sulfide_mask = np.isin(structure.res_name, cys_resnames) & (
        structure.atom_name == "SG"
    )
    # sulfides in adjacency to other sulfides are detected in an
    # efficient manner via a cell list
    cell_list = struc.CellList(
        structure, cell_size=distance + distance_tol, selection=sulfide_mask
    )
    # Iterate over every index corresponding to an S-gamma atom
    for sulfide_i in np.where(sulfide_mask)[0]:
        # Find indices corresponding to other S-gamma atoms,
        # that are adjacent to the position of structure[sulfide_i]
        # We use the faster 'get_atoms_in_cells()' instead of
        # `get_atoms()`, as precise distance measurement is done
        # afterwards anyway
        potential_bond_partner_indices = cell_list.get_atoms_in_cells(
            coord=structure.coord[sulfide_i]
        )
        # Iterate over every index corresponding to an S-gamma atom
        # as bond partner
        for sulfide_j in potential_bond_partner_indices:
            if sulfide_i == sulfide_j:
                # A sulfide cannot create a bond with itself:
                continue
            # Create 'Atom' instances
            # of the potentially bonds S-gamma atoms
            sg1 = structure[sulfide_i]
            sg2 = structure[sulfide_j]
            # For dihedral angle measurement the corresponding
            # C-beta atoms are required, too
            cb1 = structure[
                (structure.chain_id == sg1.chain_id)
                & (structure.res_id == sg1.res_id)
                & (structure.atom_name == "CB")
            ]
            cb2 = structure[
                (structure.chain_id == sg2.chain_id)
                & (structure.res_id == sg2.res_id)
                & (structure.atom_name == "CB")
            ]
            # Measure distance and dihedral angle and check criteria
            bond_dist = struc.distance(sg1, sg2)
            bond_dihed = np.abs(np.rad2deg(struc.dihedral(cb1, sg1, sg2, cb2)))
            if (
                bond_dist > distance - distance_tol
                and bond_dist < distance + distance_tol
                and (bond_dihed > dihedral - dihedral_tol).any()
                and (bond_dihed < dihedral + dihedral_tol).any()
            ):
                # Atom meet criteria -> we found a disulfide bond
                # -> the indices of the bond S-gamma atoms
                # are put into a tuple with the lower index first
                bond_tuple = sorted((sulfide_i, sulfide_j))
                # Add bond to list of bonds, but each bond only once
                if bond_tuple not in disulfide_bonds:
                    disulfide_bonds.append(bond_tuple)
    return np.array(disulfide_bonds, dtype=int)
