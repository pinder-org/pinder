import pandas as pd
import numpy as np
from numpy.linalg import svd
from pathlib import Path
from biotite.structure import filter_amino_acids
from pinder.core.structure.atoms import atom_array_from_pdb_file
from pinder.core.structure.contacts import pairwise_chain_contacts


def get_planarity(pdb_file: Path) -> float:
    """Calculate the planarity of the interface between two proteins.

    Parameters
    ----------
    pdb_file : Path
        The path to the pdb file.

    Returns
    -------
    float
        The root mean square error (RMSE) of the distance of the interface
        C-alpha atoms to the plane of the interface. If the interface
        C-alpha atoms are less than 3, returns -1.
    """
    # model = pdbfile.read(pdb_file.as_posix())
    # structure = model.get_structure(model=1, extra_fields=["atom_id", "b_factor"])
    structure = atom_array_from_pdb_file(pdb_file)
    structure = structure[structure.element != "H"]
    structure = structure[filter_amino_acids(structure)]

    conts = pairwise_chain_contacts(atom_array=structure)
    if isinstance(conts, pd.DataFrame) and conts.shape[0] == 1:
        conts = conts.iloc[0, :]
        chain1_resi = np.array(list(set([int(p[0]) for p in conts.contacts])))
        chain2_resi = np.array(list(set([int(p[1]) for p in conts.contacts])))

        prot1 = structure[structure.chain_id == conts.chain1]
        prot2 = structure[structure.chain_id == conts.chain2]

        prot1_int = prot1[np.isin(prot1.res_id, chain1_resi)]
        prot2_int = prot2[np.isin(prot2.res_id, chain2_resi)]

        prot1_int = prot1_int[prot1_int.atom_name == "CA"]
        prot2_int = prot2_int[prot2_int.atom_name == "CA"]

        interface_calpha_coords = np.concatenate(
            [prot1_int.coord, prot2_int.coord], axis=0
        )

        # need at least 3 points to calculate the plane
        if interface_calpha_coords.shape[0] < 3:
            return -1

        center = np.mean(interface_calpha_coords, axis=0)
        centered_coords = interface_calpha_coords - center
        _, _, Vt = svd(centered_coords)
        # The normal vector to the plane is the last row of Vt
        normal = Vt[-1, :]
        # Calculate the distance from the points to the plane
        error = np.abs(np.dot(centered_coords, normal))
        # Calculate the RMSE
        rmse: float = np.sqrt(np.mean(error * error))
        return rmse
    else:
        return -1.0
