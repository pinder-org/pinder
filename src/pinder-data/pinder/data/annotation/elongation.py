from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
from numpy.typing import NDArray
from pinder.core.index.id import Dimer
from pinder.core.structure.atoms import atom_array_from_pdb_file

_OptFloat = Optional[float]
_OptInt = Optional[int]
_OptStr = Optional[str]


def get_max_var(coords: NDArray[np.double]) -> float:
    """
    Get the maximum variance of the coordinates of the CA atoms of the two
    """
    # TODO: consolidate with core.loader.filters.CheckChainElongation
    centered_coords = coords - np.mean(coords, axis=0)

    cov_matrix = np.cov(centered_coords, rowvar=False)

    _, eigenvectors = np.linalg.eigh(cov_matrix)

    V = eigenvectors[:, ::-1]
    projection = np.dot(centered_coords, V)
    variance = np.var(projection, axis=0)

    max_var: float = (variance / variance.sum()).max()
    return max_var


def calculate_elongation(
    pdb_filename: str | Path,
) -> tuple[
    str, _OptFloat, _OptFloat, _OptInt, _OptInt, _OptInt, str, str, _OptStr, _OptStr
]:
    """Get the maximum variance of the coordinates of the CA atoms of the two
    chains in the PDB file. Also get the length of the two chains and the
    number of atom types.
    """
    try:
        structure = atom_array_from_pdb_file(pdb_filename)
        chains = list(sorted(set(structure.chain_id)))
        if len(chains) != 2:
            raise ValueError("The structure does not contain exactly two chains.")
        # Dimers are supposed to be chain R = chain 1, chain L = chain 2.
        # Sorted will return L, R
        chain2_id, chain1_id = chains
        chain1 = structure[structure.chain_id == chain1_id]
        chain2 = structure[structure.chain_id == chain2_id]

        max_var_1 = get_max_var(chain1.coord)
        max_var_2 = get_max_var(chain2.coord)

        length1 = len(set(chain1.res_id))
        length2 = len(set(chain2.res_id))

        # May want to make this len(set(chain.atom_name)) ?
        num_atom_types = len(set(chain1.element) | set(chain2.element))

        dimer = Dimer.from_string(Path(pdb_filename).stem)
        chain_id1 = dimer.monomer1.proteins[0].chain
        chain_id2 = dimer.monomer2.proteins[0].chain
        return (
            Path(pdb_filename).name,
            max_var_1,
            max_var_2,
            length1,
            length2,
            num_atom_types,
            chain_id1,
            chain_id2,
            chain1_id,
            chain2_id,
        )
    except Exception as e:
        dimer = Dimer.from_string(Path(pdb_filename).stem)
        chain_id1 = dimer.monomer1.proteins[0].chain
        chain_id2 = dimer.monomer2.proteins[0].chain
        print(f"Processing failed on {pdb_filename} with error {str(e)}")
        return (
            Path(pdb_filename).name,
            None,
            None,
            None,
            None,
            None,
            chain_id1,
            chain_id2,
            None,
            None,
        )
