from pinder.core.index.system import PinderSystem
from pinder.core.loader.structure import Structure
from scipy.spatial.transform import Rotation as R
from biotite.structure import AtomArray
import numpy as np
from numpy.typing import NDArray


class TransformBase:
    def __init__(self) -> None:
        pass

    def __call__(self, dimer: PinderSystem) -> PinderSystem:
        return self.transform(dimer)

    def transform(self, dimer: PinderSystem) -> PinderSystem:
        raise NotImplementedError


class StructureTransform:
    def __call__(self, structure: Structure) -> Structure:
        return self.transform(structure)

    def transform(self, structure: Structure) -> Structure:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class SelectAtomTypes(StructureTransform):
    def __init__(self, atom_types: list[str] = ["CA"]) -> None:
        self.atom_types = atom_types

    def transform(self, structure: Structure) -> Structure:
        return structure.filter("atom_name", self.atom_types)


class SuperposeToReference(TransformBase):
    def __init__(self, reference_type: str = "holo") -> None:
        assert reference_type in {"holo", "native"}
        self.reference_type = reference_type

    def transform(self, ppi: PinderSystem) -> PinderSystem:
        assert ppi.entry.holo_L and ppi.entry.holo_L
        if self.reference_type == "native":
            ppi.holo_receptor = ppi.aligned_holo_R
            ppi.holo_ligand = ppi.aligned_holo_L

        R_ref = ppi.holo_receptor
        L_ref = ppi.holo_ligand
        for R_monomer in ["apo_receptor", "pred_receptor"]:
            R_struc = getattr(ppi, R_monomer)
            if R_struc:
                unbound_super, _, _ = R_struc.superimpose(R_ref)
                setattr(ppi, R_monomer, unbound_super)

        for L_monomer in ["apo_ligand", "pred_ligand"]:
            L_struc = getattr(ppi, L_monomer)
            if L_struc:
                unbound_super, _, _ = L_struc.superimpose(L_ref)
                setattr(ppi, L_monomer, unbound_super)
        return ppi


class RandomLigandTransform(StructureTransform):
    def __init__(self, max_translation: float = 10.0) -> None:
        self.max_translation = max_translation

    def transform(self, structure: Structure) -> Structure:
        assert {"L", "R"}.intersection(set(structure.atom_array.chain_id)) == {"L", "R"}

        ligand_atom_array = structure.atom_array[structure.atom_array.chain_id == "L"]
        receptor_atom_array = structure.atom_array[structure.atom_array.chain_id == "R"]

        ligand_atom_array = self.transform_struct(
            ligand_atom_array,
            R.random().as_matrix(),
            np.random.randn(3) * self.max_translation,
        )

        structure.atom_array = receptor_atom_array + ligand_atom_array
        return structure

    @staticmethod
    def transform_struct(
        atom_array: AtomArray,
        rotation_matrix: NDArray[np.float64],
        translation_vector: NDArray[np.float64],
    ) -> AtomArray:
        centroid = atom_array.coord.mean(axis=0)
        atom_array.coord = atom_array.coord - centroid
        atom_array.coord = (rotation_matrix @ atom_array.coord.T).T
        atom_array.coord = atom_array.coord + centroid + translation_vector
        return atom_array
