from pinder.core.index.system import PinderSystem
from pinder.core.loader.structure import Structure
from scipy.spatial.transform import Rotation as R
import numpy as np


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


class RandomLigandTransform(TransformBase):
    def __init__(self, max_translation: float = 10.0) -> None:
        self.max_translation = max_translation

    def transform(self, ppi: PinderSystem) -> PinderSystem:
        assert ppi.entry.holo_L and ppi.entry.holo_L
        ppi = self.transform_struct(
            ppi, R.random().as_matrix(), np.random.randn(3) * self.max_translation
        )
        return ppi

    @staticmethod
    def transform_struct(pinder_system, rotation_matrix, translation_vector):
        centroid = pinder_system.atom_array.coord.mean(axis=0)
        pinder_system.atom_array.coord = pinder_system.atom_array.coord - centroid
        pinder_system.atom_array.coord = (
            rotation_matrix @ pinder_system.atom_array.coord.T
        ).T
        pinder_system.atom_array.coord = (
            pinder_system.atom_array.coord + centroid + translation_vector
        )
        for L_monomer in ["apo_ligand", "pred_ligand"]:
            L_struc = getattr(pinder_system, L_monomer)
            if L_struc:
                L_struc.atom_array.coord = L_struc.atom_array.coord - centroid
                L_struc.atom_array.coord = (
                    rotation_matrix @ L_struc.atom_array.coord.T
                ).T
                L_struc.atom_array.coord = (
                    L_struc.atom_array.coord + centroid + translation_vector
                )
        return pinder_system
