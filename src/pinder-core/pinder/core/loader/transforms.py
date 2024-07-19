from pinder.core.index.system import PinderSystem


class TransformBase:
    def __init__(self) -> None:
        pass

    def __call__(self, dimer: PinderSystem) -> PinderSystem:
        return self.transform(dimer)

    def transform(self, dimer: PinderSystem) -> PinderSystem:
        raise NotImplementedError


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


class AddNoise(TransformBase):
    pass


class AddEdges(TransformBase):
    pass


class Noise(TransformBase):
    pass


class CenterSystems(TransformBase):
    pass


class SampleContact(TransformBase):
    pass


class MarkContacts(TransformBase):
    pass


class CheckLength(TransformBase):
    pass


class CheckLengthPrody(TransformBase):
    pass


class CenterOnReceptor(TransformBase):
    pass


class RandomLigandPosition(TransformBase):
    pass


class SetTime(TransformBase):
    pass


class RandomSystemRotation(TransformBase):
    pass


class GetContacts(TransformBase):
    pass


class SampleContacts(TransformBase):
    pass
