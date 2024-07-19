from __future__ import annotations
from pathlib import Path

from pinder.core.index.system import PinderSystem


class PinderWriterBase:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def __call__(self, dimer: PinderSystem) -> None:
        return self.write(dimer)

    def write(self, dimer: PinderSystem) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class PinderDefaultWriter(PinderWriterBase):
    def __init__(self, output_path: Path) -> None:
        super().__init__(output_path=output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def write(self, dimer: PinderSystem) -> None:
        pinder_id = dimer.entry.id
        pinder_path = self.output_path / pinder_id
        pinder_path.mkdir(exist_ok=True, parents=True)
        for attr in dimer.filepaths.keys():
            structure = getattr(dimer, attr)
            if structure is not None:
                structure.to_pdb(pinder_path / structure.filepath.name)


class PinderClusteredWriter(PinderDefaultWriter):
    def __init__(self, output_path: Path) -> None:
        super().__init__(output_path=output_path)

    def write(self, dimer: PinderSystem) -> None:
        pinder_id = dimer.entry.id
        cluster_id = dimer.entry.cluster_id
        apo_path = self.output_path / cluster_id / pinder_id / "apo"
        holo_path = self.output_path / cluster_id / pinder_id / "holo"
        af2_path = self.output_path / cluster_id / pinder_id / "predicted"
        apo_path.mkdir(parents=True, exist_ok=True)
        holo_path.mkdir(parents=True, exist_ok=True)
        af2_path.mkdir(parents=True, exist_ok=True)

        subdirs = {
            "holo": holo_path,
            "apo": apo_path,
            "pred": af2_path,
        }
        for attr in dimer.filepaths.keys():
            path = subdirs[attr.split("_")[0]]
            structure = getattr(dimer, attr)
            if structure is not None:
                structure.to_pdb(path / structure.filepath.name)
