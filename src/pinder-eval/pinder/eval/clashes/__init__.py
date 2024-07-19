from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
from biotite.structure.atoms import AtomArray

from pinder.core import PinderSystem
from pinder.core.structure.contacts import pairwise_clashes
from pinder.core.utils import setup_logger


log = setup_logger(__name__)


def count_pinder_clashes(pinder_id: str, radius: float = 1.2) -> pd.DataFrame:
    def _monomer_clash(
        array: AtomArray,
        radius: float,
        pinder_id: str,
        monomer_name: str,
        holo_mask: bool,
    ) -> dict[str, float | int | str | bool] | None:
        try:
            clash: dict[str, float | int | str | bool] = pairwise_clashes(
                array, radius=radius
            )
            clash["id"] = pinder_id
            clash["monomer_name"] = monomer_name
            clash["holo_mask"] = holo_mask
            return clash
        except Exception as e:
            log.error(f"Failed to process {pinder_id}: {str(e)}")
            return None

    try:
        ps = PinderSystem(pinder_id)
        complex_variants = []
        for monomer_name in ["holo", "apo", "predicted"]:
            if monomer_name == "holo":
                binary = ps.native
            else:
                if monomer_name == "apo":
                    R = ps.apo_receptor or ps.holo_receptor
                    L = ps.apo_ligand or ps.holo_ligand
                elif monomer_name == "predicted":
                    R = ps.pred_receptor or ps.holo_receptor
                    L = ps.pred_ligand or ps.holo_ligand

                holo_R = ps.aligned_holo_R
                holo_L = ps.aligned_holo_L
                R_super, _, _ = R.superimpose(holo_R)
                L_super, _, _ = L.superimpose(holo_L)
                binary = R_super + L_super

            complex_variants.append((binary.atom_array, monomer_name, False))
        # Create holo sequence-masked apo and predicted complexes
        apo_complex = ps.create_apo_complex()
        pred_complex = ps.create_pred_complex()
        complex_variants.append((apo_complex.atom_array, "apo", True))
        complex_variants.append((pred_complex.atom_array, "predicted", True))

        pinder_clash = []
        for atoms, monomer_name, holo_mask in complex_variants:
            clash = _monomer_clash(
                atoms, radius, pinder_id, monomer_name, holo_mask=holo_mask
            )
            if clash:
                pinder_clash.append(clash)

        return pd.DataFrame(pinder_clash)
    except Exception as e:
        log.error(f"Failed to calculate clashes for pinder {pinder_id}: {str(e)}")
        pinder_clash = [{"id": pinder_id, "exception": str(e)}]
        return pd.DataFrame(pinder_clash)


def count_clashes(
    pdb_file: Path, radius: float = 1.2
) -> dict[str, str | int | float] | dict[str, str]:
    if os.stat(str(pdb_file)).st_size == 0:
        clash = {"pdb_file": str(pdb_file), "exception": "PDB is empty"}
        return clash
    try:
        clash = pairwise_clashes(pdb_file, radius=radius)
        clash["pdb_file"] = str(pdb_file)
    except Exception as e:
        log.error(f"Failed to process {pdb_file}: {str(e)}")
        clash = {"pdb_file": str(pdb_file), "exception": str(e)}
    return clash
