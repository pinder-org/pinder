from __future__ import annotations
from pathlib import Path
from pinder.core import get_index, get_systems
from pinder.core.utils import setup_logger
from pinder.methods import SUPPORTED_PAIRS

log = setup_logger(__name__)
pindex = get_index()


def prepare_inference_inputs(
    method_dir: Path,
    subset: str | None = None,
    ids: list[str] | None = None,
    pairs: str = "all",
) -> list[dict[str, str | Path]]:
    mono_keys = (
        [pair.value for pair in SUPPORTED_PAIRS]
        if pairs == SUPPORTED_PAIRS.ALL
        else [pairs]
    )
    if subset:
        pinder_ids = set(pindex.query(f'{subset.replace("-", "_")}').id)
    elif ids:
        pinder_ids = set(ids)
    else:
        pinder_ids = set(pindex.query('split == "test"').id)

    inference_config = []
    for system in get_systems(pinder_ids):
        filepaths = system.filepaths
        system_dir = method_dir / system.entry.id
        for monomer in mono_keys:
            has_R = getattr(system.entry, f"{monomer}_R", None)
            has_L = getattr(system.entry, f"{monomer}_L", None)
            if not all((has_R, has_L)):
                log.debug(f"{system.entry.id} missing {monomer} monomers!")
                continue
            prefix = "pred" if monomer == "predicted" else monomer
            R = getattr(system, f"{prefix}_receptor", None)
            L = getattr(system, f"{prefix}_ligand", None)
            assert R is not None and L is not None

            input_dir = system_dir / f"inputs/{monomer}"
            input_dir.mkdir(exist_ok=True, parents=True)
            R_pdb = input_dir / f"{system.entry.pdb_id}_{monomer}_R.pdb"
            L_pdb = input_dir / f"{system.entry.pdb_id}_{monomer}_L.pdb"
            R.to_pdb(R_pdb)
            L.to_pdb(L_pdb)
            complex_config: dict[str, str | Path] = {
                "id": system.entry.id,
                "kind": monomer,
                "receptor": R_pdb,
                "ligand": L_pdb,
            }
            inference_config.append(complex_config)

    log.info(
        f"Created inference input directory at {method_dir} containing {len(pinder_ids)} systems"
    )
    return inference_config
