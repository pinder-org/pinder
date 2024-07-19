from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from pinder.core.utils.log import setup_logger
from pinder.core.utils.paths import empty_file
from pinder.data.pipeline import constants


log = setup_logger(__name__)


def skip_step(
    step_name: str, run_specific_step: str = "", skip_specific_step: str = ""
) -> bool:
    skip = (
        run_specific_step != "" and run_specific_step != step_name
    ) or skip_specific_step == step_name
    if skip:
        log.info(
            f"Skipping {step_name}... "
            f"(run_specific_step={run_specific_step}, skip_specific_step={skip_specific_step})"
        )
    return skip


def get_uningested_mmcif(ingest_cifs: list[Path]) -> list[Path]:
    # Check cached results
    uningested = []
    for mmcif_file in tqdm(ingest_cifs):
        pdb_id = mmcif_file.stem.split("_")[1][-4:].lower()
        metadata_file = mmcif_file.parent / f"{pdb_id}-metadata.tsv"
        interacting_chains_tsv = mmcif_file.parent / f"{pdb_id}-interacting_chains.tsv"
        if empty_file(metadata_file):
            uningested.append(mmcif_file)
            continue
        try:
            cached_metadata = pd.read_csv(metadata_file, sep="\t")
        except Exception as e:
            log.error(f"Failed to read existing metadata file. Removing...")
            metadata_file.unlink()
            cached_metadata = None

        if cached_metadata is not None and "status" in cached_metadata.columns:
            meta_status = cached_metadata.status.values[0]
            if interacting_chains_tsv.is_file():
                try:
                    interacting_chains = pd.read_csv(interacting_chains_tsv, sep="\t")
                    is_dimer = interacting_chains.shape[0] > 0
                except Exception as e:
                    is_dimer = False
            else:
                is_dimer = False
            if meta_status in constants.PROCESSED_STATUS_CODES and is_dimer:
                # Fix for bug where monomers had almost random uniprot assigned
                continue
        uningested.append(mmcif_file)
    return uningested


def get_pisa_unannotated(
    ingest_cifs: list[Path], use_checkpoint: bool = True
) -> list[Path]:
    unannotated = []
    for mmcif_file in tqdm(ingest_cifs):
        path_to_pdb = mmcif_file.parent
        pdb_id = path_to_pdb.stem[-4:]
        checkpoint_file = path_to_pdb / "checkpoint-pisa.txt"
        assembly_file = path_to_pdb / f"{pdb_id}-pisa-lite-assembly.json"
        interface_file = path_to_pdb / f"{pdb_id}-pisa-lite-interfaces.json"
        if use_checkpoint and checkpoint_file.is_file():
            continue

        if assembly_file.is_file() or interface_file.is_file():
            continue
        unannotated.append(mmcif_file)
    return unannotated


def complete_rcsb_annotation(pdb_id: str, annotation_fp: Path) -> bool:
    pfam_fp = annotation_fp / "pfam"
    feat_fp = annotation_fp / "features"
    annot_fp = annotation_fp / "annotations"
    req_files = [fp / f"{pdb_id}.pkl" for fp in [pfam_fp, feat_fp, annot_fp]]
    return all([f.is_file() for f in req_files])


def get_rcsb_unannotated(pdb_ids: list[str], pinder_dir: Path) -> list[str]:
    annotation_fp = pinder_dir / "rcsb_annotations"
    remaining_ids = [
        pdb_id
        for pdb_id in pdb_ids
        if not complete_rcsb_annotation(pdb_id, annotation_fp)
    ]
    return remaining_ids


def get_unannotated_dimer_pdbs(
    dimer_pdbs: list[Path],
    use_cache: bool = True,
) -> list[Path]:
    # Check cached results
    unannotated = []
    for pdb_file in tqdm(dimer_pdbs):
        output_tsv = pdb_file.parent / f"{pdb_file.stem}.tsv"
        if empty_file(output_tsv):
            unannotated.append(pdb_file)
        elif use_cache:
            continue
        else:
            output_tsv.unlink()
            unannotated.append(pdb_file)
    return unannotated


def get_dimer_pdbs_missing_foldseek_contacts(
    dimer_pdbs: list[Path],
    config_hash: str,
    use_cache: bool = True,
) -> list[Path]:
    # Check cached results for foldseek contacts
    unannotated = []
    for pdb_file in tqdm(dimer_pdbs):
        contact_fp = pdb_file.parent / "foldseek_contacts"
        config_dir = contact_fp / config_hash
        contact_json = config_dir / f"{pdb_file.stem}.json"
        if not empty_file(contact_json) and use_cache:
            # Probably not a good assumption to check only the file size.
            # In theory it could be corrupt.
            continue
        if contact_json.is_file():
            contact_json.unlink()
        unannotated.append(pdb_file)
    return unannotated
