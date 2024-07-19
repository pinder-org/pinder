from __future__ import annotations
import logging
import re
from pathlib import Path
import biotite.structure as struc
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pinder.core.structure.atoms import get_seq_alignments, get_seq_identity
from pinder.core.structure.superimpose import superimpose_chain
from pinder.core.loader.structure import reverse_dict, Structure
from pinder.core.utils import setup_logger, unbound
from pinder.data.config import ApoPairingConfig


UNIPROT_UNDEFINED = "UNDEFINED"


log = setup_logger(__name__, log_level=logging.WARNING)


def sufficient_atom_types(struct: Structure, min_atom_types: int = 3) -> bool:
    """Checks if the given structure contains at least a minimum number of unique atom types.

    Args:
        struct (Structure): The structure to be evaluated.
        min_atom_types (int, optional): The minimum number of unique atom types required. Default is 3.

    Returns:
        bool: True if the structure contains at least the specified number of unique atom types, including 'CA'.
    """
    atom_tys = set(struct.atom_array.atom_name)
    # Require at least 3 atom types to consider as apo monomer
    sufficient = (len(atom_tys) >= min_atom_types) and ("CA" in atom_tys)
    return sufficient


def sufficient_residues(struct: Structure, min_residues: int = 5) -> bool:
    """Determines if a structure contains at least a specified minimum number of residues.

    Args:
        struct (Structure): The structure to be evaluated.
        min_residues (int, optional): The minimum number of residues required. Default is 5.

    Returns:
        bool: True if the structure has at least the specified number of residues.
    """
    # Require at least 5 residues to consider as apo monomer
    n_res = len(struct.residues)
    return n_res >= min_residues


def valid_structure(pdb_file: Path) -> Structure | None:
    """Attempts to create a Structure instance from a PDB file.

    Args:
        pdb_file (Path): The path to the PDB file.

    Returns:
        Structure | None: The loaded Structure object if successful, None if an error occurs.
    """
    try:
        apo_struct = Structure(pdb_file)
    except Exception as e:
        log.warning(f"Unable to load {pdb_file} structure")
        apo_struct = None
    return apo_struct


def validate_apo_monomer(
    apo_id: str,
    pdb_dir: Path,
    config: ApoPairingConfig = ApoPairingConfig(),
) -> dict[str, str | bool]:
    """Validates an apo monomer by checking atom types and residue count against configuration thresholds.

    Args:
        apo_id (str): The identifier for the apo monomer.
        pdb_dir (Path): The directory containing PDB files.
        config (ApoPairingConfig, optional): Configuration settings with thresholds for validation.

    Returns:
        dict[str, str | bool]: A dictionary containing the monomer ID and its validation status.
    """
    pdb_file = pdb_dir / f"{apo_id}.pdb"
    struct = valid_structure(pdb_file)
    apo_info: dict[str, str | bool] = {"id": apo_id}
    valid = False
    if isinstance(struct, Structure):
        if sufficient_atom_types(struct, config.min_atom_types) and sufficient_residues(
            struct, config.min_residues
        ):
            valid = True
    apo_info["valid_as_apo"] = valid
    return apo_info


def holo_apo_seq_identity(holo_seq: str, apo_seq: str) -> dict[str, str | float]:
    """Computes the sequence identity between a holo sequence and an apo sequence.

    Args:
        holo_seq (str): The sequence of the holo structure.
        apo_seq (str): The sequence of the apo structure.

    Returns:
        dict[str, str | float]: A dictionary containing the holo sequence, apo sequence, and their sequence identity.
    """
    seq_id = get_seq_identity(ref_seq=holo_seq, subject_seq=apo_seq)
    identity_info: dict[str, str | float] = {
        "holo_sequence": holo_seq,
        "apo_sequence": apo_seq,
        "sequence_identity": seq_id,
    }
    return identity_info


def chain_instance_from_chain(ch: str) -> int:
    """Extracts the instance/copy number from a chain identifier.

    Args:
        ch (str): The chain identifier, which may contain digits representing the instance number.

    Returns:
        int: The extracted instance number, defaulting to 1 if no digits are found.
    """
    re_match = re.findall(r"\d+", ch)
    if re_match:
        instance = int(re_match[0])
    else:
        instance = 1
    return instance


def remove_apo_chain_copies(monomer_df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate chain entries from a DataFrame based on chain instance numbers.

    Args:
        monomer_df (pd.DataFrame): The DataFrame containing monomer data with a 'chain' column.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only the first instance of each chain.
    """
    monomer_df.loc[:, "chain_instance"] = monomer_df["chain"].apply(
        chain_instance_from_chain
    )
    monomer_df = monomer_df.query("chain_instance == 1").reset_index(drop=True)
    return monomer_df


def remove_dimer_chain_copies(dimer_df: pd.DataFrame) -> pd.DataFrame:
    """Filters out dimer chains that are duplicates based on their instance numbers.

    Args:
        dimer_df (pd.DataFrame): A DataFrame containing data for dimers with columns for 'chain_R' and 'chain_L'.

    Returns:
        pd.DataFrame: The DataFrame filtered to exclude entries where both chains are copies.
    """
    dimer_df.loc[:, "chain_instance_1"] = dimer_df.chain_R.apply(
        chain_instance_from_chain
    )
    dimer_df.loc[:, "chain_instance_2"] = dimer_df.chain_L.apply(
        chain_instance_from_chain
    )
    # Copy if and only if both interacting chains are asymmetric unit assembly copies
    dimer_df.loc[:, "chain_instance_copy"] = [
        min([inst1, inst2]) > 1
        for inst1, inst2 in zip(
            dimer_df["chain_instance_1"], dimer_df["chain_instance_2"]
        )
    ]
    dimer_df = dimer_df.query("chain_instance_copy == False").reset_index(drop=True)
    return dimer_df


def hybrid_align(
    apo_monomer: Structure,
    holo_monomer: Structure,
    align_method: str = "pymol",
) -> tuple[Structure, dict[str, int | float]]:
    """Performs structural alignment between an apo monomer and a holo monomer using specified alignment methods.

    The function supports alignment using either PyMOL or Biotite libraries, depending on the 'align_method' specified.
    The alignment results include the aligned structure and metrics such as RMSD and the number of aligned atoms.

    Args:
        apo_monomer (Structure): The apo monomer structure to align.
        holo_monomer (Structure): The holo monomer structure as the reference.
        align_method (str): The alignment method to use; defaults to "pymol". Options include "pymol" and "biotite".

    Returns:
        tuple[Structure, dict[str, int | float]]: A tuple containing the aligned apo monomer structure and
                                                  a dictionary with alignment metrics.
    """

    def _pymol_align(
        apo_monomer: Structure,
        holo_monomer: Structure,
    ) -> tuple[tuple[float, float, int, int, int], NDArray[np.double]]:
        from pymol import cmd

        cmd.reinitialize()
        cmd.load(apo_monomer.filepath, "apo_mono")
        cmd.load(holo_monomer.filepath, "holo_ref")
        aln_info = cmd.align("apo_mono", "holo_ref")
        super_coord = cmd.get_coords("apo_mono")
        # RMSD after refinement
        refine_rmsd = aln_info[0]
        # Number of aligned atoms after refinement
        refine_ats = aln_info[1]
        # RMSD before refinement
        raw_rmsd = aln_info[3]
        # Number of aligned atoms before refinement
        raw_ats = aln_info[4]
        # Number of residues aligned
        aln_res = aln_info[-1]
        aln: tuple[float, float, int, int, int] = (
            raw_rmsd,
            refine_rmsd,
            raw_ats,
            refine_ats,
            aln_res,
        )
        return aln, super_coord

    def _biotite_align(
        apo_monomer: Structure, holo_monomer: Structure
    ) -> tuple[tuple[float, float, int, int, int], NDArray[np.double]]:
        # max_iterations=1 -> no outlier removal
        apo_superimposed, _, holo_anchors, apo_anchors = superimpose_chain(
            holo_monomer.atom_array, apo_monomer.atom_array, max_iterations=1
        )
        raw_rmsd = struc.rmsd(
            holo_monomer.atom_array.coord[holo_anchors],
            apo_superimposed.coord[apo_anchors],
        )
        raw_atom_number = _atom_number_in_anchor_residues(
            apo_monomer.atom_array, apo_anchors
        )
        # Aligned C-alpha residues prior to refinement
        aligned_residues = len(apo_anchors)

        # Now we refine the alignment by removing outliers
        apo_superimposed, _, holo_anchors, apo_anchors = superimpose_chain(
            holo_monomer.atom_array, apo_monomer.atom_array
        )
        refine_rmsd = struc.rmsd(
            holo_monomer.atom_array.coord[holo_anchors],
            apo_superimposed.coord[apo_anchors],
        )
        refine_atom_number = _atom_number_in_anchor_residues(
            apo_monomer.atom_array, apo_anchors
        )

        metrics_tuple = (
            raw_rmsd,
            refine_rmsd,
            raw_atom_number,
            refine_atom_number,
            aligned_residues,
        )
        return metrics_tuple, apo_superimposed.coord

    apo_mono_at = apo_monomer.atom_array.copy()
    aln_func = _pymol_align if align_method == "pymol" else _biotite_align
    if align_method == "pymol":
        try:
            from pymol import cmd
        except ImportError as e:
            log.warning(
                f"Requested {align_method} but pymol not installed. Falling back to biotite method..."
            )
            aln_func = _biotite_align
    aln, super_coord = aln_func(apo_monomer, holo_monomer)
    (raw_rmsd, refine_rmsd, raw_ats, refine_ats, aln_res) = aln
    metrics: dict[str, int | float] = {
        "raw_rmsd": raw_rmsd,
        "refine_rmsd": refine_rmsd,
        "raw_aln_ats": raw_ats,
        "refine_aln_ats": refine_ats,
        "aln_res": aln_res,
    }
    apo_mono_at.coord = super_coord
    apo_monomer.atom_array = apo_mono_at.copy()
    return apo_monomer, metrics


def get_superimposed_metrics(
    holo_ref: Structure,
    apo_mono: Structure,
    body: str,
    unbound_id: str,
    holo_R: Structure,
    holo_L: Structure,
    rec_res: list[int],
    lig_res: list[int],
    bound_contacts: set[tuple[str, str, int, int]],
    holo2apo_seq: dict[str, dict[int, int]],
    apo2holo_seq: dict[str, dict[int, int]],
    config: ApoPairingConfig = ApoPairingConfig(),
) -> dict[str, str | float | int]:
    """Calculates and returns various metrics after superimposing the apo monomer onto the holo reference.

    This function assesses interface contacts, sequence identity, and structural alignment quality between
    an apo and a holo structure, providing metrics that aid in evaluating the apo-holo pairing suitability.

    Args:
        holo_ref (Structure): The holo reference structure used for alignment.
        apo_mono (Structure): The apo structure to align and analyze.
        body (str): Indicates whether the structure represents the 'receptor' or 'ligand' side.
        unbound_id (str): A unique identifier for the pairing, typically combining IDs of involved structures.
        holo_R (Structure): The holo structure of the receptor side.
        holo_L (Structure): The holo structure of the ligand side.
        rec_res (list[int]): List of receptor residues involved in holo interface contacts.
        lig_res (list[int]): List of ligand residues involved in holo interface contacts.
        bound_contacts (set[tuple[str, str, int, int]]): Set of tuples detailing contacts in the bound state.
        holo2apo_seq (dict[str, dict[int, int]]): Mapping of holo to apo sequences by residue numbers.
        apo2holo_seq (dict[str, dict[int, int]]): Mapping of apo to holo sequences by residue numbers.
        config (ApoPairingConfig): Configuration object with parameters like contact radius and alignment method.

    Returns:
        dict[str, str | float | int]: Dictionary of calculated metrics including interface residues, RMSD,
                                      sequence identity, and alignment scores.
    """
    fnat_metrics: dict[str, str | float | int]
    FAILED_METRICS: dict[str, str | float | int] = {
        "Fnat": -1.0,
        "Fnonnat": -1.0,
        "common_contacts": -1,
        "differing_contacts": -1,
        "bound_contacts": len(bound_contacts),
        "unbound_contacts": -1,
        "fnonnat_R": -1.0,
        "fnonnat_L": -1.0,
        "fnat_R": -1.0,
        "fnat_L": -1.0,
        "difficulty": "not determined",
        "I-RMSD": -1.0,
        "matched_interface_chains": -1,
        "refine_rmsd": -1.0,
        "raw_rmsd": -1.0,
        "refine_aln_ats": 0,
        "raw_aln_ats": 0,
        "aln_res": 0,
        "unbound_id": unbound_id,
        "holo_receptor_interface_res": len(rec_res),
        "holo_ligand_interface_res": len(lig_res),
        "apo_receptor_interface_res": -1,
        "apo_ligand_interface_res": -1,
    }
    try:
        R_chain = holo_R.chains[0]
        L_chain = holo_L.chains[0]
        holo_complex = holo_R + holo_L
        holo_at = holo_complex.atom_array.copy()
        # Align apo_mono to holo_ref
        apo_mono, aln_metrics = hybrid_align(
            apo_mono, holo_ref, align_method=config.align_method
        )
        if body == "receptor":
            apo_RL = apo_mono + holo_L
        else:
            apo_RL = holo_R + apo_mono

        apo_complex = apo_RL.atom_array

        # Get contacts in unbound superimposed structure
        unbound_contacts = apo_RL.get_contacts(
            radius=config.contact_rad,
            backbone_only=config.backbone_only,
            heavy_only=config.heavy_only,
        )
        apo_interface_res = apo_RL.get_interface_residues(unbound_contacts)
        interface_count = {
            "holo_receptor_interface_res": len(rec_res),
            "holo_ligand_interface_res": len(lig_res),
            "apo_receptor_interface_res": len(apo_interface_res[R_chain]),
            "apo_ligand_interface_res": len(apo_interface_res[L_chain]),
        }
        renumbered_contacts = set()
        for cp in unbound_contacts:
            c1, c2, r1, r2 = cp
            r1_renum = apo2holo_seq.get(c1, {r1: r1}).get(r1, r1 * -100)
            r2_renum = apo2holo_seq.get(c2, {r2: r2}).get(r2, r2 * -100)
            renumbered_contacts.add((c1, c2, r1_renum, r2_renum))

        # Remove holo interface residues not in mapping and get corresponding apo
        mapped_rec_res, apo_rec_res = unbound.get_corresponding_residues(
            rec_res, R_chain, holo2apo_seq
        )
        mapped_lig_res, apo_lig_res = unbound.get_corresponding_residues(
            lig_res, L_chain, holo2apo_seq
        )

        interface_res = {R_chain: mapped_rec_res, L_chain: mapped_lig_res}
        holo_interface_mask = holo_complex.get_interface_mask(interface_res)

        # get atom mask for Calpha atoms of residues in the unbound structure
        # corresponding to their equivalents in the bound structure
        apo_interface_res = {R_chain: apo_rec_res, L_chain: apo_lig_res}
        apo_interface_mask = apo_RL.get_interface_mask(apo_interface_res)

        holo_interface = holo_at[holo_interface_mask].copy()
        apo_interface = apo_complex[apo_interface_mask].copy()
        holo_interface, apo_interface = unbound.filter_interface_intersect(
            holo_interface,
            apo_interface,
            holo2apo_seq,
            apo2holo_seq,
            R_chain,
            L_chain,
        )
        fitted, transformation = struc.superimpose(
            holo_interface,
            apo_interface,
        )
        # We should probably perform this check and say that iRMSD is >> 2.5
        # if there is only one chain in common in the interface.
        # It would indicate an invalid apo-holo mapping / structure
        if fitted.shape[0]:
            n_apo_ch = len(struc.get_chains(fitted))
        else:
            n_apo_ch = 0
        # assert n_apo_ch > 1

        # Get RMSD of the interface-Calpha atoms after superposition
        irmsd = struc.rmsd(fitted, holo_interface)

        # Calculate fnat + fnonnat
        # https://pubmed.ncbi.nlm.nih.gov/12784368/
        fnat_metrics = unbound.fnat_unbound_summary(bound_contacts, renumbered_contacts)
        fnonnat = fnat_metrics["Fnonnat"]
        difficulty = unbound.get_db4_difficulty(irmsd, float(fnonnat))
        fnat_metrics["difficulty"] = difficulty
        fnat_metrics["I-RMSD"] = irmsd
        fnat_metrics["matched_interface_chains"] = n_apo_ch
        fnat_metrics.update(aln_metrics)
        fnat_metrics["unbound_id"] = unbound_id
        fnat_metrics.update(interface_count)
    except Exception as e:
        log.error(f"Failed to superimpose or get unbound metrics for {unbound_id}: {e}")
        fnat_metrics = FAILED_METRICS
    return fnat_metrics


def get_sequence_based_metrics(
    apo_monomer_id: str,
    body: str,
    apo_complex: Structure,
    apo_R: Structure,
    apo_L: Structure,
    R_chain: str,
    L_chain: str,
    rec_res: list[int],
    lig_res: list[int],
    holo2apo_seq: dict[str, dict[int, int]],
) -> dict[str, str | float | int]:
    """Gathers sequence-based metrics for an apo monomer pairing based on sequence alignment and structural data.
    Metrics calculated here do not require any structural superposition.

    Args:
        apo_monomer_id (str): Identifier for the apo monomer.
        body (str): Designates whether the monomer is treated as 'receptor' or 'ligand'.
        apo_complex (Structure): Combined structure of apo monomer and holo counterpart body.
        apo_R (Structure): Structure of the apo monomer acting as the receptor.
        apo_L (Structure): Structure of the apo monomer acting as the ligand.
        R_chain (str): Chain identifier for the receptor.
        L_chain (str): Chain identifier for the ligand.
        rec_res (list[int]): List of holo receptor interface residues.
        lig_res (list[int]): List of holo ligand interface residues.
        holo2apo_seq (dict[str, dict[int, int]]): Mapping from holo to apo residues.

    Returns:
        dict[str, str | float | int]: Metrics related to sequence alignment and interface composition.
    """
    apoL_res, apoL_atoms = len(apo_L.residues), apo_L.atom_array.shape[0]
    apoR_res, apoR_atoms = len(apo_R.residues), apo_R.atom_array.shape[0]

    # Seq align apo_mono to holo_ref
    apoR_missing = [r for r in rec_res if not holo2apo_seq[R_chain].get(r)]
    apoL_missing = [r for r in lig_res if not holo2apo_seq[L_chain].get(r)]
    R_miss = len(apoR_missing)
    L_miss = len(apoL_missing)
    apo_R_code = apo_monomer_id if body == "receptor" else "holo"
    apo_L_code = apo_monomer_id if body == "ligand" else "holo"
    R_nat = len(rec_res)
    L_nat = len(lig_res)
    Fmiss_R = R_miss / R_nat if R_nat > 0 else -1.0
    Fmiss_L = L_miss / L_nat if L_nat > 0 else -1.0

    sequence_metrics: dict[str, str | float | int] = {
        "receptor_residues": apoR_res,
        "ligand_residues": apoL_res,
        "receptor_atoms": apoR_atoms,
        "ligand_atoms": apoL_atoms,
        "complex_residues": len(apo_complex.residues),
        "complex_atoms": apo_complex.atom_array.shape[0],
        "receptor_missing": R_miss,
        "receptor_native": R_nat,
        "ligand_missing": L_miss,
        "ligand_native": L_nat,
        "Fmiss_R": Fmiss_R,
        "Fmiss_L": Fmiss_L,
        "apo_R_code": apo_R_code,
        "apo_L_code": apo_L_code,
    }
    return sequence_metrics


def get_unbound_id(
    holo_R: Structure,
    holo_L: Structure,
    apo_R: Structure,
    apo_L: Structure,
    body: str,
) -> str:
    if body == "receptor":
        R = apo_R.pinder_id
        L = holo_L.pinder_id
    else:
        R = holo_R.pinder_id
        L = apo_L.pinder_id
    unbound_id: str = R + "--" + L
    return unbound_id


def get_apo_pairing_metrics_for_id(
    df: pd.DataFrame,
    pdb_dir: Path,
    config: ApoPairingConfig = ApoPairingConfig(),
) -> pd.DataFrame | None:
    """Computes various structural and sequence-based metrics for apo-holo pairings for a given dataset of identifiers.

    This function loads structures, performs alignments, and computes interface and sequence identity metrics
    for a set of potential apo-holo pairs specified in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing identifiers and other data for apo-holo pairings.
        pdb_dir (Path): Path to the directory containing PDB files of the structures.
        config (ApoPairingConfig): Configuration object specifying parameters for alignment and analysis.

    Returns:
        pd.DataFrame | None: DataFrame containing computed metrics for each pairing, or None if an error occurs.
    """
    df.reset_index(drop=True, inplace=True)
    holo_R_pdb = df.holo_R_pdb.values[0]
    holo_L_pdb = df.holo_L_pdb.values[0]
    try:
        holo_R = Structure(pdb_dir / f"{holo_R_pdb}")
        holo_L = Structure(pdb_dir / f"{holo_L_pdb}")
    except Exception as e:
        print(f"ERROR: failed to load holo monomers {holo_R_pdb} {holo_L_pdb}: {e}")
        return None

    holo_complex = holo_R + holo_L
    holoR_res, holoR_atoms = len(holo_R.residues), holo_R.atom_array.shape[0]
    holoL_res, holoL_atoms = len(holo_L.residues), holo_L.atom_array.shape[0]
    holoR_seq, holoL_seq = holo_R.sequence, holo_L.sequence
    R_chain = holo_R.chains[0]
    L_chain = holo_L.chains[0]
    bound_contacts = holo_complex.get_contacts(
        radius=config.contact_rad,
        backbone_only=config.backbone_only,
        heavy_only=config.heavy_only,
    )
    interface_res = holo_complex.get_interface_residues(bound_contacts)
    rec_res = interface_res[R_chain]
    lig_res = interface_res[L_chain]
    apo_metrics = []
    for _, r in df.iterrows():
        try:
            apo_eval_struct = Structure(pdb_dir / f"{r.apo_monomer_id}.pdb")
        except Exception as e:
            continue

        metrics: dict[str, str | float | int | bool] = {
            "id": r.id,
            "apo_monomer_id": r.apo_monomer_id,
            "holo_R_residues": holoR_res,
            "holo_R_atoms": holoR_atoms,
            "holo_L_residues": holoL_res,
            "holo_L_atoms": holoL_atoms,
            "radius": config.contact_rad,
            "backbone_only": config.backbone_only,
            "heavy_only": config.heavy_only,
            "unbound_body": r.body,
            "monomer_name": "apo",
        }
        if r.body == "R":
            body = "receptor"
            apo_eval_chain = R_chain
            apo_ref_struct = holo_L
            holo_counterpart_res = holoR_res
            holo_counterpart_seq = holoR_seq

        elif r.body == "L":
            body = "ligand"
            apo_eval_chain = L_chain
            apo_ref_struct = holo_R
            holo_counterpart_res = holoL_res
            holo_counterpart_seq = holoL_seq

        # require apo to have at least min_holo_resolved_frac fraction of holo residues resolved
        # might want to also set a max on fraction of "extra" residues
        apo_res_frac = len(apo_eval_struct.residues) / holo_counterpart_res
        if apo_res_frac < config.min_holo_resolved_frac:
            continue

        identity_info = holo_apo_seq_identity(
            holo_seq=holo_counterpart_seq, apo_seq=apo_eval_struct.sequence
        )
        if identity_info["sequence_identity"] < config.min_seq_identity:
            continue

        apo_eval_struct.atom_array.chain_id[
            apo_eval_struct.atom_array.chain_id == config.apo_chain
        ] = apo_eval_chain

        if body == "receptor":
            apo_R = apo_eval_struct
            apo_L = apo_ref_struct
        else:
            apo_R = apo_ref_struct
            apo_L = apo_eval_struct

        apo_complex = apo_R + apo_L
        # Seq align apo_mono to holo_ref
        apo2holo_seq = apo_complex.get_per_chain_seq_alignments(holo_complex)
        holo2apo_seq = {ch: reverse_dict(rmap) for ch, rmap in apo2holo_seq.items()}

        sequence_metrics = get_sequence_based_metrics(
            apo_monomer_id=r.apo_monomer_id,
            body=body,
            apo_complex=apo_complex,
            apo_R=apo_R,
            apo_L=apo_L,
            R_chain=R_chain,
            L_chain=L_chain,
            rec_res=rec_res,
            lig_res=lig_res,
            holo2apo_seq=holo2apo_seq,
        )
        metrics.update(sequence_metrics)

        # Calculate metrics after swapping the apo monomer for the holo monomer after superposition.
        fnat_metrics = get_superimposed_metrics(
            holo_ref=holo_R if body == "receptor" else holo_L,
            apo_mono=apo_R if body == "receptor" else apo_L,
            body=body,
            unbound_id=get_unbound_id(holo_R, holo_L, apo_R, apo_L, body),
            holo_R=holo_R,
            holo_L=holo_L,
            rec_res=rec_res,
            lig_res=lig_res,
            bound_contacts=bound_contacts,
            holo2apo_seq=holo2apo_seq,
            apo2holo_seq=apo2holo_seq,
            config=config,
        )
        metrics.update(fnat_metrics)
        metrics.update(identity_info)
        apo_metrics.append(metrics)

    if len(apo_metrics):
        metric_df = pd.DataFrame(apo_metrics)
    else:
        metric_df = None
    return metric_df


def calculate_frac_monomer_dimer_overlap(
    df: pd.DataFrame,
    pdb_dir: Path,
    config: ApoPairingConfig = ApoPairingConfig(),
) -> pd.DataFrame:
    """Calculates the fractional overlap of residues between apo monomers and their corresponding holo forms in the dimer.
    This method attempts to capture metrics for cases where a single apo monomer contains all or most of the full dimer complex,
    thereby making it impossible to predicted holo starting from apo.

    Args:
        df (pd.DataFrame): DataFrame containing data for which overlap metrics need to be calculated.
        pdb_dir (Path): Directory where PDB files are located.
        config (ApoPairingConfig): Configuration parameters used in the calculation, such as sequence alignment settings.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated overlap metrics.
    """
    dimer_id = df.id.values[0]
    R, L = dimer_id.split("--")
    rename_cols = ["asym_id", "resi", "resn", "one_letter_code_uniprot"]
    R_map = pd.read_parquet(pdb_dir.parent / f"mappings/{R}-R.parquet")
    L_map = pd.read_parquet(pdb_dir.parent / f"mappings/{L}-L.parquet")
    dimer_map = pd.concat([R_map, L_map], ignore_index=True)
    dimer_map = dimer_map[
        ["resi", "asym_id", "resn", "resi_uniprot", "one_letter_code_uniprot"]
    ].copy()
    dimer_map = dimer_map[~dimer_map.resi_uniprot.isna()].reset_index(drop=True)
    dimer_map.resi_uniprot = dimer_map.resi_uniprot.astype(int)

    dimer_pdb = pdb_dir / f"{dimer_id}.pdb"
    dimer = Structure(dimer_pdb)
    dimer_R = dimer.filter("chain_id", "R")
    dimer_L = dimer.filter("chain_id", "L")
    dimer_R.atom_array.chain_id[dimer_R.atom_array.chain_id == "R"] = config.apo_chain
    dimer_L.atom_array.chain_id[dimer_L.atom_array.chain_id == "L"] = config.apo_chain
    dimer_A = dimer_R + dimer_L
    dimer_seq = dimer_A.sequence
    dimer_seq_len = len(dimer_seq)
    metrics = []
    for apo_id, body in zip(df["apo_monomer_id"], df["unbound_body"]):
        apo_map_pqt = pdb_dir.parent / f"mappings/{apo_id}.parquet"
        if apo_map_pqt.is_file():
            apo_map = pd.read_parquet(apo_map_pqt)
            apo_map = apo_map[["resi_uniprot"] + rename_cols]
            apo_map.rename({c: f"apo_{c}" for c in rename_cols}, axis=1, inplace=True)
            apo_map = apo_map[~apo_map.resi_uniprot.isna()].reset_index(drop=True)
            apo_map.resi_uniprot = apo_map.resi_uniprot.astype(int)
            apo_dimer_coverage = (
                pd.merge(
                    apo_map, dimer_map[["resi_uniprot"] + rename_cols], how="inner"
                )
                .drop_duplicates("resi_uniprot")
                .reset_index(drop=True)
            )
            frac_monomer_dimer_map = apo_dimer_coverage.shape[0] / dimer_map.shape[0]
        else:
            log.warning(f"{apo_map_pqt} mapping is missing!")
            frac_monomer_dimer_map = -1.0
        apo_struct = Structure(pdb_dir / f"{apo_id}.pdb")
        alignments = get_seq_alignments(dimer_seq, apo_struct.sequence)
        aln = alignments[0]
        gap_seq1, gap_seq2 = aln.get_gapped_sequences()
        apo_aln_len = len([s for s in gap_seq1 if s != "-"])
        # dimer_aln_len = len([s for s in gap_seq2 if s != '-'])
        frac_monomer_dimer = apo_aln_len / dimer_seq_len
        metrics.append(
            {
                "id": dimer_id,
                "apo_monomer_id": apo_id,
                "unbound_body": body,
                "frac_monomer_dimer": frac_monomer_dimer,
                "frac_monomer_dimer_uniprot": frac_monomer_dimer_map,
            }
        )
    metric_df = pd.DataFrame(metrics)
    return metric_df


def _atom_number_in_anchor_residues(
    atoms: struc.AtomArray, anchor_indices: NDArray[np.int_]
) -> int:
    """Get the total number of atoms in the residues indicated by `anchor_indices`.
    `anchor_indices` themselves point to CA atoms inside the residues.
    """
    return int(np.count_nonzero(struc.get_residue_masks(atoms, anchor_indices)))
