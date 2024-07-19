"""Assess difficulty of unbound monomer or dimer with respect to bound holo."""

from __future__ import annotations

import biotite.structure as struc
import numpy as np

from biotite.structure.atoms import AtomArray
from pinder.core.loader.structure import reverse_dict, Structure
from pinder.core.utils import setup_logger
from pinder.core.structure import surgery

log = setup_logger(__name__)


def _safe_divide(num: int, denom: int, default_val: float = 0.0) -> float:
    if denom:
        return num / denom
    else:
        log.warning("Contacts in denominator are zero!")
        return default_val


def get_db4_difficulty(irmsd: float, fnonnat: float) -> str:
    if irmsd <= 1.5 and fnonnat <= 0.4:
        difficulty = "Rigid-body"
    elif (1.5 < irmsd <= 2.2) or (irmsd <= 1.5 and fnonnat > 0.4):
        difficulty = "Medium"
    else:
        difficulty = "Difficult"
    return difficulty


def fnat_unbound_summary(
    bound_contacts: set[tuple[str, str, int, int]],
    unbound_contacts: set[tuple[str, str, int, int]],
) -> dict[str, float | int | str]:
    n_nat = len(bound_contacts.intersection(unbound_contacts))
    L_bound_conts = {(c2, r2) for c1, c2, r1, r2 in bound_contacts}
    R_bound_conts = {(c1, r1) for c1, c2, r1, r2 in bound_contacts}
    L_unbound_conts = {(c2, r2) for c1, c2, r1, r2 in unbound_contacts}
    R_unbound_conts = {(c1, r1) for c1, c2, r1, r2 in unbound_contacts}
    n_nat_R = len(R_bound_conts.intersection(R_unbound_conts))
    n_nat_L = len(L_bound_conts.intersection(L_unbound_conts))
    n_nonnat = len(unbound_contacts - bound_contacts)
    n_tot_nat = len(bound_contacts)
    n_tot_unbound = len(unbound_contacts)
    n_nonnat_R = len(R_unbound_conts - R_bound_conts)
    n_nonnat_L = len(L_unbound_conts - L_bound_conts)
    n_tot_nat = len(bound_contacts)
    fnat = n_nat / n_tot_nat
    fnat_R = _safe_divide(n_nat_R, len(R_bound_conts), 1.0)
    fnat_L = _safe_divide(n_nat_L, len(L_bound_conts), 1.0)
    fnonnat = _safe_divide(n_nonnat, n_tot_unbound, 0.0)
    fnonnat_R = _safe_divide(n_nonnat_R, len(R_unbound_conts), 0.0)
    fnonnat_L = _safe_divide(n_nonnat_L, len(L_unbound_conts), 0.0)

    return {
        "Fnat": fnat,
        "Fnonnat": fnonnat,
        "common_contacts": n_nat,
        "differing_contacts": n_nonnat,
        "bound_contacts": n_tot_nat,
        "unbound_contacts": n_tot_unbound,
        "fnonnat_R": fnonnat_R,
        "fnonnat_L": fnonnat_L,
        "fnat_R": fnat_R,
        "fnat_L": fnat_L,
    }


def get_corresponding_residues(
    res_list: list[int], chain: str, mapping: dict[str, dict[int, int]]
) -> tuple[list[int], list[int]]:
    """Get corresponding residues with default mapping."""
    corresponding = []
    original = []
    for res in res_list:
        other_res = mapping.get(chain, {res: res}).get(res)
        # Make sure res has a mapping to other structure
        if other_res:
            corresponding.append(other_res)
            original.append(res)
    return original, corresponding


def filter_interface_intersect(
    holo_interface: AtomArray,
    apo_interface: AtomArray,
    holo2apo_seq: dict[str, dict[int, int]],
    apo2holo_seq: dict[str, dict[int, int]],
    R_chain: str = "R",
    L_chain: str = "L",
) -> tuple[AtomArray, AtomArray]:
    if holo_interface.shape[0] != apo_interface.shape[0]:
        holo_interface = surgery.remove_duplicate_calpha(holo_interface)
        apo_interface = surgery.remove_duplicate_calpha(apo_interface)

    # There are cases where the residues exist, but no CA atoms
    # or atoms named CA1 with non-standard res
    apo_R_interface = apo_interface[apo_interface.chain_id == R_chain]
    apo_L_interface = apo_interface[apo_interface.chain_id == L_chain]
    holo_R_interface = holo_interface[holo_interface.chain_id == R_chain]
    holo_L_interface = holo_interface[holo_interface.chain_id == L_chain]

    apo_rec_ids = list(apo_R_interface.res_id)
    apo_lig_ids = list(apo_L_interface.res_id)
    rec_res = [
        at.res_id
        for at in holo_R_interface
        if holo2apo_seq.get(R_chain, {at.res_id: at.res_id})[at.res_id] in apo_rec_ids
    ]
    lig_res = [
        at.res_id
        for at in holo_L_interface
        if holo2apo_seq.get(L_chain, {at.res_id: at.res_id})[at.res_id] in apo_lig_ids
    ]
    apo_rec_res = [
        at.res_id
        for at in apo_R_interface
        if apo2holo_seq.get(R_chain, {at.res_id: at.res_id})[at.res_id] in rec_res
    ]
    apo_lig_res = [
        at.res_id
        for at in apo_L_interface
        if apo2holo_seq.get(L_chain, {at.res_id: at.res_id})[at.res_id] in lig_res
    ]
    holo_R_interface = holo_R_interface[np.isin(holo_R_interface.res_id, rec_res)]
    holo_L_interface = holo_L_interface[np.isin(holo_L_interface.res_id, lig_res)]
    apo_R_interface = apo_R_interface[np.isin(apo_R_interface.res_id, apo_rec_res)]
    apo_L_interface = apo_L_interface[np.isin(apo_L_interface.res_id, apo_lig_res)]
    apo_interface = apo_R_interface + apo_L_interface
    holo_interface = holo_R_interface + holo_L_interface
    return holo_interface, apo_interface


def get_unbound_interface_metrics(
    holo_complex: Structure,
    apo_RL: Structure,
    R_chain: str,
    L_chain: str,
    holo2apo_seq: dict[str, dict[int, int]],
    apo2holo_seq: dict[str, dict[int, int]],
) -> dict[str, float | int | str]:
    holo_at = holo_complex.atom_array.copy()
    apo_complex = apo_RL.atom_array
    # Contacts defined as any atom within 5A of each other
    # https://pubmed.ncbi.nlm.nih.gov/12784368/
    bound_contacts = holo_complex.get_contacts()
    interface_res = holo_complex.get_interface_residues(bound_contacts)
    rec_res = interface_res[R_chain]
    lig_res = interface_res[L_chain]

    # Get contacts in unbound superimposed structure
    unbound_contacts = apo_RL.get_contacts()
    apo_interface_res = apo_RL.get_interface_residues(unbound_contacts)
    renumbered_contacts = set()
    for cp in unbound_contacts:
        c1, c2, r1, r2 = cp
        r1_renum = apo2holo_seq.get(c1, {r1: r1}).get(r1, r1 * -100)
        r2_renum = apo2holo_seq.get(c2, {r2: r2}).get(r2, r2 * -100)
        renumbered_contacts.add((c1, c2, r1_renum, r2_renum))

    # Remove holo interface residues not in mapping and get corresponding apo
    rec_res, apo_rec_res = get_corresponding_residues(rec_res, R_chain, holo2apo_seq)
    lig_res, apo_lig_res = get_corresponding_residues(lig_res, L_chain, holo2apo_seq)

    interface_res = {R_chain: rec_res, L_chain: lig_res}
    holo_interface_mask = holo_complex.get_interface_mask(interface_res)

    # get atom mask for Calpha atoms of residues in the unbound structure
    # corresponding to their equivalents in the bound structure
    apo_interface_res = {R_chain: apo_rec_res, L_chain: apo_lig_res}
    apo_interface_mask = apo_RL.get_interface_mask(apo_interface_res)

    holo_interface = holo_at[holo_interface_mask].copy()
    apo_interface = apo_complex[apo_interface_mask].copy()
    holo_interface, apo_interface = filter_interface_intersect(
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
    fnat_metrics = fnat_unbound_summary(bound_contacts, renumbered_contacts)
    fnonnat = fnat_metrics["Fnonnat"]
    difficulty = get_db4_difficulty(irmsd, float(fnonnat))
    fnat_metrics["difficulty"] = difficulty
    fnat_metrics["I-RMSD"] = irmsd
    fnat_metrics["matched_interface_chains"] = n_apo_ch
    return fnat_metrics


def get_unbound_difficulty(
    holo_R: Structure,
    holo_L: Structure,
    apo_R: Structure,
    apo_L: Structure,
) -> dict[str, float | int | str]:
    holo_complex = holo_R + holo_L
    R_chain = holo_R.chains[0]
    L_chain = holo_L.chains[0]
    apo_complex = apo_R + apo_L
    apo2holo_seq = apo_complex.get_per_chain_seq_alignments(holo_complex)
    holo2apo_seq = {ch: reverse_dict(rmap) for ch, rmap in apo2holo_seq.items()}

    # Even if atom counts are identical, annotation categories must be the same
    apo_R_super, Lrmsd_R = apo_R.superimpose(holo_R)
    apo_L_super, Lrmsd_L = apo_L.superimpose(holo_L)
    apo_RL = apo_R_super + apo_L_super

    fnat_metrics = get_unbound_interface_metrics(
        holo_complex, apo_RL, R_chain, L_chain, holo2apo_seq, apo2holo_seq
    )
    fnat_metrics["L-RMSD"] = Lrmsd_L
    fnat_metrics["R-RMSD"] = Lrmsd_R
    fnat_metrics["unbound_id"] = apo_RL.pinder_id
    fnat_metrics["unbound_body"] = "receptor_ligand"
    return fnat_metrics


def get_apo_monomer_difficulty(
    holo_R: Structure, holo_L: Structure, apo_mono: Structure, apo_body: str
) -> dict[str, float | int | str]:
    holo_complex = holo_R + holo_L
    R_chain = holo_R.chains[0]
    L_chain = holo_L.chains[0]
    if apo_body == "receptor":
        holo_ref = holo_R
    else:
        holo_ref = holo_L

    # Seq align apo_mono to holo_ref
    apo2holo_seq = apo_mono.get_per_chain_seq_alignments(holo_ref)
    holo2apo_seq = {ch: reverse_dict(rmap) for ch, rmap in apo2holo_seq.items()}

    # Align apo_mono to holo_ref
    apo_mono_super, Lrmsd = apo_mono.superimpose(holo_ref)
    holo_ref_at = holo_ref.atom_array.copy()
    apo_mono_at = apo_mono_super.atom_array.copy()
    # Even if atom counts are identical, annotation categories must be the same
    holo_ref_at, apo_mono_at = surgery.fix_annotation_mismatch(holo_ref_at, apo_mono_at)
    apo_mono_super.atom_array = apo_mono_at
    if apo_body == "receptor":
        apo_RL = apo_mono_super + holo_L
    else:
        apo_RL = holo_R + apo_mono_super

    fnat_metrics = get_unbound_interface_metrics(
        holo_complex, apo_RL, R_chain, L_chain, holo2apo_seq, apo2holo_seq
    )
    fnat_metrics["L-RMSD"] = Lrmsd
    fnat_metrics["unbound_id"] = apo_RL.pinder_id
    fnat_metrics["unbound_body"] = apo_body
    return fnat_metrics
