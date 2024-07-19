from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from pinder.core.utils.process import process_map
from pinder.core.structure.atoms import atom_array_from_pdb_file
from pinder.core.structure.contacts import (
    pairwise_chain_contacts,
    label_structure_gaps,
    get_atom_neighbors,
)


def annotate_interface_gaps(
    pdb_file: Path, smaller_radius: float = 4.0, larger_radius: float = 8.0
) -> pd.DataFrame | None:
    """Find atomic gaps near the PPI interface.

    Look for atoms (and count residues) that are part of the interface
    and within a radius of one of the residue gaps (as defined by numbering).

    """
    atoms = atom_array_from_pdb_file(pdb_file)
    gaps = label_structure_gaps(atoms)
    if not isinstance(gaps, pd.DataFrame):
        return None

    contacts = pairwise_chain_contacts(atoms)
    annotations = []
    for i, r in contacts.iterrows():
        ch1 = r.chain1
        ch2 = r.chain2
        conts = r.contacts
        ch1_resi = set([p[0] for p in conts])
        ch2_resi = set([p[1] for p in conts])

        interface_mask = (
            (atoms.chain_id == ch1) & np.isin(atoms.res_id, list(ch1_resi))
        ) | ((atoms.chain_id == ch2) & np.isin(atoms.res_id, list(ch2_resi)))
        interface = atoms[interface_mask].copy()

        ch1_gaps = gaps.query(f'chain == "{ch1}"')
        ch2_gaps = gaps.query(f'chain == "{ch2}"')
        ch1_gaps = set(ch1_gaps.gap_start).union(set(ch1_gaps.gap_end))
        ch2_gaps = set(ch2_gaps.gap_start).union(set(ch2_gaps.gap_end))
        gap_mask = ((atoms.chain_id == ch1) & np.isin(atoms.res_id, list(ch1_gaps))) | (
            (atoms.chain_id == ch2) & np.isin(atoms.res_id, list(ch2_gaps))
        )
        gap_atoms = atoms[gap_mask].copy()

        if not gap_atoms.shape[0]:
            continue

        annotation = {
            "pdb_id": pdb_file.stem,
            "chain1": ch1,
            "chain2": ch2,
        }

        for radius in [smaller_radius, larger_radius]:
            gap_neigh = get_atom_neighbors(interface, gap_atoms, radius=radius)
            annot = pd.DataFrame(
                [
                    {"resi": at.res_id, "resn": at.res_name, "chain": at.chain_id}
                    for at in gap_neigh
                ]
            )
            missing_interface_res = 0
            rad_label = int(round(radius, 0))
            if annot.empty:
                missing_interface_res = 0
                annotation[f"interface_atom_gaps_{rad_label}A"] = 0
                annotation[f"missing_interface_residues_{rad_label}A"] = (
                    missing_interface_res
                )
            else:
                for ch, ch_gap in [(ch1, ch1_gaps), (ch2, ch2_gaps)]:
                    missing_interface_res += len(
                        set(annot.query(f'chain == "{ch}"').resi).intersection(ch_gap)
                    )
                annotation[f"interface_atom_gaps_{rad_label}A"] = annot.shape[0]
                annotation[f"missing_interface_residues_{rad_label}A"] = (
                    missing_interface_res
                )
        annotations.append(annotation)

    if annotations:
        annotation_df = pd.DataFrame(annotations)
        return annotation_df

    return None


def mp_annotate_interface_gaps(
    pdb_files: list[Path],
    parallel: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame | None:
    annotations = process_map(
        annotate_interface_gaps, pdb_files, parallel=parallel, max_workers=max_workers
    )
    annotations = [df for df in annotations if isinstance(df, pd.DataFrame)]
    if len(annotations):
        annotations_df = pd.concat(annotations).reset_index(drop=True)
        return annotations_df
    return None
