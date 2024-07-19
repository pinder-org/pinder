"""Select single representative apo monomers for a dimer entry.

Uses suite of apo-holo difficulty assessment metrics to create a scaled score
and select a single receptor and single ligand monomer for a given pinder_id
when apo structures are available.
"""

from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pinder.core import PinderSystem
from pinder.core.index.utils import get_index, IndexEntry
from pinder.core.utils import setup_logger
from pinder.core.structure.atoms import get_seq_identity


pindex = get_index()
log = setup_logger(__name__)


def get_apo_monomer_weighted_score(
    apo_data: pd.DataFrame, scale_type: str = "standard"
) -> pd.DataFrame:
    if scale_type == "standard":
        scaler = StandardScaler()
    elif scale_type == "minmax":
        scaler = MinMaxScaler()

    metric_weights = {
        "I-RMSD": 0.2,
        "L-RMSD": 0.2,
        "apo_seq_identity": 0.2,
        "Fnat": 0.2,
        "Fnonnat": 0.2,
    }
    scaled = pd.DataFrame(
        scaler.fit_transform(apo_data[metric_weights.keys()]),
        columns=list(metric_weights.keys()),
    )

    # Invert scaling on i-RMSD. Higher = worse
    # The score will be higher = better
    scaled.loc[:, "L-RMSD"] = scaled["L-RMSD"] * -1
    scaled.loc[:, "I-RMSD"] = scaled["I-RMSD"] * -1
    scaled.loc[:, "Fnonnat"] = scaled["Fnonnat"] * -1
    apo_data["apo_score"] = scaled.dot(
        pd.DataFrame(
            metric_weights.values(),
            index=metric_weights.keys(),
        )
    )
    return apo_data


def get_system_monomer_difficulty(pinder_id: str) -> pd.DataFrame | None:
    row = pindex.query(f'id == "{pinder_id}"').squeeze()
    entry = IndexEntry(**row.to_dict())
    apo_rec_codes = [p.split("__")[0] for p in entry.apo_R_pdbs.split(";") if p != ""]
    apo_lig_codes = [p.split("__")[0] for p in entry.apo_L_pdbs.split(";") if p != ""]
    if not (len(apo_rec_codes) or len(apo_lig_codes)):
        log.error(f"No apo monomers exist for {pinder_id}!")
        return None

    metrics = []
    for rec_code in apo_rec_codes:
        try:
            ps = PinderSystem(
                pinder_id,
                apo_receptor_pdb_code=rec_code,
            )
            difficulty = ps.apo_monomer_difficulty("apo", "receptor")
            difficulty["id"] = pinder_id
            difficulty["apo_code"] = rec_code
            apo_R = ps.apo_receptor or ps.holo_receptor
            holo_R = ps.aligned_holo_R
            r_seq = get_seq_identity(holo_R.sequence, apo_R.sequence)
            difficulty["apo_seq_identity"] = r_seq
            metrics.append(difficulty)
        except Exception as e:
            log.error(
                f"Failed to calculate difficulty for {pinder_id} {rec_code}: {str(e)}"
            )

    for lig_code in apo_lig_codes:
        try:
            ps = PinderSystem(
                pinder_id,
                apo_ligand_pdb_code=lig_code,
            )
            difficulty = ps.apo_monomer_difficulty("apo", "ligand")
            difficulty["id"] = pinder_id
            difficulty["apo_code"] = lig_code
            apo_L = ps.apo_ligand or ps.holo_ligand
            holo_L = ps.aligned_holo_L
            l_seq = get_seq_identity(holo_L.sequence, apo_L.sequence)
            difficulty["apo_seq_identity"] = l_seq
            metrics.append(difficulty)
        except Exception as e:
            log.error(
                f"Failed to calculate difficulty for {pinder_id} {lig_code}: {str(e)}"
            )
    if metrics:
        diff_df = pd.DataFrame(metrics)
        scored = (
            diff_df.groupby("unbound_body", as_index=False)
            .apply(lambda x: get_apo_monomer_weighted_score(x.reset_index(drop=True)))
            .reset_index(drop=True)
            .sort_values("apo_score", ascending=False)
            .reset_index(drop=True)
        )
        return scored
    return None


def get_canonical_apo_codes(pinder_id: str) -> dict[str, str] | None:
    diff_df = get_system_monomer_difficulty(pinder_id)
    if not isinstance(diff_df, pd.DataFrame):
        return None
    top_df = diff_df.drop_duplicates(["id", "unbound_body"], keep="first").reset_index(
        drop=True
    )
    canonical_selections = {
        rec["unbound_body"]: rec["apo_code"] for rec in top_df.to_dict(orient="records")
    }
    return canonical_selections
