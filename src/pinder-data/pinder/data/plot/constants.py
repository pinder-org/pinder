from __future__ import annotations
from pinder.data.plot.plot import Colors


split_color_map = {
    "Train": Colors.palepurple,
    "Val": Colors.whitepurple,
    "PINDER-XL": Colors.green,
    "PINDER-S": Colors.blue,
    "PINDER-AF2": Colors.pink,
    "Invalid": Colors.midnight,
}

dataset_color_map = {
    "DIPS-equidock": Colors.blue,
    "PPIRef": Colors.green,
    "ProteinFlow": Colors.palepurple,
    # "DB5.5": Colors.blue,
    "PINDER": Colors.pink,
    "Sequence": Colors.palepurple,
}

monomer_color_map = {
    "Apo": Colors.pink,
    "Predicted (AF2)": Colors.blue,
    "Holo": Colors.green,
}

rank_color_map = {
    "Oracle": Colors.blue,
    "Top 1": Colors.brightpink,
    "Top 5": Colors.pink,
}

category_color_map = {
    # Neff
    "High": Colors.pink,
    "Medium": Colors.blue,
    "Low": Colors.green,
    "Not available": Colors.palepurple,
    # Novelty
    "Neither": Colors.pink,
    "Single": Colors.blue,
    "Both": Colors.green,
}

flexibility_color_map = {
    "Difficult": Colors.pink,
    "Flexible": Colors.pink,
    "Medium": Colors.blue,
    "Rigid-body": Colors.green,
}

LABELS = {
    # Monomer names
    "apo": "Apo",
    "predicted": "Predicted (AF2)",
    "holo": "Holo",
    # Metadata/properties
    "n_residues": "Number of residues",
    "length1": "Length (chain 1)",
    "length2": "Length (chain 2)",
    "buried_sasa": "Buried SASA",
    "probability": "PRODIGY-Cryst Probability",
    "resolution": "Resolution (Å)",
    "max_var_1": "Max variance (chain 1)",
    "max_var_2": "Max variance (chain 2)",
    "link_density": "Interfacial link density",
    "planarity": "Planarity",
    "pKd_pKi_pIC50": "Affinity (pKd/pKi/pIC50)",
    "number_of_components_1": "Number of components (chain 1)",
    "number_of_components_2": "Number of components (chain 2)",
    "num_atom_types": "Number of atom types",
    "n_residue_pairs": "Number of residue pairs",
    "intermolecular_contacts": "Intermolecular contacts",
    "charged_charged_contacts": "Charged-charged contacts",
    "charged_polar_contacts": "Charged-polar contacts",
    "charged_apolar_contacts": "Charged-apolar contacts",
    "polar_polar_contacts": "Polar-polar contacts",
    "apolar_polar_contacts": "Apolar-polar contacts",
    "apolar_apolar_contacts": "Apolar-apolar contacts",
    "interface_atom_gaps_4A": "Interface atom gaps (4Å)",
    "missing_interface_residues_8A": "Missing interface residues (8Å)",
    "chain1_neff": "N<sub>eff</sub> (chain 1)",
    "chain2_neff": "N<sub>eff</sub> (chain 2)",
    # ref + ref-free metrics
    "monomer_name_label": "Monomer",
    "iRMS": "Interface RMSD (Å)",
    "LRMS": "Ligand RMSD (Å)",
    "min_dist_vdw_ratio": "Smallest distance between receptor and ligand (divided by sum of vdW radii)",
    "min_dist": "Smallest distance between receptor and ligand atoms (Å)",
    "sel_voromqa_v1_energy_norm": "VoroMQA normalized interface energy",
    "sel_voromqa_v1_clash_score": "VoroMQA interface clash score",
    "percent_capri": "CAPRI success",
    "percent_capri_min_dist": "<br></br>CAPRI success &<br>No clashes</br>",
    "percent_capri_vmqa_clash": "<br></br>CAPRI success &<br>ClashScore pass</br>",
    "method_name": "Method",
    # Splits
    "pinder_xl": "PINDER-XL",
    "pinder_s": "PINDER-S",
    "pinder_af2": "PINDER-AF2",
    "train": "Train",
    "val": "Val",
    "invalid": "Invalid",
    # iAlign and Neff SR curves
    "log(P-value)": "iAlign log(P-value) threshold",
    "Neff": "Paired N<sub>eff</sub>",
    "sr": "Success rate (%)",
    # Families
    "panther_class": "PANTHER family pairs",
    "pfam_clan": "PFAM clan pairs",
    "ecod_pair": "ECOD family pairs",
    "percent_reps": " <b>Percentage of cluster representatives (%)</b>",
    "iRMS": "Interface RMSD (Å)",
    "LRMS": "Ligand RMSD (Å)",
}


METHOD_LABELS = {
    # Method names
    "Crystal structures (raw)": "Crystal structures<br>(no mask)</br>",
    "Crystal structures (holo sequence mask)": "Crystal structures<br>(holo sequence mask)</br>",
    "xtal": "Crystal structures",
    "af2mm_full-length": "AlphaFold-Multimer<br>(full-length)</br>",
    "af2mm_truncated": "AlphaFold-Multimer<br>(truncated)</br>",
    "af2mm": "AlphaFold-Multimer",
    "af2mm_wt": "AlphaFold-Multimer<br>(with templates)</br>",
    "diffdockpp_subset1_train2": "DiffDock-PP",
    "frodock": "FRODOCK",
    "hdock": "HDOCK",
    "patchdock": "PatchDock",
}

LABELS.update(METHOD_LABELS)


prop_plot_cols = [
    "resolution",
    "length1",
    "length2",
    "buried_sasa",
    "probability",
    "max_var_1",
    "max_var_2",
    "link_density",
    "planarity",
    "num_atom_types",
    "chain1_neff",
    "chain2_neff",
    "intermolecular_contacts",
    "charged_charged_contacts",
    "charged_polar_contacts",
    "charged_apolar_contacts",
    "polar_polar_contacts",
    "apolar_polar_contacts",
    "apolar_apolar_contacts",
    "missing_interface_residues_8A",
]


INT2WORD = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


MONOMER_LEADERBOARD_TEMPLATE = r"""
{\renewcommand{\arraystretch}{1.3}}
\begin{table*}[h!]\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lcccccccccr@{}}
\toprule
& \multicolumn{9}{c}{DockQ CAPRI classification} & \\
\cmidrule(lr){2-10}
& \multicolumn{3}{c}{Top-1} & \multicolumn{3}{c}{Top-5} & \multicolumn{3}{c}{Oracle} & \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Method & \multicolumn{1}{c}{Acceptable} & \multicolumn{1}{c}{Medium} & \multicolumn{1}{c}{High} & \multicolumn{1}{c}{Acceptable} & \multicolumn{1}{c}{Medium} & \multicolumn{1}{c}{High} & \multicolumn{1}{c}{Acceptable} & \multicolumn{1}{c}{Medium} & \multicolumn{1}{c}{High} & Miss. Sys. \\
\midrule
$ROWS
\bottomrule
\end{tabular}
}
\caption{$CAPTION}
\label{tab:$LABEL}
\end{table*}
"""


SUBSET_LEADERBOARD_TEMPLATE = r"""
{\renewcommand{\arraystretch}{1.3}}
\begin{table*}[h!]\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}clcccccccccr@{}}
\toprule
\textbf{\$DS_HEADING} & \multicolumn{11}{c}{DockQ CAPRI classification} \\
\cmidrule(lr){3-11}
& & \multicolumn{3}{c}{Top-1} & \multicolumn{3}{c}{Top-5} & \multicolumn{3}{c}{Oracle} & \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
Input & Method & \multicolumn{1}{c}{Acceptable} & \multicolumn{1}{c}{Medium} & \multicolumn{1}{c}{High} & \multicolumn{1}{c}{Acceptable} & \multicolumn{1}{c}{Medium} & \multicolumn{1}{c}{High} & \multicolumn{1}{c}{Acceptable} & \multicolumn{1}{c}{Medium} & \multicolumn{1}{c}{High} & Miss. Sys. \\
\midrule
\multirow{$N_METHODS}{*}{\shortstack{Holo\\($HOLO_COUNT)}}
$ROWS_HOLO
\midrule
\multirow{$N_METHODS}{*}{\shortstack{Apo\\($APO_COUNT)}}
$ROWS_APO
\midrule
\multirow{$N_METHODS}{*}{\shortstack{Predicted\\($PRED_COUNT)}}
$ROWS_AF2

\bottomrule
\end{tabular}
}
\caption{$CAPTION}
\label{tab:$LABEL}
\end{table*}
"""

LEADERBOARD_METRIC_COLS = [
    "DockQ % acceptable (Max(Top 1))",
    "DockQ % medium (Max(Top 1))",
    "DockQ % high (Max(Top 1))",
    "DockQ % acceptable (Max(Top 5))",
    "DockQ % medium (Max(Top 5))",
    "DockQ % high (Max(Top 5))",
    "DockQ % acceptable (oracle)",
    "DockQ % medium (oracle)",
    "DockQ % high (oracle)",
]

LEADERBOARD_SUMMARY_COLS = [
    "Method",
    "Dataset",
    "Monomer",
    "Missing systems",
] + LEADERBOARD_METRIC_COLS


LEADERBOARD_METHODS = {
    # 'diffdockpp_seq_subset1_train1',
    "diffdockpp_subset1_train2",
    "frodock",
    "hdock",
    "patchdock",
    "af2mm",
}


DOCKQ_COLUMNS = [
    "DockQ",
    "iRMS",
    "LRMS",
    "rank",
    "CAPRI",
    "CAPRI_rank",
]
CLASH_COLUMNS = [
    "method_name",
    "id",
    "monomer_name",
    "min_dist_vdw_ratio",
    "atom_clashes",
    "residue_clashes",
    "min_dist",
] + DOCKQ_COLUMNS
VOROMQA_COLUMNS = [
    "method_name",
    "id",
    "monomer_name",
    "voromqa_v1_score",
    "sel_voromqa_v1_score",
    "sel_voromqa_v1_area",
    "sel_voromqa_v1_energy",
    "sel_voromqa_v1_energy_norm",
    "sel_voromqa_v1_clash_score",
] + DOCKQ_COLUMNS


PINDER_SETS = ["PINDER-XL", "PINDER-S", "PINDER-AF2"]
