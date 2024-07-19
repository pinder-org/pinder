from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pinder.core.structure.models import BackboneDefinition


@dataclass
class PinderDataGenConfig:
    """A class to represent configuration parameters used to generate dataset.

    Attributes
    ----------
    interacting_chains_backbone_only : bool
        Whether to define contacts between interacting chains based on backbone atoms.
    interacting_chains_radius : float
        The radius to use when detecting contacts between putative interacting chains.
    connected_component_radius : float
        The radius to use when calculating connected components.
    max_assembly_chains : int
        The maximum number of chains allowed in the bio-assembly to consider
        for ingestion.

    """

    interacting_chains_backbone_only: bool = True
    interacting_chains_radius: float = 10.0
    connected_component_radius: float = 15.0
    max_assembly_chains: int = 500


@dataclass
class ContactConfig:
    """A class to represent configuration parameters used to generate foldseek contacts.

    Attributes
    ----------
    heavy_only : bool
        Whether to limit contact search to heavy atoms.
    backbone_only : bool
        Whether to limit contact search to only backbone atoms for ingestion.
    backbone_definition: BackboneDefinition
        Which atoms names define backbone atoms.
        dockq: (CA, N, C, O) vs. biotite: (CA, N, C)
    radius : float
        The radius to use for detecting contacts between interacting chains.
    only_unique_resi : bool
        Whether to only return unique residue IDs making contacts.
    min_length : int
        Minimum interface length per chain.
        Note: its not currently used when extracting the contacts.

    """

    heavy_only: bool = True
    backbone_only: bool = True
    backbone_definition: BackboneDefinition = "dockq"
    radius: float = 10.0
    only_unique_resi: bool = True
    min_length: int = 3


@dataclass
class TransientInterfaceConfig:
    """A class to represent configuration parameters used to annotate potentially transient interfaces.

    Attributes
    ----------
    radius : float
        Radius used to detect inter-chain bonds like di-sulfide bonds that may be inducing/stabilizing the interface.
        Default is 2.3 Å.
    min_buried_sasa : float
        The minimum buried surface area to not be considered a potentially transient interface.
        Default is 1000.0 Å^2.
    disulfide_bond_distance : float
        Bond distance used to detect potential disulfide bridges.
    disulfide_bond_distance_tol : float
        Tolerance to pad bond distance threshold by when calculating distances.
    disulfide_bond_dihedral : float
        Bond dihedral angle used to detect potential disulfide bridges.
    disulfide_bond_dihedral_tol : float
        Tolerance to pad bond dihedral angle threshold by when calculating dihedrals.

    """

    radius: float = 2.3
    min_buried_sasa: float = 1000.0
    disulfide_bond_distance: float = 2.05
    disulfide_bond_distance_tol: float = 0.05
    disulfide_bond_dihedral: float = 90.0
    disulfide_bond_dihedral_tol: float = 10.0


@dataclass
class FoldseekConfig:
    """A class to represent configuration parameters used in foldseek search.

    Attributes
    ----------
    sensitivity: float
        Adjust sensitivity to speed trade-off; lower is faster, higher more sensitive
        (1.0 faster; 4.0 fast; 7.5 sensitive; default 9.5; pinder default 11.0)
    evalue: float
        List matches below this E-value (range 0.0-inf, default: 0.001); increasing it
        reports more distant structures. Pinder default is 0.05.
    score_type: str
        Alignment metric to use as primary score. Must be one of `lddt`, `alntmscore`.
        Default is `lddt`.
    max_seqs: int
        Maximum results per query sequence allowed to pass the prefilter (affects sensitivity).
        Default is 1000.
    alignment_type: int
        Which alignment type to use in generating alignments.
        Main options are:
        1. TMalign which is actually an optimized version of TM, Foldseek-TM
            a. this option is global and slow
            b. `--alignment-type 1`
        2. 3Di+AA Gotoh-Smith-Waterman, which is the default
            a. this option is local and fast
            b. `--alignment-type 2`
    alignment_filename: str
        Alignment output filename. Defaults to alignment.txt.

    """

    sensitivity: float = 11.0
    evalue: float = 0.05
    score_type: str = "lddt"
    max_seqs: int = 1000
    alignment_type: int = 2
    alignment_filename: str = "alignment.txt"

    def __post_init__(self) -> None:
        assert int(self.alignment_type) in {
            1,
            2,
        }, f"alignment_type must be 1 or 2, got {self.alignment_type}"
        assert self.score_type in {
            "lddt",
            "alntmscore",
        }, f"score_type must be alntmscore or lddt, got {self.score_type}"


@dataclass
class MMSeqsConfig:
    """A class to represent configuration parameters used in MMSeqs2 search.

    Attributes
    ----------
    sensitivity: float
        Adjust sensitivity to speed trade-off; lower is faster, higher more sensitive
        Sensitivity: 1.0 faster; 4.0 fast; 7.5 sensitive [5.700 default in mmseqs, 11.0 default in pinder.]
    evalue: float
        List matches below this E-value (range 0.0-inf, default: 0.001); increasing it
        reports more distant structures. Pinder default is 0.05.
    score_type: str
        Alignment metric to use as primary MMSeqs2 score. Currently only `pident` is allowed.
    min_seq_id: float
        List matches above this sequence identity (for clustering) (range 0.0-1.0).
        Default is 0.2.
    max_seqs: int
        Maximum results per query sequence allowed to pass the prefilter (affects sensitivity).
        Default is 1000.
    alignment_filename: str
        Alignment output filename. Defaults to alignment.txt.

    """

    sensitivity: float = 11.0
    evalue: float = 0.05
    score_type: str = "pident"
    min_seq_id: float = 0.2
    max_seqs: int = 1000
    alignment_filename: str = "alignment.txt"

    def __post_init__(self) -> None:
        assert self.score_type in {
            "pident",
        }, f"score_type must be pident, got {self.score_type}"


@dataclass
class GraphConfig:
    """A class to represent configuration parameters used in constructing graphs from alignments.

    Attributes
    ----------
    min_interface_length: int
        Minimum length of interface for clustering.
        Default is 7.
    min_alignment_length: int
        Minimum length of alignment for clustering
        Default is 10
    score_threshold: float
        Score threshold for clustering
        Default is 0.5
    upper_threshold: float
        Upper score threshold for clustering.
        Default is 1.1
    mmseqs_score_threshold: float
        MMSeqs2 score threshold for clustering
        Default is 30.
    mmseqs_upper_threshold: float
        Upper score threshold for MMSeqs2 clustering.
        Default is 110.
    coverage_threshold: float
        Coverage threshold for clustering
        Default is 0.5

    """

    min_interface_length: int = 7
    min_alignment_length: int = 10
    score_threshold: float = 0.5
    upper_threshold: float = 1.1
    mmseqs_score_threshold: float = 30.0
    mmseqs_upper_threshold: float = 110.0
    coverage_threshold: float = 0.5


@dataclass
class ScatterConfig:
    """A class to represent batching parameters used to scatter data pipeline tasks.

    Attributes
    ----------
    two_char_batch_size: int
        Target number of two_char_codes per task batch.
    mmcif_batch_size: int
        Target number of raw mmcif files to ingest per task batch.
    graphql_batch_size: int
        Target number of PDB IDs per graphql task batch.
    dimer_batch_size: int
        Target number of dimer PDB files to annotate per task batch.
    predicted_batch_size: int
        Target number of pdb entries per predicted monomer population task.
    foldseek_db_size: int
        Target number of PDB file per sub-database to run all-vs-all foldseek on.
    apo_pairing_id_batch_size: int
        Target number of holo-apo-R/L pairing IDs per apo eval task batch.

    """

    two_char_batch_size: int = 2
    mmcif_batch_size: int = 250
    graphql_batch_size: int = 50_000
    dimer_batch_size: int = 5000
    predicted_batch_size: int = 20_000
    foldseek_db_size: int = 50_000
    apo_pairing_id_batch_size: int = 20_000


@dataclass
class ClusterConfig:
    """Configuration parameters for clustering pinder dimers and generating splits.

    Attributes
    ----------
    seed: int
        Random seed to use for AsynLPA clustering.
    canonical_method: str
        Name of the "primary" clustering method.
        Default is `foldseek_community`.
    edge_weight: str | None
        The edge attribute for nx.Graph inputs representing the weight of an edge.
        If None, uses 1 for all weights. Used for AsynLPA clustering.
        Defaults to "weight".
    foldseek_cluster_edge_threshold: float
        The edge weight threshold to use when clustering the foldseek graph.
        All edges below this threshold are removed from the graph.
        Defaults to 0.7.
    foldseek_edge_threshold: float
        The edge weight threshold to use when searching for neighboring nodes in the foldseek graph.
        Defaults to 0.55.
    foldseek_af2_difficulty_threshold: float
        The edge weight threshold to use when searching for neighboring nodes in the foldseek graph
        when establishing an alternative 'difficulty' level for the af2mm holdout set using a less strict
        threshold than the default threshold used for transitive hits deleaking.
        Defaults to 0.70.
    mmseqs_edge_threshold: float
        The edge weight threshold to use when searching for neighboring nodes in the mmseqs graph.
        Defaults to 0.0 (all alignment hits).
    resolution_thr: float
        Test set criteria: The maximum resolution threshold. Defaults to 3.5.
    min_chain_length: int
        Test set criteria: The minimum chain length threshold. Defaults to 40.
    min_atom_types: int
        Test set criteria: The minimum nubmer of atom types (currently tracked as number of elements). Defaults to 3.
    max_var_thr: float
        Test set criteria: The maximum variance threshold. Defaults to 0.98.
    oligomeric_count: int
        Test set criteria: oligomer count in the original RCSB entry. Defaults to 2 for dimers.
    method: str
        Test set criteria: experimental method used to generate structure. Defaults to `X-RAY DIFFRACTION`
    interface_atom_gaps_4A: int
        Test set criteria: maximum number of atom gaps within 4A of the interface residues. Defaults to 0.
    prodigy_label: str
        Test set criteria: the interaction type label as reported by prodigy_cryst. Defaults to `BIO` for biological interactions.
    number_of_components: int
        Test set criteria: maximum number of components in a chain (checks for detached components). Defaults to 0.
    alphafold_cutoff_date: str
        Test set criteria: The AF2 training cutoff date to use when constructing a holdout set for evaluating AF2-MM.
        Defaults to 2021-10-01.
    depth_limit: int
        Deleaking: maximum depth to hop between node neighborhoods when performing depth-first search on the graph for transitive hits.
        Default is 2.
    max_node_degree: int
        Deleaking: The maximum node degree at which we assume there is leakage when performing search for transitive hits.
        Defaults to 1_000.
    top_n: int
        Splitting: The maximum number of representatives per cluster ID.
        Defaults to 1.
    min_depth_2_hits_with_comm: int
        Splitting: The minimum number of depth_2 (or depth_limit) hits with community clustering.
        Defaults to 1.
    max_depth_2_hits_with_comm: int
        Splitting: The maximum number of depth_2 (or depth_limit) hits with community clustering.
        Defaults to 2_000.
    max_depth_2_hits: int
        Splitting: The maximum number of depth_2 (or depth_limit) hits.
        Defaults to 1_000.

    """

    seed: int = 40
    canonical_method: str = "foldseek_community"
    edge_weight: str | None = "weight"
    foldseek_cluster_edge_threshold: float = 0.70
    foldseek_edge_threshold: float = 0.55
    foldseek_af2_difficulty_threshold: float = 0.70
    mmseqs_edge_threshold: float = 0.0
    resolution_thr: float = 3.5
    min_chain_length: int = 40
    min_atom_types: int = 3
    max_var_thr: float = 0.98
    oligomeric_count: int = 2
    method: str = "X-RAY DIFFRACTION"
    interface_atom_gaps_4A: int = 0
    prodigy_label: str = "BIO"
    number_of_components: int = 1
    alphafold_cutoff_date: str = "2021-10-01"
    depth_limit: int = 2
    max_node_degree: int = 1_000
    top_n: int = 1
    min_depth_2_hits_with_comm: int = 1
    max_depth_2_hits_with_comm: int = 2_000
    max_depth_2_hits: int = 1_000


@dataclass
class ApoPairingConfig:
    """Configuration parameters for evaluating and selecting apo-holo pairs.

    Attributes
    ----------
    apo_chain: str
        The apo structure chain name. Default is 'A' for all monomer structures.
    contact_rad : float
        The radius to use for detecting contacts between interacting chains in apo-holo evaluation.
    backbone_only : bool
        Whether to limit contact search to only backbone atoms.
    heavy_only : bool
        Whether to limit contact search to heavy atoms.
    min_atom_types: int
        Minimum number of unique atom types to consider a monomer for apo evaluation.
    min_residues: int
        Minimum number of monomer residues to consider for apo evaluation.
    min_holo_resolved_frac: int
        Limit apo pairing to those monomers which have at least this fraction of
        the holo monomer residues resolved. Note: this does not take into account sequence
        alignment or interface residues.
    align_method: str
        Alignment backend to use when superimposing apo monomers to their holo counterparts.
        Allowed values are `pymol` and `biotite`. Default is `pymol`.
    max_refine_rmsd: float
        Maximum RMSD between the superimposed apo atoms after refinement cycles.
    min_aligned_apo_res_frac: float
        Minimum fraction of holo residues that are covered by the apo monomer superposition.
    min_seq_identity: int
        Minimum sequence identity between monomer and holo monomer to consider for apo pairing.
    max_interface_miss_frac: float
        Maximum fraction of holo interface residues that can be missing in the apo monomer.
    max_frac_monomer_dimer_sequence: float
        Maximum fraction of full holo dimer sequence represented by the single-body apo monomer.
        See PDB 2G3D (holo) and 1YJF for an example where this is needed.
    invalid_coverage_upper_bound: float
        Upper bound on ratio of the number of apo interface residues after superimposing to the counterpart holo monomer
        vs the holo interface residues for the monomer that it is being paired to before being considered invalid domain coverage.
    invalid_coverage_lower_bound: float
        Lower bound on ratio of the number of apo interface residues after superimposing to the counterpart holo monomer
        vs the holo interface residues for the monomer that it is being paired to before being considered invalid domain coverage.
    scaled_score_metrics: tuple[str]
        Metrics to use when constructing a scaled score for selecting a single
        canonical apo monomer for receptor and ligand holo monomers.

    """

    apo_chain: str = "A"
    contact_rad: float = 10.0
    backbone_only: bool = False
    heavy_only: bool = False
    min_atom_types: int = 3
    min_residues: int = 5
    min_holo_resolved_frac: float = 0.3
    align_method: str = "pymol"
    max_refine_rmsd: float = 10.0
    min_aligned_apo_res_frac: float = 0.7
    min_seq_identity: float = 0.3
    max_interface_miss_frac: float = 0.3
    max_frac_monomer_dimer_sequence: float = 0.75
    invalid_coverage_upper_bound: float = 2.0
    invalid_coverage_lower_bound: float = 0.5
    scaled_score_metrics: tuple[str, str, str, str, str] = (
        "I-RMSD",
        "refine_rmsd",
        "sequence_identity",
        "Fnat",
        "Fnonnat",
    )


@dataclass
class IalignConfig:
    """Configuration parameters for evaluating potential alignment leakage via iAlign.

    Attributes
    ----------
    rmsd_threshold: float
        The maximum RMSD reported by iAlign for considering an interface pair as similar.
    log_pvalue_threshold : float
        The maximum log P-value reported by iAlign for considering an interface pair as similar.
    is_score_threshold : bool
        The minimum IS-score value reported by iAlign for considering an interface pair as similar.
    alignment_printout : int
        The -a flag to pass to ialign.pl.
        0 - no alignment printout, 1 - concise, 2 - detailed.
    speed_mode : int
        The -q flag to pass to ialign.pl.
        1 - normal (default), 2 - fast.
    min_residues : int
        The -minp flag to pass to ialign.pl.
        Minimum number of residues for a protein chain.
    min_interface : int
        The -mini flag to pass to ialign.pl.
        Minimum number of residues for an interface.
    distance_cutoff : float
        The -dc flag to pass to ialign.pl.
        Distance cutoff for an interfacial contact, default 10.0 A.
    output_prefix: str
        The -w flag to pass to ialign.pl. Workpath or path to parsed PDB files.

    """

    rmsd_threshold: float = 5.0
    log_pvalue_threshold: float = -9.0
    is_score_threshold: float = 0.30
    alignment_printout: int = 0
    speed_mode: int = 1
    min_residues: int = 5
    min_interface: int = 5
    distance_cutoff: float = 10.0
    output_prefix: str = "output"


def get_config_hash(config_obj: ContactConfig | GraphConfig) -> str:
    config_hash = hashlib.md5(
        json.dumps(config_obj.__dict__, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return config_hash
