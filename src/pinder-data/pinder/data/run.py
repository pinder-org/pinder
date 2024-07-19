"""Examples for running via CLI:
# Full pipeline
```
pinder_data run
```

# Specify a custom pinder root directory/mount point
```
pinder_data --pinder_mount_point ./pinder-data-pipeline run
```

# Run specific stage (download RCSB files associated with two-character code `bo`)
```
pinder_data --two_char_code bo run_stage download_rcsb_files
# OR
pinder_data --t bo run_stage download_rcsb_files
```

Example PDB ingest data directory structure:
./pinder-data-pipeline/data/bo/pdb_00006boo/
├── 6boo-assembly.cif
├── 6boo-entities.parquet
├── 6boo-interacting_chains.tsv
├── 6boo-metadata.tsv
├── 6boo-pisa-lite-assembly.json
├── 6boo-pisa-lite-interfaces.json
├── 6boo__A1_B0YD89--6boo__C1_B0YD89.pdb
├── 6boo__A1_B0YD89--6boo__C1_B0YD89.tsv
├── 6boo__A1_B0YD89-R.parquet
├── 6boo__A1_B0YD89-R.pdb
├── 6boo__C1_B0YD89-L.parquet
├── 6boo__C1_B0YD89-L.pdb
├── checkpoint-mapping.txt
├── checkpoint-pisa.txt
├── foldseek_contacts
│   └── 2f2691f67bd6fde5ba4ad0152799dc95
│       └── 6boo__A1_B0YD89--6boo__C1_B0YD89.json
└── pdb_00006boo_xyz-enrich.cif.gz

and the resulting PINDER dataset directory structure:

```
./pinder-data-pipeline/2024-02/
├── apo_metrics
│   ├── pair_eval
│   │   └── metrics_0.parquet
│   ├── scored_noisy_train_apo_pairings.parquet
│   └── two_sided_apo_monomer_metrics.parquet
├── chain_metadata.parquet
├── cluster
│   └── f6e35584321f647887eacb8ee369305f
│       ├── af2_lddt070_test_sys_table.csv
│       ├── af2_lddt070_transitive_hits_mapping.csv
│       ├── af2_lldt070_test_sys_table.csv
│       ├── af2_lldt070_transitive_hits_mapping.csv
│       ├── foldseek_af2_lddt070_leakage_dict.pkl
│       ├── foldseek_af2_lldt070_leakage_dict.pkl
│       ├── foldseek_af2_lldt070_potential_leaks.pkl
│       ├── foldseek_communities.pkl
│       ├── foldseek_components.pkl
│       ├── foldseek_leakage_dict.pkl
│       ├── foldseek_potential_leaks.pkl
│       ├── index.2.csv.gz
│       ├── mmseqs_components.pkl
│       ├── mmseqs_leakage_dict.pkl
│       ├── mmseqs_potential_leaks.pkl
│       ├── pindex_checkpoint.3.csv
│       ├── pindex_checkpoint.4.csv
│       ├── test_subset.csv
│       ├── test_sys_table.csv
│       └── transitive_hits_mapping.csv
├── dimer_ids.parquet
├── ecod_metadata.parquet
├── entity_metadata.parquet
├── enzyme_classification_metadata.parquet
├── external_annotations
│   └── sabdab_summary_all.tsv
├── foldseek  [672 entries exceeds filelimit, not opening dir]
├── graphs
│   └── 52d26a07886d2d2300c364a381680e8b
│       ├── cleaned_foldseek_alignment_graph.pkl
│       ├── cleaned_mmseqs_alignment_graph.pkl
│       ├── foldseek_alignment_graph.pkl
│       ├── min_length_interfaces.pkl
│       └── mmseqs_alignment_graph.pkl
├── ialign_metrics
│   ├── ialign_potential_leaks.parquet
│   ├── ialign_split_similarity_labels.parquet
│   ├── metrics.parquet
│   ├── pindex_checkpoint.5.parquet
│   └── potential_alignment_leaks.parquet
├── index.1.csv.gz
├── index.parquet
├── index_with_apo.parquet
├── index_with_pred.parquet
├── interface_annotations.parquet
├── interfaces.parquet
├── mappings  [1363 entries exceeds filelimit, not opening dir]
├── metadata.1.csv.gz
├── metadata.2.csv.gz
├── metadata.parquet
├── mmseqs2
│   ├── input.fasta
│   └── mmseqs_dbs
│       ├── 00000
│       │   ├── 00000
│       │   │   ├── alignment.txt
│       │   │   └── mmseqs_error.txt
│       │   └── db
│       │       └── input_00000.fasta
│       ├── alignment.parquet
│       ├── alignment.txt
│       └── filtered_alignment.parquet
├── monomer_ids.parquet
├── monomer_predicted_ids.parquet
├── pdbs  [2890 entries exceeds filelimit, not opening dir]
├── putative_apo_monomer_ids.parquet
├── putative_two_sided_apo_pairings.parquet
├── rcsb_annotations
│   ├── annotations  [268 entries exceeds filelimit, not opening dir]
│   ├── annotations_cath.csv.gz
│   ├── annotations_ecod.csv.gz
│   ├── annotations_other.csv.gz
│   ├── annotations_scop.csv.gz
│   ├── enzyme_classification  [268 entries exceeds filelimit, not opening dir]
│   ├── enzyme_classification.csv.gz
│   ├── features  [268 entries exceeds filelimit, not opening dir]
│   ├── features_asa.csv.gz
│   ├── features_binding_site.csv.gz
│   ├── features_cath.csv.gz
│   ├── features_ecod.csv.gz
│   ├── features_occupancy.csv.gz
│   ├── features_other.csv.gz
│   ├── features_outlier.csv.gz
│   ├── features_sabdab.csv.gz
│   ├── features_scop.csv.gz
│   ├── features_unobserved.csv.gz
│   ├── pfam  [268 entries exceeds filelimit, not opening dir]
│   ├── pfam.csv.gz
│   └── query_data  [268 entries exceeds filelimit, not opening dir]
├── sabdab_metadata.parquet
├── scored_apo_pairings.parquet
├── structural_metadata.parquet
├── supplementary_metadata.parquet
└── test_set_pdbs
    ├── 4boq__A1_Q5VVQ6-R.pdb
    ├── 4boq__A2_Q5VVQ6-L.pdb
    ├── 4boz__A1_Q5VVQ6-R.pdb
    ├── 4boz__B1_P0CG48-L.pdb
    ├── 5bot__A1_P45452-R.pdb
    ├── 5bot__B1_P45452-L.pdb
    ├── 8bo1__A1_P68135-L.pdb
    ├── 8bo1__B1_A0A9P1NJI6-R.pdb
    ├── 8bo8__A1_Q04609-R.pdb
    ├── 8bo8__A2_Q04609-L.pdb
    ├── 8bos__A1_P01112-L.pdb
    ├── 8bos__B1_P20936-R.pdb
    ├── 8bou__A1_A0A7G5MNS2-R.pdb
    └── 8bou__B1_A0A7G5MNS2-L.pdb
```

"""

from pinder.core.utils import setup_logger
from pinder.data import (
    rcsb_rsync,
    get_data,
    get_dimers,
    get_annotations,
    foldseek_utils,
    get_clusters,
)
from pinder.data.pipeline.data_pipeline import DataIngestPipeline

log = setup_logger(__name__)


def method_main() -> None:
    import fire

    fire.Fire(
        {
            "download": rcsb_rsync.download_rscb_files,
            "ingest": get_data.ingest_rscb_files,
            "annotate": get_annotations.get_annotations,
            "collect": get_annotations.collect,
            "format": get_dimers.index_dimers,
            "foldseek": foldseek_utils.run_foldseek_on_pinder_chains,
            "mmseqs": foldseek_utils.run_mmseqs_on_pinder_chains,
            "cluster": get_clusters.cluster,
            # "select_benchmark": select_benchmark.clusters_to_df,
            # "add_interface_contact": contact_classification.get_crystal_contact_parallel,
            # "clean": deleak_edges.find_leaked_edges,
            # "split": select_benchmark.move_raw_files_to_splits,
        }
    )


def main() -> None:
    import fire

    fire.Fire(DataIngestPipeline)


if __name__ == "__main__":
    main()
