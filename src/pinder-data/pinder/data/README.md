# PINDER Dataset Generation

The PINDER dataset is generated through a series of steps, each contributing to the final dataset. Here is a brief overview of the process:

Stages encoded in the `DataPipeline`:

1. download_rcsb_files: runs rsync to fetch mmcif files from the `rsync-nextgen.wwpdb.org::rsync` RCSB server
2. ingest_rcsb_files: process mmcif files to generate bio assemblies, save structural metadata, classify structures into monomers and dimers, and save PDB files
3. get_pisa_annotations: fetch annotations for each PDB ID from PDBe PISA Lite service.
4. get_rcsb_annotations: fetch annotations for each PDB ID from RCSB GraphQL API
5. get_dimer_annotations: extract custom annotations for each dimer PDB file generated in step 2.
6. get_dimer_contacts: extract interface residues in foldseek numbering (see note below) for each dimer PDB file generated in step 2.
7. populate_entries: transfer required files generated in step 2-5 into the target pinder directory.
8. populate_predicted: populate target pinder pdbs directory with AF2 monomers based on uniprot ID (if they exist).
9. index_dimers: collate all of the interface and structural metadata + annotations and construct an intermediate index of all dimer entries.
10. add_predicted_monomers: set the `predicted_R_pdb` and `predicted_L_pdb` columns in the intermediate index corresponding to the AF2 monomers found in step 8.
11. get_apo: evaluate all possible pairings between holo dimer monomers and true monomers based on data from step 7, initially using uniprot to pair followed by subsequent structure and sequence-based metrics. Remove any invalid pairings and select a "canonical" pair for the receptor and ligand side, respectively, based on a normalized score comprised of multiple metrics. Set the `apo_R/L` boolean, `apo_R/L_pdb` canonical pair PDB, and `apo_R/L_pdbs` alternative pairs(s) columns in the index.
12. foldseek: run foldseek on the unique asym_id's for each PDB ID. (e.g., XXXX_A1 & XXXX_A2 will reduce to XXXX_A)
13. mmseqs: run mmseqs on the unique asym_id's for each PDB ID. (e.g., XXXX_A1 & XXXX_A2 will reduce to XXXX_A)
14. interface_graph: process the interfaces computed in step 6. into a graph of `Interface` objects.
15. foldseek_graph: process the foldseek alignment into a directed (`nx.DiGraph`) graph, with nodes corresponding to the monomer chains with an alignment hit and edges with attributes storing the aligned residue indices and the `lddt` metric as a score. Cross this graph with the interface graph from step 15, only including edges if the aligned residue indices overlap with the interface residue indices and add the alignment score as an edge weight. Convert the final graph to an undirected reciprocal graph.
16. mmseqs_graph: process the mmseqs alignment into a directed (`nx.DiGraph`) graph, with nodes corresponding to the monomer chains with an alignment hit and edges with attributes storing the aligned residue indices and the `pident` metric as a score. Cross this graph with the interface graph from step 15, only including edges if the aligned residue indices overlap with the interface residue indices and add the alignment score as an edge weight. Convert the final graph to an undirected reciprocal graph.
17. cluster: cluster dimer-monomers based on their interfaces using the alignment graphs from step 16 and 17. Assign primary cluster ID based on an asynchronous community clustering of the foldseek graph. Store alternative cluster identifiers based on and independent component clustering of the foldseek and mmseqs graphs.
18. deleak: split the dataset into "proto-test" and "proto-train", based on desired structural criteria for the test set. Search the interface graphs for transitive hits found in different cluster IDs by traversing the graph to find shortest path neighbors within a depth of 2 away from each of the systems in the "proto-test" split. Remove any such leakage and define the valid members for test and val splits to be sampled in step 19.
19. get_splits: Remove leakage identified in step 18 and assign the final test, val, train split labels. Anything that fails to be assigned to `test, val, train` is kept in the index and assigned `split="invalid"`. Assign a putative pinder-af2 split based on a time split (PDB entries deposited after the AF2 training cutoff date) and then find systems within this split that have low or no interface similarity to any other members in the pinder dataset (including test) using a more rigorous deleaking process performed by interface alignment via iAlign. Remove members from the time split that have similarity to other AF2 train set members and define the final `pinder_af2` holdout set.
20. get_test_set: assign `pinder_xl` test subset to all systems with `split="test"`. Curate the `pinder_s` test subset by prioritizing heteodimer systems with paired apo and AF2 monomer availability and requiring unique uniprots across all systems.


# Running the data pipeline locally via CLI

## Full pipeline
```bash
pinder_data run
```

**Specify a custom pinder root directory/mount point**

```bash
pinder_data --pinder_mount_point ./pinder-data-pipeline run
```

## Run specific stage (download RCSB files associated with two-character code `bo`)

```bash
pinder_data --two_char_code bo run_stage download_rcsb_files
# OR
pinder_data --t bo run_stage download_rcsb_files
```


**Example PDB ingest data directory structure:**
```bash
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
```

and the resulting PINDER dataset directory structure:

```bash
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


Running the command:
```
pinder_data download -d /tmp/pinder -t ww
```

will results in the following directory strucutre

```
data_dir/
└── ww
    ├── pdb_00001wwz
    │   └── pdb_00001wwz_xyz-enrich.cif.gz
    ├── ...
    ...
```

1. **NextGen RCSB PDB Ingestion**: The RCSB Protein Data Bank (PDB) is a database of 3D structural data of large biological molecules, such as proteins and nucleic acids. The NextGen RCSB PDB Ingestion process involves fetching the latest data from the PDB.

2. **Metadata Generation**: Metadata for each RCSB PDB entity, such as its deposition date, method, source, resolution, oligomeric state and other relevant information, is generated.

3. **Bioassembly Generation**: Bioassemblies are the biologically active forms of molecules. They are generated from the NextGen RCSB PDB data that comes in annotated mmCIF format. Only the first bioassembly is generated. Source data is downloaded with `rsync` and each file is processed to generate the bioassemblies and their metadata.

4. **Chain Renaming**: The chains in the bioassembly are renamed for consistency and ease of use. In particular, copies of the same protein are renamed to make them uniquely identifiable: `A` is renamed to `A1`, `A2`, and so on.

5. **Uniprot Mapping Generation on Residue Level**: The PDB data is mapped to Uniprot data at the residue level. Uniprot is a database of protein sequence and annotation data.

6. **PDB File Naming**: The PDB files are named in a consistent manner for ease of use. In a dimer complex, the receptor chain always comes first and the ligand chain comes second. Receptor vs. ligand is assigned based on the residue count of each chain. If the chains have identical residue counts, the receptor chain is defined alphabetically. The pinder IDs for each dimer follow the convention: `<pdbid>__<asym_id><copy_number>_<uniprot>--<pdbid>__<asym_id><copy_number>_<uniprot>`.

7. **Annotation**: Each dimer is annotated with its binding site, and interface properties such as buried solvent accessible surface area (SASA), assessment of the oligomeric state in the bioassembly, planarity of the interface as well as other properties of the biomolecule (such as presence of structurally-unresolved gaps)

8. **Generation of Monomers and Dimers**: Monomers (single molecule units) and dimers (two molecule units) are generated from the PDB data. A structure is defined as a monomer if the biological assembly has no interacting chains within a pre-defined contact radius (default is 10Å backbone-backbone atom radius).

Note that `<google-cloud-project>` is the name of the Google Cloud project that you have access to in order to access the AlphaFold models in DeepMind's Google Cloud Storage bucket `gs://public-datasets-deepmind-alphafold-v4`, where the user has to pay for egress costs. Refer to `gcloud` utility documentation for more information on how to set up your `gcloud auth` environment and project. You can also modify the `alphafold_path` variable to change the path to the AlphaFold models.

The `index.1.csv.gz` and `metadata.1.csv.gz` files represent "work in progress" intermediate metadata and index. The index still has multiple empty columns that will be filled in downstream steps.

9. **Foldseek**: All monomers are compared with structural alignement (FoldSeek)

10. **Clustering**: Structural alignment (FoldSeek) and sequence identity of the interfacial residues are used to create interface similarity graph. Community clustering of the interface similarity graph yields interface clusters, which are then joined into paired interface clusters by mapping the corresponding dimers.

11. **Identification of matching monomers**: For each dimer, the corresponding unbound (apo) monomers are identified. This is done by aligning the monomers to the dimer and finding the best match according to the selection criteria

12. **Generation of Splits**: The dimers are split into training, validation, and test sets according to their clusters. Additionally, for test and validation splits a set of filters is applied and only one cluster representive is kept.
