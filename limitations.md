# Known limitationns


## Dimers
`PINDER` exclusively utilizes interactions at dimeric scale for training and evaluation. Even though we utilize biological dimers in the test and validation sets, in training split they were often extracted from higher-order oligomeric complexes. This could limit their biological plausibility in isolation, for example, the binding of two monomers can create a neo-substrate for a third interacting chain that would otherwise not have the requisite binding affinity.

Additionally, the classification of protein complexes into oligomeric states, such as homodimers and heterodimers, relies heavily on specific definitions and contextual interpretations. This variability can introduce inconsistencies in the analysis and interpretation of interaction data. Furthermore, the `pinder` dataset exhibits a heavy bias towards homodimers, which is reflective of the bias observed in the PDB source data.


## AlphaFold2 monomers
While the `pinder` dataset includes many predicted AlphaFold2 monomers, it is limited to only those present in AFDB 2.3. For structures with undefined UniProt IDs, we could have inferred sequences to broaden the dataset.


## Apo pairing
Pairing *apo* forms could be improved by integrating Foldseek and MMseqs2 to search for homologous monomers, as the current approach is primarily UniProt-based.


## Single reference conformation
Evaluating protein interactions using a single reference conformation restricts analysis flexibility. Proteins can adopt multiple conformations, and relying on one reference may not capture the full range of interactions, thereby reducing accuracy. This is particularly important to note given the inclusion of *apo* and predicted monomers as augmentation. Incorporating multiple conformational states can provide a more accurate and comprehensive assessment of protein-protein interactions.


## Alignment methodology
Our primary focus was to minimize leakage within the test/validation splits. To achieve this, we employed low score thresholds in our all-vs-all score computations, specifically using Foldseek and MMseqs2. While alignments between monomers were required to have a minimum coverage of the interface for each monomer, the alignments themselves were not specific to the interfaces. Additionally, we masked training data with indirect (transitive) hits at a depth of two to enhance recall. Although this approach did yield a diverse set of clusters, it has limitations. Dissimilar interfaces may have been considered similar and consequently removed from the training data, and individual clusters can have a high range of interfaces. Future iterations could address this by incorporating tools with greater precision, such as iAlign, into the splitting algorithm.


### Foldseek limitations
Foldseek is a tool designed for fast and sensitive alignment of protein structures. It excels in identifying similar structures by comparing the 3D coordinates of proteins, making it useful for structural clustering. However, Foldseek has several limitations. First, Foldseek primarily handles dimeric structures and does not efficiently accommodate higher-order oligomeric complexes. This limitation means that for oligomeric proteins, the interactions within larger complexes may not be accurately represented. Additionally, Foldseek's reliance on structural alignment can miss functionally significant variations that do not manifest as large-scale structural differences. For example, subtle changes in the side-chain orientations at the binding interface, which can be critical for interaction specificity, might not be captured by Foldseek's alignment algorithms. These limitations highlight the need for complementary approaches and tools to achieve a more accurate and comprehensive clustering of protein interfaces.


## PRODIGY-Cryst
For the `pinder` test, we selected protein complexes using multiple criteria to ensure quality, though each choice involved trade-offs. The threshold of 0.5 in the PRODIGY algorithm signifies a relatively low-quality interaction prediction, with scores between 0.5 and 0.6 falling into this category. Moreover, scores below 0.5 might still represent biologically relevant interfaces, potentially affecting the accuracy of interaction assessments.


## Data types in the `pinder` index

By default, the index and metadata has dtypes that have been optimized for reduced memory usage due to the large table sizes. While this should be a non-issue for most use-cases, there may be certain operations that require casting the column dtypes to their more commonly used variants. Some known issues:
* `float16` not supported by `pandas.cut` -> need to cast to float `data[column] = data[column].astype(float)`
* `category` dtype can have some quirks
  * `pandas.DataFrame.groupby` should pass `observed=True` to ensure categorical columns only report the *observed* values, rather than all categories encoded in the column prior to any mutation.
  * Depending on the environment and how you are accessing items from the dataset, you may need to convert to object type: `for col in index.select_dtypes(include=["category"]).columns: index[col] = index[col].astype('object');`
