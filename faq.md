## Is there a sequence database available?
Yes! As of the latest release (2024-02), there is now an interface to fetch the full database of sequences for all PDB files available in the pinder dataset.

```python
>>> from pinder.core.index.utils import get_sequence_database
>>> seq_db = get_sequence_database()
>>> seq_db
                                            pdb                                           sequence  pdb_kind
0                    7sqc__ME1_A0A2K3DEG1-L.pdb  QKERVAFVTAVAEMAKNPQNIEALQQAGAMALLRPLLLDNVPSIQQ...    ligand
1                      5f2y__C1_UNDEFINED-R.pdb                          GEIAKALREIAKALREXAWAHREEA  receptor
2        2xvr__F51_P19726--2xvr__G55_P19726.pdb  LTADVLIYDIEDAMNHYDVRSEYTSQLGESLAMAADGAVLAEIAGL...     dimer
3                        7c9s__D39_Q8QWB2-R.pdb  GAQVSTQKTGAHIIHYTNINYYKDSASNSLNRQDFTQDPSKFTEPV...  receptor
4                        4v82__FA1_Q8DHA7-R.pdb                 MEVNQLGLIATALFVLVPSVFLIILYVQTESQQK  receptor
...                                         ...                                                ...       ...
3826759    4pkn__Q1_Q548M1--4pkn__T1_Q548M1.pdb  AAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAP...     dimer
3826760                   3lu0__C1_P0A8V2-L.pdb  KKRIRKDFGKRPQVLDVPYLLSIQLDSFQKFIEQDPEGQYGLEAAF...    ligand
3826761    6i0o__A1_P12268--6i0o__A4_P12268.pdb  SYVPDDGLTAQQLFNCGDGLTYNDFLILPGYIDFTADQVDLTSALT...     dimer
3826762                   8fc1__N1_Q5SHQ4-R.pdb  RLTAYERRKFRVRNRIKRTGRLRLSVFRSLKHIYAQIIDDEKGVTL...  receptor
3826763                  7ebr__D28_J9Z449-R.pdb                       NFYKDSYAASASKQDFSQDPSKFTEPVV  receptor

[3826764 rows x 3 columns]
>>> seq_db.describe()
                               pdb                                           sequence pdb_kind
count                      3826764                                            3826764  3826764
unique                     3826764                                             624894        5
top     7sqc__ME1_A0A2K3DEG1-L.pdb  ASNFTQFVLVDNGGTGDVTVAPSNFANGVAEWISSNSRSQAYKVTC...    dimer
freq                             1                                              17772  2319564
>>> seq_db.dtypes
pdb           object
sequence    category
pdb_kind    category
dtype: object
```

## Can `pinder` be used to detect steric clashes?
Yes! For `PinderSystem`-level evaluation, you can use the `pinder.eval.clashes.count_pinder_clashes` method.
```python
>>> from pinder.eval.clashes import count_pinder_clashes
>>> pinder_id = "2e31__A1_Q80UW2--2e31__B1_P63208"
>>> clashes = count_pinder_clashes(pinder_id)
>>> clashes
   atom_clashes  residue_clashes  min_dist  min_dist_vdw_ratio  vdw_sum  radius                                id monomer_name  holo_mask
0             0                0  2.738283            0.894864     3.06     1.2  2e31__A1_Q80UW2--2e31__B1_P63208         holo      False
1             0                0  2.513461            0.759354     3.31     1.2  2e31__A1_Q80UW2--2e31__B1_P63208          apo      False
2            15                7  0.388205            0.111233     3.49     1.2  2e31__A1_Q80UW2--2e31__B1_P63208    predicted      False
3             0                0  3.368342            0.990689     3.40     1.2  2e31__A1_Q80UW2--2e31__B1_P63208          apo       True
4             8                5  0.627183            0.206991     3.03     1.2  2e31__A1_Q80UW2--2e31__B1_P63208    predicted       True
```

To detect clashes in an arbitrary PDB file, use `pinder.core.structure.contacts`. **Note: currently clash detection is enabled only between two chains**

```python
>>> from pinder.core.structure.contacts import pairwise_clashes
>>> from pinder.core.loader.structure import Structure
>>> struct = Structure(pdb_file)
>>> AB = struct.filter("chain_id", ["A", "B"])
>>> clashes = pairwise_clashes(AB.atom_array)
>>> clashes
{'atom_clashes': 0, 'residue_clashes': 0, 'min_dist': 2.7268461070070824, 'min_dist_vdw_ratio': 0.8796277764538977, 'vdw_sum': 3.0999999999999996, 'radius': 1.2}
```

## What do holo, apo, native, and predicted structures within `PinderSystem` mean?
* `native`: really its the same as holo , but in the pinder case it means the native dimer interaction as extracted from RCSB, without any coordinate transformations (except for stripping of solvent and non-protein/modified-protein atoms). It should align with the RCSB structure if you load the `pinder` dimer and e.g. run `fetch <pdb_id>` in pymol.
* `holo`: bound conformation. Where bound in the `pinder` context means there are other interacting protein chains in the original RCSB crystal.
  * It is generally used to refer to the individual monomer chains that compose dimers. `holo-R` = Receptor from native, `holo-L` = Ligand from native
  * The only caveat is that for the test split, we transform the coordinates of the monomers using a PCA-based normalization to ensure that there are no biases introduced during evaluation from methods that may be leaky if the input proteins are pre-oriented.
  * All of the train split is left un-perturbed.
  * `PinderSystem` loads all of these for convenience, in case you need to make comparisons to the un-transformed version but tucks this into the code rather than the files on disk to ensure fair evaluation across methods.
* `apo`: unbound conformation. A uniprot-matched pairing of a structure from a different RCSB entry that is *not* interacting with other proteins
* `predicted`: structures from AF2 predictions which were uniprot-matched to the holo monomers across all systems
