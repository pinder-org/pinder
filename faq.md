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


## For some protein pairs, when I extract the apo and holo structures and align their sequence, the results don't have the same atoms and/or sequence. How can I align monomers such that all have the same shape?

For a more detailed example, see the [example notebook](examples/superposition.ipynb) (also available in the [documentation](https://pinder-org.github.io/pinder/superposition.html)).


```python
from pinder.core import PinderSystem

pinder_id = "2gct__C1_P00766--2gct__A1_P00766"
ps = PinderSystem(pid)

# Single alternative monomer use-case without cropping
apo_RL = ps.create_apo_complex(remove_differing_atoms=False)
holo_RL = ps.aligned_holo_R + ps.aligned_holo_L

print(apo_RL.atom_array.shape[0], (holo_RL).atom_array.shape[0])
>>> (764, 758)

# With cropping such that apo only contains atom that are in holo. No guarantee that holo doesn't contain "extra" atoms not in apo.
apo_RL = ps.create_apo_complex(remove_differing_atoms=True)
print(apo_RL.atom_array.shape[0], (holo_RL).atom_array.shape[0])
>>> (756, 758)

# You can also renumber the residues in the apo structure to match the holo structure.
apo_RL = ps.create_apo_complex(remove_differing_atoms=True, renumber_residues=True)

# Pairwise cropped structure superposition: ensure ALL monomer have the same shape.
# Note: this will modify the ground-truth structure.
holo_cropped, apo_cropped, pred_cropped = ps.create_masked_bound_unbound_complexes(
  monomer_types=["apo"],
  renumber_residues=True,
)
print((
    holo_cropped.atom_array.shape[0],
    apo_cropped.atom_array.shape[0],
    pred_cropped.atom_array.shape[0],
))
>>> (756, 756, 3598)

# Pairwise including predicted monomers
holo_cropped, apo_cropped, pred_cropped = ps.create_masked_bound_unbound_complexes(
  monomer_types=["apo", "predicted"],
  renumber_residues=True,
)
print((
    holo_cropped.atom_array.shape[0],
    apo_cropped.atom_array.shape[0],
    pred_cropped.atom_array.shape[0],
))
>>> (756, 756, 756)
```


## How can I use the evaluation harness outside of a `pinder` context?

For a more detailed example, see the [example notebook](examples/pinder-eval.ipynb) (also available in the [documentation](https://pinder-org.github.io/pinder/pinder-eval.html#pinder-eval-entrypoint)).

In short, the DockQ-related classes provided in pinder roughly follow this hierarchy:
* `pinder.eval.dockq.method.MethodMetrics`
  * For evaluating methods over many systems and producing a pinder-specific leaderboard
  * Wraps BiotiteDockQ
* `pinder.eval.dockq.biotite_dockq.BiotiteDockQ`
  * The DockQ calculation interface, designed to take one native/reference and score N decoys
  * Performs I/O
  * Does calculations that only need to be done one (native contacts, patching canonical order, etc)
  * Makes calls to `DecoyDockQ`  in serial for list of `AtomArray` or vectorized over `AtomArrayStack`
* `pinder.eval.dockq.biotite_dockq.DecoyDockQ`
  * Takes pre-prepared models and native information and gets the metrics


For the simplest interface to calculating DockQ metrics for arbitrary inputs, say you have a ground-truth structure with 2 chains, R and L, and a list of predicted models with chains A and B.

If you know which chains between the ground-truth and models correspond with each other, you can specify the chains via `native_receptor_chains, native_ligand_chain, decoy_receptor_chain, decoy_ligand_chain`. If the chains are not provided, the receptor chain will be inferred based on the larger chain in both native and decoy structures.

**Note the use of a list: it is possible to specify more than one chain to assign as receptor or ligand.**

```python
native_receptor_chain = ["R"]
native_ligand_chain = ["L"]
decoy_receptor_chain = ["A"]
decoy_ligand_chain = ["B"]
```

```python
from pathlib import Path
from pinder.eval import BiotiteDockQ

native = Path("./native.pdb")
models = list(Path("./models").glob("*.pdb"))

bdq = BiotiteDockQ(
    native,
    models,
    backbone_definition="dockq",
    native_receptor_chain=native_receptor_chain,
    native_ligand_chain=native_ligand_chain,
    decoy_receptor_chain=decoy_receptor_chain,
    decoy_ligand_chain=decoy_ligand_chain,
    parallel_io=True,
)
metrics = bdq.calculate()
```

## Do I have to use the `pinder` loader? How can I write my own loader?

No! The `PinderDataset` torch dataset and `PPIDataset` torch-geometric dataset serve as example implementations of a loader. The dataset wraps the abstract `PinderLoader` class under the hood, which is the primary interface for loading `PinderSystem` objects and applying filters and/or transforms.

While not required, it is recommended to use the `PinderLoader` class directly to implement your own loader.

The `PinderLoader` brings together filters, transforms and writers to create an optionally parallel `PinderSystem` iterator. It can accept either a specific split name or list of system IDs as input.
```python
from pinder.core import PinderLoader
from pinder.core.loader import filters

base_filters = [
    filters.FilterByMissingHolo(),
    filters.FilterSubByContacts(min_contacts=5, radius=10.0, calpha_only=True),
    filters.FilterByHoloElongation(max_var_contribution=0.92),
    filters.FilterDetachedHolo(radius=12, max_components=2),
]
sub_filters = [
    filters.FilterSubByAtomTypes(min_atom_types=4),
    filters.FilterByHoloOverlap(min_overlap=5),
    filters.FilterByHoloSeqIdentity(min_sequence_identity=0.8),
    filters.FilterSubLengths(min_length=0, max_length=1000),
    filters.FilterSubRmsds(rmsd_cutoff=7.5),
    filters.FilterByElongation(max_var_contribution=0.92),
    filters.FilterDetachedSub(radius=12, max_components=2),
]

# Load a list of system IDs
systems = [
    "1df0__A1_Q07009--1df0__B1_Q64537",
    "117e__A1_P00817--117e__B1_P00817",
]

loader = PinderLoader(
    ids=systems,
    base_filters = base_filters,
    sub_filters = sub_filters
)
print(loader)
>>> PinderLoader(split=None, monomers=holo, systems=2)

# Iterate over loader to apply the filters and transforms. Systems that pass all filters will be yielded.
passing_ids = []
for item in loader:
    system, feature_complex, target_complex = item
    passing_ids.append(system.entry.id)

# Each dimer object can now be used in your own loader to access any of the properties exposed by the PinderSystem class, the monomer `Structure` objects, or the underlying `biotite.structure.AtomArray` objects.
```
