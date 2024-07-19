![pinder](https://github.com/pinder-org/pinder/raw/main/assets/pinder.png)
<div align="center">
    <h1>PINDER: The Protein INteraction Dataset and Evaluation Resource</h1>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/pinder)](https://pypi.org/project/pinder/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/pinder-org/pinder/blob/master/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/pinder-org/pinder)](https://github.com/pinder-org/pinder/stargazers)
[![test](https://github.com/pinder-org/pinder/actions/workflows/pr.yaml/badge.svg)](https://github.com/pinder-org/pinder/actions/workflows/pr.yaml)
[![codecov](https://codecov.io/gh/pinder-org/pinder/graph/badge.svg?token=NPQAYW75OD)](https://codecov.io/gh/pinder-org/pinder)

# 📚 About

**pinder**, short for **p**rotein **in**teraction **d**ataset and **e**valuation **r**esource, is a dataset and resource for training and evaluation of protein-protein docking algorithms. It is ~500x larger than previous state of the art datasets and is the first dataset to include paired predicted and apo structures to train flexible docking methods.

The dataset is large (~700Gb) and hosted on Google Cloud Storage (available at the `gs://pinder` bucket).

# 👨‍💻 Getting Started

## Prerequisites

### fastpdb support

pinder uses [fastpdb](https://github.com/biotite-dev/fastpdb) to accelerate PDB
I/O operations. fastpdb is a dependency of pinder-core, and pip will attempt to
install it for you during the installation of pinder. Pre-built wheels of
fastpdb are available on PyPI for the following platforms:
1. Linux with `glibc>=2.34` (e.g., Debian 12, Ubuntu 22.04, RHEL 9, etc.)
2. Intel-based (x86, not Apple Silicon) MacOS Sierra (10.12) or newer
3. Windows

If your platform doesn't match these conditions, you will not get a wheel and
pip will attempt to build fastpdb from source. In order to build fastpdb from
source, you will need the rust toolchain, which you can install by running:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

before installing pinder.

## Install pinder

### Initialize a virtual environment or conda environment

We recommend installing pinder into a clean virtual environment or conda
environment. This can be done using
[`mamba`](https://github.com/mamba-org/mamba) or `conda` (you can swap `mamba`
for `conda` for the same functionality):

**Linux and Intel-based (x86-64) CPU architecture**

```bash
mamba create --name pinder python=3.10
mamba activate pinder
```

**Apple Silicon-based (ARM) CPU architecture**

Unfortunately, until all pinder dependencies have an ARM wheel on PyPi, we have to instruct conda to use an osx-64 target arch.

```bash
CONDA_SUBDIR=osx-64 mamba create --name pinder python=3.10
mamba activate pinder
mamba env config vars set CONDA_SUBDIR=osx-64
mamba deactivate
mamba activate pinder
```

or via `venv` from the Python standard library:

```bash
# clone the repo and cd into it, unless you plan to install from PyPI, then
python3 -m venv venv
source venv/bin/activate
```

## Install optional dependencies

### pytorch-cluster

`pytorch-cluster` is an optional dependency for pinder. If you wish to make use
of its features, you will need to install it separately.

To install from a wheel, run

```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where `${TORCH}` should be replaced by the version of PyTorch installed and `${CUDA}` should be replaced by either `cpu`, `cu118`, or `cu121`
depending on your PyTorch installation.

To install from source, first make sure you have pytorch installed in the
current environment (`pip install torch`), then run

```bash
pip install torch-cluster
```

Note that on Apple Silicon MacOS machines, installation from source is the only option.

### PRODIGY-cryst

PRODIGY-cryst is used in the data ingestion pipeline to predict the probability that an interface is a biological interaction. While it is not needed to use `pinder.core`, it is an optional dependency of `pinder.data` and can be installed as a git-based installation. To install, run

```bash
pip install git+https://github.com/yusuf1759/prodigy-cryst.git
```

### Install pinder packages from PyPI

To install with the minimal dependencies needed to use `pinder.core`
```bash
pip install pinder
```

Install optional extras, for instance to use the `pinder.eval` package

```bash
pip install pinder[eval]
```

Or, install all extras

```bash
pip install pinder[all]
```


# ⬇️ Getting the dataset

We strongly recommend to interact via the provided python API in `pinder-core`, as follows, which will automatically download and load the data into either `$PINDER_BASE_DIR` or `$XDG_DATA_HOME` (usually `~/.local/share/pinder` on Mac and Linux) if no explicit download path is provided (recommended)

NOTE: the default location for the dataset is `~/.local/share/pinder/<release version>`

If you want to use a different location, you can do so by setting the `PINDER_BASE_DIR` environment variable.

The base dir refers to a fully qualified path name up until the `<release version>` (not inclusive).

For instance, you could:
```bash
export PINDER_BASE_DIR=~/my-custom-location-for-pinder/pinder
```

You can always check the current location of the dataset like so:
```python
from pinder.core import get_pinder_location
get_pinder_location()
```

The current release version of pinder is `2024-02`.

You can find the list of available dataset releases and the associated changes in the [data changelog](changelog_data.md).

## To download the complete dataset run the following

```
pinder_download --help

usage: Download latest pinder dataset to disk [-h] [--pinder_base_dir PINDER_BASE_DIR] [--pinder_release PINDER_RELEASE] [--skip_inflation]

optional arguments:
  -h, --help            show this help message and exit
  --pinder_base_dir PINDER_BASE_DIR
                        specify a non-default pinder base directory
  --pinder_release PINDER_RELEASE
                        specify a pinder dataset version
  --skip_inflation      if passed, will only download the compressed archives without unpacking
```

The full dataset should look like this:

```bash
get_pinder_location()/
    pdbs/
    test_set_pdbs/
    mappings/
    index.parquet
    metadata.parquet
```

* `pdbs/` contains individual monomer and ground-truth dimer PDB structures
* `mappings/` contains mapping information for holo and apo monomers for PDB<->uniprot, as well as original PDB assembly information used in some utilities
* `index.parquet` contains the master index of every dimer in pinder. See [here](examples/pinder-index.ipynb) for more details.
* `metadata.parquet` contains additional metadata detail for each entry in the index.

It is also possible to download it manually, via

```bash
export PINDER_RELEASE=2024-02
export PINDER_ROOT=pinder/$PINDER_RELEASE
mkdir -p $XDG_DATA_HOME/$PINDER_ROOT/
gsutil -m cp gs://$PINDER_ROOT/pdbs.zip $XDG_DATA_HOME/$PINDER_ROOT/
gsutil -m cp gs://$PINDER_ROOT/test_set_pdbs.zip $XDG_DATA_HOME/$PINDER_ROOT/
gsutil -m cp gs://$PINDER_ROOT/mappings.zip $XDG_DATA_HOME/$PINDER_ROOT/
gsutil -m cp gs://$PINDER_ROOT/index.parquet $XDG_DATA_HOME/$PINDER_ROOT/
gsutil -m cp gs://$PINDER_ROOT/metadata.parquet $XDG_DATA_HOME/$PINDER_ROOT/
cd $XDG_DATA_HOME/$PINDER_ROOT
unzip pdbs.zip && rm pdbs.zip
unzip test_set_pdbs.zip && rm test_set_pdbs.zip
unzip mappings.zip && rm mappings.zip
```

however, this is discouraged and requires installing gsutil.

Note: to download the full dataset, you will need ~700Gb of free disk space.
```
# compressed
144G    pdbs.zip
149M    test_set_pdbs.zip
6.8G    mappings.zip

# unpacked
672G    pdbs
705M    test_set_pdbs
25G     mappings
```

## Updating the dataset
In the event that there are patch (non-breaking) changes to the index or metadata, you can sync your local copy of the index using a similar command-line interface:

```
pinder_update_index --help

usage: Download latest pinder index to disk [-h] [--pinder_base_dir PINDER_BASE_DIR] [--pinder_release PINDER_RELEASE] [--skip_inflation]

optional arguments:
  -h, --help            show this help message and exit
  --pinder_base_dir PINDER_BASE_DIR
                        specify a non-default pinder base directory
  --pinder_release PINDER_RELEASE
                        specify a pinder dataset version
```

If any *structure* files have been changed (will be announced in [data changelog](changelog_data.md)), but a major release (PINDER_RELEASE) has not yet been published, to sync your local dataset:

```
pinder_sync_data --help
usage: Sync missing pinder structural data files to disk [-h] [--pinder_base_dir PINDER_BASE_DIR] [--pinder_release PINDER_RELEASE] [--skip_inflation]

optional arguments:
  -h, --help            show this help message and exit
  --pinder_base_dir PINDER_BASE_DIR
                        specify a non-default pinder base directory
  --pinder_release PINDER_RELEASE
                        specify a pinder dataset version

```


# Pinder datasets & resources

## 1. 🏅 Gold standard benchmark sets
A set of 4 interface structure & sequence-deleaked, gold standard benchmark sets, all of which were redundancy removed and filtered to be of highest quality

| Dataset         |   # of PDB IDs |   # of Clusters |   # Holo pairs |   # Apo pairs |   # AF2 Pairs | Description                                                                                                                                                                                                                                                                                                                                                               |
|:----------------|---------------:|----------------:|---------------:|--------------:|--------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PINDER-XL       |           1955 |            1955 |           1955 |           342 |          1747 | Full test set, 1,955 cluster representatives, including 342 apo paired structures and 1,747 AFDB structures                                                                                                                                                                                                                                                               |
| PINDER-S        |            250 |             250 |            250 |            93 |           250 | A smaller subset of PINDER-XL, comprised of 250 clusters (188 heterodimer and 62 homodimers) sampled for diverse Uniprot and PFAM annotations, 93 of which have apo paired structures (143 have at least one apo monomer) and all of which have paired AFDB structures, to evaluate methods for which sampling from the full set is too slow                              |
| PINDER-AF2      |            180 |             180 |            180 |            30 |           127 | A smaller subset of PINDER-XL, comprised of 180 clusters, 30 of which have paired apo structures and 131 with paired AFDB structures, which were deleaked against the AF2MM training set with a more rigorous deleaking process to remove any members with interfaces similar to the AF2MM training set as determined by iAlign, to evaluate methods against AF2MM                                                                                                                                                |


All of these contain ready to use, pre-rotated & translated monomer structures


**A validation holdout set:**

| Dataset   |   # of PDB IDs |   # of Clusters |   # Holo pairs |   # Apo pairs |   # AF2 Pairs | Description                                                                                                                                         |
|:----------|---------------:|----------------:|---------------:|--------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| Val       |           1958 |            1958 |           1958 |           342 |          1789 | Validation set, consisting of 1,958 cluster representatives, of which 342 have paired apo structures and 1,789 of which have paired AFDB structures |


**A training set which provides an extensive number of possible training examples:**

|                              |                                                                                                                                                             |
|:-----------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset                      | Train                                                                                                                                                           |
| Size<sup>1</sup>             | 2456152                                                                                                                                                         |
| Theoretical Size<sup>2</sup> | 25130994                                                                                                                                                        |
| # of PDB IDs                 | 62706                                                                                                                                                           |
| # of Clusters                | 42220                                                                                                                                                           |
| # Apo Pairs                  | 136498                                                                                                                                                          |
| # AF2 Pairs                  | 566171                                                                                                                                                          |
| # At least one Apo Monomer   | 274194                                                                                                                                                          |
| # At least one AF2 Monomer   | 621276                                                                                                                                                          |
| Description                  | Training set, consisting of 1,560,682 dimers from 42,220 clusters, of which 136,498 have paired apo structures and 566,171 of which have paired AFDB structures |



1. Size refers to the sum of training examples with at least one Apo monomer (274,194), at least one AF2 monomer (621,276), and the holo monomers (1,560,682)
2. Theoretical size refers to the theoretical number of training examples made available by pinder. It includes all of the available Apo monomers for each of receptor and ligand, respectively, and all of the combinations with other monomer types. E.g., holo-receptor + apo-ligand1, AF2-receptor + apo-ligand2, etc.



See [Dataset Generation](#-dataset-generation) for details on how the dataset was generated.

##  2. 📊 Leaderboard
A **leaderboard** of the current state of the art physics-based docking methods as reference

| Type    | Name       | Train Dataset | Leaderboards
|---------|------------|---------| ----|
| Physics | FroDock    | N/A     | `PINDER-XL`, `PINDER-S`, `PINDER-AF2` |
| Physics | PatchDock  | N/A     |`PINDER-XL`, `PINDER-S`, `PINDER-AF2` |
| Physics | HDock      | N/A     |`PINDER-XL`, `PINDER-S`, `PINDER-AF2` |
| ML      | DiffDock-PP| pinder-holo    |`PINDER-XL`, `PINDER-S`, `PINDER-AF2` |
| ML      | AF2-MM    | af2mm    |`PINDER-AF2` |


## 3. ⚖️ Evaluation harness

A complete evaluation harness with a set of highly efficient pure-python or rust implementations of standard metrics for evaluation, such as DockQ is provided.

We use the community-standard CAPRI metrics for assessing docking methods. Further detail can be found [here](https://predictioncenter.org/casp15/doc/presentations/Day2/Assessment_Assembly-CAPRI_MLensink.pdf)

The evaluation harness can be used either through methods in pinder.eval or as a CLI script:

```
pinder_eval --help

usage: pinder_eval [-h] --eval_dir eval_dir [--serial] [--method_name method_name] [--allow_missing] [--custom_index CUSTOM_INDEX] [--max_workers MAX_WORKERS]

options:
  -h, --help            show this help message and exit
  --eval_dir eval_dir, -f eval_dir
                        Path to eval
  --serial, -s          Whether to disable parallel eval over systems
  --method_name method_name, -m method_name, -n method_name
                        Optional name for output csv
  --allow_missing, -a   Whether to allow missing systems for a given pinder-set + monomer
  --custom_index CUSTOM_INDEX, -c CUSTOM_INDEX
                        Optional local filepath or GCS uri to a custom index with non-pinder splits. Note: must still follow the pinder index schema and define test holdout sets, but does not need to share the same
                        split members.
  --max_workers MAX_WORKERS, -w MAX_WORKERS
                        Optional maximum number of processes to spawn in multiprocessing. Default is None (all available cores).
```

The expected format for the contents of `eval_dir` are shown below:
```
eval_dir_example/
└── some_method
    ├── 1an1__A1_P00761--1an1__B1_P80424
    │   ├── apo_decoys
    │   │   ├── model_1.pdb
    │   │   └── model_2.pdb
    │   ├── holo_decoys
    │   │   ├── model_1.pdb
    │   │   └── model_2.pdb
    │   └── predicted_decoys
    │       ├── model_1.pdb
    │       └── model_2.pdb
    └── 1b8m__A1_P23560--1b8m__B1_P34130
        ├── holo_decoys
        │   ├── model_1.pdb
        │   └── model_2.pdb
        └── predicted_decoys
            ├── model_1.pdb
            └── model_2.pdb
```

The eval directory should contain one or more methods to evaluate as sub-directories.

Each method sub-directory should contains sub-directories that are named by pinder system ID.

Inside of each pinder system sub-directory, you should have three subdirectories:
* `holo_decoys` (predictions that were made using holo monomers)
* `apo_decoys` (predictions made using apo monomers)
* `predicted_decoys` (predictions made using predicted, e.g. AF2, monomers)

You can have any number of decoys in each directory; however, the decoys should be named in a way that the prediction rank can be extracted. In the above example, the decoys are named using a `model_<rank>.pdb` convention. Other names for decoy models are accepted, so long as they can match the regex pattern used in `pinder.eval.dockq.MethodMetrics`: `r"\d+(?=\D*$)"`

Each model decoy should have exactly two chains: {R, L} for {Receptor, Ligand}, respectively.


⚠️ **Note: in order to make a fair comparison of methods across complete test sets, if a method is missing predictions for a system, the following metrics are used as a penalty**

```python

{
    "iRMS": 100.0,
    "LRMS": 100.0,
    "Fnat": 0.0,
    "DockQ": 0.0,
    "CAPRI": "Incorrect",
}
```

For more details on the implementations of the eval metrics, see the [eval docs](examples/eval/).


### 📨 pinder_create_submission

It is recommended to run through the `pinder_eval` script at least once to get familiar with the format and any common issues encountered with input validation. Once you are ready to submit your method to the leaderboard, use the `pinder_create_submission` CLI script to create a single archive to upload:

```
pinder_create_submission --help

usage: pinder_create_submission [-h] --eval_dir eval_dir [--submission_name submission_name]

optional arguments:
  -h, --help            show this help message and exit
  --eval_dir eval_dir, -f eval_dir
                        Path to eval
  --submission_name submission_name, -s submission_name, -n submission_name
                        Optional name for submission

```

This will create a single archive which would be uploaded to a google drive that will be configured shortly.

Leaderboards will be generated based on the valid submissions received.

For more details on leaderboard generation, see the [Quarto dashboard](examples/eval/leaderboard/pinder-eval.qmd) and the [MethodMetrics](src/pinder-eval/pinder/eval/dockq/method.py) implementation.


## 4. 🧪 Training set

We provide a ready-to-use, large training set, `PINDER-Train` with **2,456,152** pairs, consisting of 1,560,682 bound structures, 274,194 structures with at least one paired apo structure and 621,276 pairs with at least one paired AFDB structures. These can be combined to yield up to 25,130,994 unique training examples.
They are clustered by __interface__ similarity via FoldSeek and deleaked by structure and interface similarity against the `PINDER-XL` (and thus against all others) and validation set, `PINDER-Val`.

`PINDER-Val` is included as a redundancy removed validation set of 1,958 holo structures from 1,958 clusters, prepared in identical fashion and distribution to the test set, including 342 apo paired structures and 1,789 AFDB structures, filtered with the same quality criteria as the test set to allow for representative monitoring of training performance.

See [Dataset Generation](#-dataset-generation) for details on how the dataset was generated.

## 5. 📦 Dataloader

We provide a standardized pytorch geometric dataloader (and are happy to provide more if there are feature requests) for easy loading of datasets.

All dataloaders are based on iterators over `PinderSystem`, a core abstraction which provides the collection of structural data associated with an entry in the pinder database.

The `PinderSystem` exposes the following structures:
* Ground-truth crystal structure
* Holo receptor and ligand
* Apo receptor and ligand (where available)
* Predicted receptor and ligand (currently from alphafold; where available)

**Note: all monomers follow the chain naming convention of R, L for receptor and ligand, respectively.**

Each structure is defined by the `Structure` abstraction. See the [example notebook](examples/pinder-system.ipynb) for more details.


We provide the following features:

| Feature                                                                                                                                                                                                                        | Abstraction                                                          | Example                                             |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:----------------------------------------------------|
| Get collection of monomers associated with a pinder entry using `PinderSystem`                                                                                                                                                 | [PinderSystem](src/pinder-core/pinder/core/index/system.py)          | [pinder-system.ipynb](examples/pinder-system.ipynb) |
| Classify system difficulty based on degree of conformational shift in unbound and bound using `PinderSystem`                                                                                                                   | [PinderSystem](src/pinder-core/pinder/core/index/system.py)          | [pinder-system.ipynb](examples/pinder-system.ipynb) |
| Get various structural features like coordinates, residues, atoms and sequence and structural utilities using the `Structure` abstraction. All of the monomers in the `PinderSystem` object are themselves `Structure` objects | [Structure](src/pinder-core/pinder/core/loader/structure.py)         | [pinder-system.ipynb](examples/pinder-system.ipynb) |
| Filter datasets to construct data mixes using `PinderFilterBase`                                                                                                                                                               | [PinderFilterBase](src/pinder-core/pinder/core/loader/filters.py)    | [pinder-loader.ipynb](examples/pinder-loader.ipynb) |
| Filter datasets to construct data mixes with specific monomers or monomers that satisfy specific filter criteria using `PinderFilterSubBase`                                                                                   | [PinderFilterSubBase](src/pinder-core/pinder/core/loader/filters.py) | [pinder-loader.ipynb](examples/pinder-loader.ipynb) |
| Construct generator for getting specific data mixes and applying collection of filters through `PinderLoader`                                                                                                                  | [PinderLoader](src/pinder-core/pinder/core/loader/loader.py)         | [pinder-loader.ipynb](examples/pinder-loader.ipynb) |
| Load datasets as pytorch geometric graph datasets using `PPIDataset`                                                                                                                                                           | [PPIDataset](src/pinder-core/pinder/core/loader/dataset.py)          | [pinder-loader.ipynb](examples/pinder-loader.ipynb) |
| Create standard pytorch dataloaders using `get_geo_loader` with `PPIDataset` as input                                                                                                                                          | [get_geo_loader](src/pinder-core/pinder/core/loader/dataset.py)      | [pinder-loader.ipynb](examples/pinder-loader.ipynb) |
| Transform individual structures before use in downstream tasks using `TransformBase`                                                                                                                                           | [TransformBase](src/pinder-core/pinder/core/loader/transforms.py)    | [examples](examples/README.md#transforms)                   |


...

We are open to feature requests to add further functionality.


### Pytorch-geometric dataloader

A standardized pytorch-geometric dataloader to load subsets of the dataset for training and validation is provided.

Pinder provides a [torch_geometric.data.Dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset) sub-class, [PPIDataset](src/pinder-core/pinder/core/loader/dataset.py), which is used to create a graph dataset.

The dataset class provides an interface for processing the `PinderSystem` object into `HeteroData` objects that are written to disk.

It can be used as follows:

```python
from pinder.core import get_pinder_location, get_geo_loader, PPIDataset
from pinder.core.loader import filters
from pinder.core.loader.geodata import NodeRepresentation


nodes = {
  NodeRepresentation("atom"), NodeRepresentation("residue")
}

train_dataset = PPIDataset(
    node_types=nodes,
    split="train",
    monomer1="holo_receptor",
    monomer2="holo_ligand",
    base_filters: list[filters.PinderFilterBase] = [],
    sub_filters: list[filters.PinderFilterSubBase] = [],
    root: Path = get_pinder_location(),
    transform: Callable[[PinderSystem], PinderSystem] | None = None,
    pre_transform: Callable[[PinderSystem], PinderSystem] | None = None,
    pre_filter: Callable[[PinderSystem], PinderSystem | bool] | None = None,
    limit_by: int | None = None,
    force_reload: bool = False,
    filenames_dir: Path | str | None = None,
    repeat: int = 1,
    use_cache: bool = False,
    ids: list[str] | None = None,
    add_edges: bool = True,
    k: int = 10,
)

loader = get_geo_loader(train_dataset)
```

**Note: this is only one example of a featurizer that illustrates how to construct a hetero graph from a `PinderSystem` object.**

We welcome and encourage contributions of additional featurizers. To implement additional featurizers, please see the [PairedPDB](src/pinder-core/pinder/core/loader/geodata.py) implementation. New featurizers should implement a way to convert `Structure` instances belonging to `PinderSystem`'s into the respective pytorch or pytorch-geometric data objects.

For more detailed usage examples, including how to use the underlying loader without torch-geometric, see the [example notebook](examples/pinder-loader.ipynb).


## 6. ℹ️ Filters & Annonations

A core philosophy behind pinder is to provide a large, unfiltered training dataset to derive data mixes for evaluating the impact of different data selection strategies. To that end, we provide extensive tooling for leveraging annotations in filters.

A large set of quality control annotations including interface cluster, resolution, interfacial gaps, planarity, elongation, and more can be accessed via the `PinderSystem` object or directly in data frames.

We also provide the effective MSA Depth ($N_{eff}$) calculated for each of the test members in `PINDER-XL/S/AF2` to allow accurate performance assessment by evolutionary information.

Each `PinderSystem` object has an `.entry` and `.metadata` property, which expose all the primary annotations in the index and detailed metadata, respectively.

For detailed schemas of these properties, see the `IndexEntry` and `MetadataEntry` objects. Their fields are shown below for reference:

**IndexEntry**

| Field             | Type    | Description                                                                                                                                                                                                                                                                                                    |
|:------------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| split             | string  | The type of data split (e.g., 'train', 'test').                                                                                                                                                                                                                                                                |
| id                | string  | The unique identifier for the dataset entry.                                                                                                                                                                                                                                                                   |
| pdb_id            | string  | The PDB identifier associated with the entry.                                                                                                                                                                                                                                                                  |
| cluster_id        | string  | The cluster identifier associated with the entry.                                                                                                                                                                                                                                                              |
| cluster_id_R      | string  | The cluster identifier associated with receptor dimer body.                                                                                                                                                                                                                                                    |
| cluster_id_L      | string  | The cluster identifier associated with ligand dimer body.                                                                                                                                                                                                                                                      |
| pinder_s          | boolean | Flag indicating if the entry is part of the Pinder-S dataset.                                                                                                                                                                                                                                                  |
| pinder_xl         | boolean | Flag indicating if the entry is part of the Pinder-XL dataset.                                                                                                                                                                                                                                                 |
| pinder_af2        | boolean | Flag indicating if the entry is part of the Pinder-AF2 dataset.                                                                                                                                                                                                                                                |
| uniprot_R         | string  | The UniProt identifier for the receptor protein.                                                                                                                                                                                                                                                               |
| uniprot_L         | string  | The UniProt identifier for the ligand protein.                                                                                                                                                                                                                                                                 |
| holo_R_pdb        | string  | The PDB identifier for the holo form of the receptor protein.                                                                                                                                                                                                                                                  |
| holo_L_pdb        | string  | The PDB identifier for the holo form of the ligand protein.                                                                                                                                                                                                                                                    |
| predicted_R_pdb   | string  | The PDB identifier for the predicted structure of the receptor protein.                                                                                                                                                                                                                                        |
| predicted_L_pdb   | string  | The PDB identifier for the predicted structure of the ligand protein.                                                                                                                                                                                                                                          |
| apo_R_pdb         | string  | The PDB identifier for the apo form of the receptor protein.                                                                                                                                                                                                                                                   |
| apo_L_pdb         | string  | The PDB identifier for the apo form of the ligand protein.                                                                                                                                                                                                                                                     |
| apo_R_pdbs        | string  | The PDB identifiers for the apo forms of the receptor protein.                                                                                                                                                                                                                                                 |
| apo_L_pdbs        | string  | The PDB identifiers for the apo forms of the ligand protein.                                                                                                                                                                                                                                                   |
| holo_R            | boolean | Flag indicating if the holo form of the receptor protein is available.                                                                                                                                                                                                                                         |
| holo_L            | boolean | Flag indicating if the holo form of the ligand protein is available.                                                                                                                                                                                                                                           |
| predicted_R       | boolean | Flag indicating if the predicted structure of the receptor protein is available.                                                                                                                                                                                                                               |
| predicted_L       | boolean | Flag indicating if the predicted structure of the ligand protein is available.                                                                                                                                                                                                                                 |
| apo_R             | boolean | Flag indicating if the apo form of the receptor protein is available.                                                                                                                                                                                                                                          |
| apo_L             | boolean | Flag indicating if the apo form of the ligand protein is available.                                                                                                                                                                                                                                            |
| apo_R_quality     | string  | Classification of apo receptor pairing quality. Can be `high, low, ''`. All test and val are labeled high. Train split is broken into `high` and `low`, depending on whether the pairing was produced with a low-confidence quality/eval metrics or `high` if the same metrics were used as for train and val. |
| apo_L_quality     | string  | Classification of apo ligand pairing quality. Can be `high, low, ''`. All test and val are labeled high. Train split is broken into `high` and `low`, depending on whether the pairing was produced with a low-confidence quality/eval metrics or `high` if the same metrics were used as for train and val.   |
| chain1_neff       | number  | The Neff value for the first chain in the protein complex.                                                                                                                                                                                                                                                     |
| chain2_neff       | number  | The Neff value for the second chain in the protein complex.                                                                                                                                                                                                                                                    |
| chain_R           | string  | The chain identifier for the receptor protein.                                                                                                                                                                                                                                                                 |
| chain_L           | string  | The chain identifier for the ligand protein.                                                                                                                                                                                                                                                                   |
| contains_antibody | boolean | Flag indicating if the protein complex contains an antibody as per SAbDab.                                                                                                                                                                                                                                     |
| contains_antigen  | boolean | Flag indicating if the protein complex contains an antigen as per SAbDab.                                                                                                                                                                                                                                      |
| contains_enzyme   | boolean | Flag indicating if the protein complex contains an enzyme as per EC ID number.                                                                                                                                                                                                                                 |                                                                                       |


**MetadataEntry**

| Field                         | Type    | Description                                                                                                                                                                                                                     |
|:------------------------------|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id                            | string  | The unique identifier for the PINDER entry. It follows the convention `<Receptor>--<Ligand>`, where `<Receptor>` is `<pdbid>__<chain_1>_<uniprotid>` and                                                                        |
| entry_id                      | string  | The RCSB entry identifier associated with the PINDER entry.                                                                                                                                                                     |
| method                        | string  | The experimental method for structure determination (XRAY, CRYO-EM, etc.).                                                                                                                                                      |
| date                          | string  | Date of deposition into RCSB PDB.                                                                                                                                                                                               |
| release_date                  | string  | Date of initial public release in RCSB PDB.                                                                                                                                                                                     |
| resolution                    | number  | The resolution of the experimental structure.                                                                                                                                                                                   |
| label                         | string  | Classification of the interface as likely to be biologically-relevant or a crystal contact, annotated using PRODIGY-cryst.                                                                                                      |
| probability                   | number  | Probability that the protein complex is a true biological complex.                                                                                                                                                              |
| chain1_id                     | string  | The Receptor chain identifier associated with the dimer entry. Should all be chain 'R'.                                                                                                                                         |
| chain2_id                     | string  | The Ligand chain identifier associated with the dimer entry. Should all be chain 'L'.                                                                                                                                           |
| assembly                      | integer | Which bioassembly is used to derive the structure. 1, 2, 3 means first, second, and third assembly, respectively.                                                                                                               |
| assembly_details              | string  | How the bioassembly information was derived. Is it author-defined or from another source.                                                                                                                                       |
| oligomeric_details            | string  | Description of the oligomeric state of the protein complex.                                                                                                                                                                     |
| oligomeric_count              | integer | The oligomeric count associated with the dataset entry.                                                                                                                                                                         |
| biol_details                  | string  | The biological assembly details associated with the dataset entry.                                                                                                                                                              |
| complex_type                  | string  | The type of the complex in the dataset entry (homomer or heteromer).                                                                                                                                                            |
| chain_1                       | string  | New chain id generated post-bioassembly generation, to reflect the asym_id of the bioassembly and also to ensure that there is no collision of chain ids, for example in homooligomers (receptor chain).                        |
| asym_id_1                     | string  | The first asymmetric identifier (author chain ID)                                                                                                                                                                               |
| chain_2                       | string  | New chain id generated post-bioassembly generation, to reflect the asym_id of the bioassembly and also to ensure that there is no collision of chain ids, for example in homooligomers (ligand chain).                          |
| asym_id_2                     | string  | The second asymmetric identifier (author chain ID)                                                                                                                                                                              |
| length1                       | integer | The number of amino acids in the first (receptor) chain.                                                                                                                                                                        |
| length2                       | integer | The number of amino acids in the second (ligand) chain.                                                                                                                                                                         |
| length_resolved_1             | integer | The structurally resolved (CA) length of the first (receptor) chain in amino acids.                                                                                                                                             |
| length_resolved_2             | integer | The structurally resolved (CA) length of the second (ligand) chain in amino acids.                                                                                                                                              |
| number_of_components_1        | integer | The number of connected components in the first (receptor) chain (contiguous structural fragments)                                                                                                                              |
| number_of_components_2        | integer | The number of connected components in the second (receptor) chain (contiguous structural fragments)                                                                                                                             |
| link_density                  | number  | Density of contacts at the interface as reported by PRODIGY-cryst. Interfacial link density is defined as the number of interfacial contacts normalized by the maximum possible number of pairwise contacts for that interface. |
| planarity                     | number  | Defined as the deviation of interfacial Cα atoms from the fitted plane. This interface characteristic quantifies interfacial shape complementarity.                                                                             |
| max_var_1                     | number  | The maximum variance of coordinates projected onto the largest principal component.                                                                                                                                             |
| max_var_2                     | number  | The maximum variance of coordinates projected onto the largest principal component.                                                                                                                                             |
| num_atom_types                | integer | Number of unique atom types in structure. This is an important annotation to identify complexes with only Cα or backbone atoms.                                                                                                 |
| n_residue_pairs               | integer | The number of residue pairs at the interface.                                                                                                                                                                                   |
| n_residues                    | integer | The number of residues at the interface.                                                                                                                                                                                        |
| buried_sasa                   | number  | The buried solvent accessible surface area upon complex formation.                                                                                                                                                              |
| intermolecular_contacts       | integer | The total number of intermolecular contacts (pair residues with any atom within a 5Å distance cutoff) at the interface.                                                                                                         |
| charged_charged_contacts      | integer | Denotes intermolecular contacts between any of the charged amino acids (E, D, H, K).                                                                                                                                            |
| charged_polar_contacts        | integer | Denotes intermolecular contacts between charged amino acids (E, D, H, K, R) and polar amino acids (N, Q, S, T).                                                                                                                 |
| charged_apolar_contacts       | integer | Denotes intermolecular contacts between charged amino acids (E, D, H, K) and apolar amino acids (A, C, G, F, I, M, L, P, W, V, Y).                                                                                              |
| polar_polar_contacts          | integer | Denotes intermolecular contacts between any of the charged amino acids (N, Q, S, T).                                                                                                                                            |
| apolar_polar_contacts         | integer | Denotes intermolecular contacts between apolar amino acids (A, C, G,F, I, M, L, P, W, V, Y) and polar amino acids (N, Q, S, T).                                                                                                 |
| apolar_apolar_contacts        | integer | Denotes intermolecular contacts between any of the charged amino acids (A, C, G, F, I, M, L, P, W, V, Y).                                                                                                                       |
| interface_atom_gaps_4A        | integer | Number of interface atoms within a 4Å radius of a residue gap.                                                                                                                                                                  |
| missing_interface_residues_4A | integer | Number of interface residues within a 4Å radius of a residue gap.                                                                                                                                                               |
| interface_atom_gaps_8A        | integer | Number of interface atoms within an 8Å radius of a residue gap.                                                                                                                                                                 |
| missing_interface_residues_8A | integer | Number of interface residues within an 8Å radius of a residue gap.                                                                                                                                                              |
| entity_id_R                   | integer | The RCSB PDB `entity_id` corresponding to the receptor dimer chain.                                                                                                                                                             |
| entity_id_L                   | integer | The RCSB PDB `entity_id` corresponding to the ligand dimer chain.                                                                                                                                                               |
| pdb_strand_id_R               | string  | The RCSB PDB `pdb_strand_id` (author chain) corresponding to the receptor dimer chain.                                                                                                                                          |
| pdb_strand_id_L               | string  | The RCSB PDB `pdb_strand_id` (author chain) corresponding to the ligand dimer chain.                                                                                                                                            |
| ECOD_names_R                  | string  | The RCSB-derived ECOD domain protein family name(s) corresponding to the receptor dimer chain. If multiple ECOD domain annotations                                                                                              |
| ECOD_names_L                  | string  | The RCSB-derived ECOD domain protein family name(s) corresponding to the ligand dimer chain. If multiple ECOD domain annotations were found, the domains are delimited with a comma.                                            |

These annotations can be used during loading either by filtering the data frames or by implementing filters and transforms.
For example, to filter on some metadata fields, you can construct a series of `FilterMetadataFields` filters:

```python

from pinder.core import PinderLoader
from pinder.core.loader import filters

base_filters = [
    filters.FilterByMissingHolo(),
    filters.FilterSubByContacts(min_contacts=5, radius=10.0, calpha_only=True),
    filters.FilterByHoloElongation(max_var_contribution=0.92),
    filters.FilterDetachedHolo(radius=12, max_components=2),

    filters.FilterMetadataFields(contains_antibody=('', False)),
    # You can also combine multiple fields in the FilterMetadataFields:
    filters.FilterMetadataFields(
      contains_enzyme=('is not', True),
      resolution=('<=', 2.75),
      method=('!=', 'X-RAY DIFFRACTION'),
    ),
]

# These operate on individual monomers
sub_filters = [
    filters.FilterSubByAtomTypes(min_atom_types=4),
    filters.FilterByHoloOverlap(min_overlap=5),
    filters.FilterByHoloSeqIdentity(min_sequence_identity=0.8),
    filters.FilterSubLengths(min_length=0, max_length=1000),
    filters.FilterSubRmsds(rmsd_cutoff=7.5),
    filters.FilterByElongation(max_var_contribution=0.92),
    filters.FilterDetachedSub(radius=12, max_components=2),
]
loader = PinderLoader(
    base_filters = base_filters,
    sub_filters = sub_filters
)

loader.load_split("test", "pinder_af2")
for dimer in loader.dimers:
    # do something
    pass
```

These are documented in greater detail in the [examples](examples/) section.


## 7. 📡 Future work

While `pinder` makes significant strides, several limitations highlight areas for future improvement. Most evidently, `pinder` is currently focusing on biological dimers. As more methods expand beyond dimers, such as via co-folding approaches, `pinder` will be generalized to higher-order oligomers. Additionally, there are a few smaller methodological limitations - for instance, the reliance on single reference conformations and the inherent bias towards homodimers in the dataset can impact the accuracy and generalizability of the models.

Further, improvements in *apo* pairing and the integration of more advanced tools, such as `iAlign`, into the alignment methodology could enhance the dataset's precision. Addressing these limitations could lead to even larger datasets, better performance and evaluation in future iterations of `pinder`. We provide a more detailed discussion of the limitations of the `pinder` dataset and methodology in [limitations](limitations.md). Below we summarize some key areas of future work:

[ ] Expansion to higher-order oligomers
[ ] Homologous *apo* pairing via Foldseek and MMseqs2 monomer matching
[ ] Rosetta-relaxed unbound structures & evaluation set for all structures in the dataset
[ ] A complete evaluation harness for reference-free metrics to evaluate the quality of the predicted structures, such as VoroMQA, PISA, and more
[ ] Confirmed negative pairs
[ ] Addition of an antibody-focused benchmark test set `pinder-ab`
[ ] Contact-conditioned benchmarks
[ ] Additional information for multimeric training examples (e.g. restraints)
[ ] Improved abstractions for pytorch-lightning data loaders


# 👨‍💻 Code organization

This code is split into 4 subpackages

- `pinder-core`: core data structures for interacting with and loading the dataset. includes a pytorch dataloader
- `pinder-data`: core code for generating the dataset, starting with downloading from the RCSB NextGen rsync server.
- `pinder-eval`: evaluation harness for the dataset that takes as an input predicted and ground truth structures in a pre-determined folder structure and returns a leaderboard-ready set of entries
- `pinder-methods`: implementations of the methods in the leaderboard that leverage pinder-primitives for training & running

# 💽 Dataset Generation

The above datasets was generated using the following steps:

## 🚪 Input

The RCSB NextGen database (as of 01.29.2024) was used as the starting point. All mmCIF files were obtained and representative biological assemblies were generated.

- **PDB PPIs**: Protein-Protein Interactions (PPIs) were detected as all pairs of chains with a backbone atom in contact at a 10Å threshold.
- **PDB Monomers for apo structures**: All monomeric PDB entries with the same UniProt ID as a monomer in the dimer PPI entries were aligned (using the UniProt numbering) to the corresponding PPI entry. A suite of evaluation metrics was calculated and only validated pairings were kept. For each dimer monomer, a single apo monomer was chosen as the canonical pair based on a normalized score derived from the evaluation metrics. The rest are made available as alternate apo pairings.
- **AFDB Monomers for af2 structures**: AFDB entries with the same UniProt ID as PPI entries were aligned (using UniProt numbering) to the corresponding PPI entry.

## ℹ️ Annotation

Annotations were obtained from the RCSB NextGen database. The following annotations are included:

1. Oligomeric state of the protein complex (homodimer, heterodimer, oligomer or higher-order
complexes)
2. Structure determination method (X-Ray, CryoEM, NMR)
3. Resolution
4. Interfacial gaps, defined as structurally-unresolved segments on PPI interfaces
5. Number of distinct atom types. Many earlier Cryo-EM structures contain only a few atom-types
such as only Cα or backbone atoms
6. Whether the interface is likely to be a physiological or crystal contact, annotated using Prodigy
7. Structural elongation, defined as the maximum variance of coordinates projected onto the largest
principal component. This allows detection of long end-to-end stacked complexes, likely to be
repetitive with small interfaces
8. Planarity, defined as deviation of interfacial Cα atoms from the fitted plane. This interface
characteristic quantifies interfacial shape complementarity. Transient complexes have smaller
and more planar interfaces than permanent and structural scaffold complexes
9. Number of components, defined as the number of connected components of a 10Å Cα radius
graph. This allows detection of structurally discontinuous domains
10. Intermolecular contacts (labeled as polar or apolar)

## 👥 Clustering

The clustering works as follows
- We first define all possible interacting pairs as holo systems by taking chain pair with any residues within a 10Å backbone atom distance threshold between the interacting chains
- All-vs-all structural alignments of complete chains were performed using FoldSeek. Note that foldseek uses both sequence (blosum matrix) and structure (3di matrix) to define similar pairs
- We start by construct a graph with chains as nodes
- An edge is then added between any two nodes with over 50% foldseek-alignment coverage of the interface residues (as defined above)
- This will connect any two chains where a substantial part of the interface is similar in either sequence or structure
- Community clustering via asynchronous label propagation was then performed on this graph to obtain interface clusters. Clusters are used in three ways:
    - Non-redundant sampling and weighing scheme during training
    - Non-redundant test/val selection by selecting test as cluster representatives
    - Deleaking by removal of other cluster members after a member of the cluster is chosen as test/val (Note: deleaking algorithm uses further steps to ensure no-leakage between test/val and train)
- From the chain-graph clusters we create paired-interface clusters paired-interface cluster of each PPI as $\{c_{a}, c_{b}\}$, where $c_a$ and $c_b$ are the interface cluster identifiers for the two interacting chains.

## 🧭 Test filters

Sampling for the test and validation sets was performed based on the following criteria:

- Physiological contact (from PRODIGY-cryst)
- Dimers (to guarantee full bioassembly available during inference)
- `X-RAY DIFFRACTION` experimental method
- Resolution $\leq$ 3.5Å
- Minimum individual chain length $\geq$ 40 residues
- Elements $\geq$ 3
- Interface atom gaps at 4Å threshold = 0
- Maximum variance $\leq$ 0.98
- Single component

All clusters where any one member passed the above criteria were kept. These filters resulted in 32,775 PPIs (we call these proto-test) in 5,047 clusters (used for val/**PINDER-XL**).

PPIs which passed the criteria in these sampled clusters were deleaked by an additional transitive neighbor search, where any system that had a transitive hit within a depth of 2 in the foldseek graph, but a different cluster ID, was removed. Leaky systems were kept, but assigned a split label of `invalid`. Eligible systems were ranked by heterodimer vs. homodimer (heterodimer preferred), whether they pass the `PINDER-AF2` criteria, and availability of apo and AFDB paired structures. 1 PPI was sampled from each cluster based on the member ranked by the criteria.


## 🔪 Split

The dataset was split as follows:

- **PINDER-XL**: Sampling from 1,955 clusters resulted in 1,955 members.
- **PINDER-Val**: Sampling from 1,958 clusters resulted in 1,958 members.
- **PINDER-Train**: The rest of the clusters (42,220) resulted in 1,560,682 members

From the PDB monomer alignments, we obtained a total of 44,330 unique apo structures, corresponding to 41,630 receptor and 36,910 ligand monomers.
This corresponds to 274,194 pinder dimers in train, 441 in val and 436 in **PINDER-XL** with at least one matched apo structure.

From the AFDB monomer alignments, we obtained a total of 42,827 unique AFDB structures, corresponding to 37,095 receptor and 38,801 ligand monomers.
This corresponds to 621,276 pinder dimers in train, 1,817 in val and 1,775 in **PINDER-XL** with at least one matched AFDB monomer structure.

These were assigned to the respective chain pairs to yield the numbers from above tables.

**PINDER-S** is a subset of **PINDER-XL**, consisting of 250 clusters (188 heterodimer and 62 homodimers) sampled for diverse Uniprot and PFAM annotations, 93 of which have apo paired structures (143 have at least one apo monomer) and all of which have paired AFDB structures, to evaluate methods for which sampling from the full set is too slow.

## 🗞️ AF2mm

Clusters which contain only members released after 10.01.2021 (the AlphaFold-2MM cutoff date) were separated into 675 clusters. From these 675 members, we further de-leaked against any similar interfaces found to any other entry released before the cutoff date as determined by `iAlign`. The members which have low or no similarity to AlphaFold2-Multimer training set (180) were assgined to the `PINDER-AF2` set. Those members in the `PINDER-AF2` set are guaranteed to be structurally distinct from AF2-MM 2.3 training data, while the remaining members are only guaranteed to come from entries released after the cutoff date (time-split).


# 📰 Updates & Versioning

Dataset and code are versioned independently.

The dataset is expected to be updated with at at maximum monthly release cycles frequency via `year-month` as subfolders in `pinder`. The current version is `2024-02`

There are 2 "types" of updates:

- minor changes in the index, or addition or change in structures that can be assigned to train without reclustering and adding leakage and thus not invalidating the test set and **not requiring re-evaluation of the leaderboard methods**
- major addition of structures that require re-clustering and re-assigning of structures to train, validation and test thus **invalidating the leaderboard**

Major changes may happen with at maximum annual frequency, and will be clearly announced. Methods can choose *not* to update and continue using previous versions of the dataset

The code is versioned using [semantic versioning](https://semver.org/) and updated regularly and contains integration test to avoid invalidating any results

# Examples & documentation

Package documentation, including API documentation, [example notebooks](examples/), and supplementary guides, are made available.

To view the latest documentation, you can checkout the [gh-pages](https://github.com/pinder-org/pinder/tree/gh-pages) branch and open the [index.html](https://github.com/pinder-org/pinder/blob/gh-pages/index.html) file in your browser.

Alternatively, you can build the package documentation locally after installing optional documentation dependencies:

```
pip install '.[docs]'
cd docs/
./build.sh --open
```

For a list of frequently asked questions, check the [FAQ section](faq.md).

# Dev guide

## Dev mode install

### Clone the repo

For http:
```bash
git clone https://github.com/pinder-org/pinder.git
```

Or ssh:
```bash
git clone git@github.com:pinder-org/pinder.git
```

### Initialize a conda env

```bash
cd pinder
mamba create --name pinder python=3.10
mamba activate pinder
```

### Install the desired pinder subpackages in dev mode

```bash
pip install -e '.[dev]'
```

### (Optional) install pre-commit hooks

These will ensure that code passes the linter before being committed.

    pre-commit install

## Test suite

    tox

We lint with ruff. See `tox.ini` and `.pre-commit-config.yaml` for details.

## Debugging

In order to change log levels, set the `LOG_LEVEL` environment variable. For example:

    export LOG_LEVEL=DEBUG

# Contributing

This is a community effort and as such we highly encourage contributions.
