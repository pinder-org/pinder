# PINDER Methods Runner

Includes entry point for running `training` and `inference or prediction`.

Running the command:
```
pinder_methods predict -m <method> -s <test set> -l <list of PINDER ids> -c `config` -r `results_dir`
```

will run inference/prediction with `method` either on a subset of index `test set`, such as `pinder-s` OR on a list of PINDER IDs.
If the list of IDS is provided it takes priority over the subset.

Any additional parameters to the methods must be provided in form of a configuration file `config` for better reproducibility. Configuration file will be copied to the `results_dir`

The results will be generated in `results_dir` in the format compatible with `pinder_eval` runner.

```
results_dir/
└── method
    ├── pinder_id
    │   └── ...
    ├── ...
    ...
```


# Building docker images for physics-based docking methods

For convenience, we provide [dockerfiles](../../../../docker/) for building images with pinder and its dependencies pre-installed, along with the docking software. We do not provide the source software.

In order to build the images, you will need to obtain the source software from the respective methods we support:
* [FRODOCK](https://chaconlab.org/modeling/frodock/frodock-donwload)
* [HDOCKlite](http://huanglab.phys.hust.edu.cn/software/hdocklite/)
* [PatchDock](http://bioinfo3d.cs.tau.ac.il/PatchDock/download.html)

Once downloaded, you need to provide a local path to the software compressed in `.tar.gz` format. See [expected software archive contents](#expected-software-archive-contents) for a detailed tree of the expected contents of the archive.

When you are ready to build, you can run:
```
docker build --build-arg FRODOCK_SOURCE_ARCHIVE=./frodock.tar.gz -t pinder-frodock -f docker/frodock/Dockerfile.frodock .
docker build --build-arg HDOCK_SOURCE_ARCHIVE=./hdock.tar.gz -t pinder-hdock -f docker/hdock/Dockerfile.hdock .
docker build --build-arg PATCHDOCK_SOURCE_ARCHIVE=./patchdock.tar.gz -t pinder-patchdock -f docker/patchdock/Dockerfile.patchdock .
```

## Expected software archive contents
### FRODOCK
```
tar -xvf frodock.tar.gz

tree frodock3_linux64/ --filelimit 20

frodock3_linux64/
├── README
├── bin
│   ├── frodock
│   ├── frodock_gcc
│   ├── frodock_mpi_gcc
│   ├── frodockcheck
│   ├── frodockcheck_gcc
│   ├── frodockcluster
│   ├── frodockcluster_gcc
│   ├── frodockgrid
│   ├── frodockgrid_gcc
│   ├── frodockgrid_mpi_gcc
│   ├── frodockonstraints
│   ├── frodockonstraints_gcc
│   ├── frodockview
│   ├── frodockview_gcc
│   └── soap.bin
├── clean_frodock.sh
├── compile_frodock.sh
├── includes
│   ├── cmdl -> ../src/cmdl
│   ├── opt
│   │   ├── nlopt.h
│   │   └── nlopt.hpp
│   └── tclap -> ../src/tclap
├── lib
├── run_frodock.sh
├── run_test_1MLC.sh
├── run_test_1WEJ.sh
├── src  [24 entries exceeds filelimit, not opening dir]
├── test_1MLC
│   ├── 1MLC_l_b.pdb
│   ├── 1MLC_l_u.pdb
│   ├── 1MLC_l_u_ASA.pdb
│   ├── 1MLC_r_fav.pdb
│   ├── 1MLC_r_u.pdb
│   ├── 1MLC_r_u_ASA.pdb
│   └── consts.txt
└── test_1WEJ
    ├── 1WEJ_l_b.pdb
    ├── 1WEJ_l_u.pdb
    ├── 1WEJ_l_u_ASA.pdb
    ├── 1WEJ_l_u_fitted.pdb
    ├── 1WEJ_r_b.pdb
    ├── 1WEJ_r_u.pdb
    └── consts.txt

9 directories, 37 files

```

### HDOCK
```
tar -xvf hdock.tar.gz

tree HDOCKlite/

HDOCKlite/
├── 1CGI_l_b.pdb
├── 1CGI_r_b.pdb
├── README
├── createpl
└── hdock

0 directories, 5 files
```

### PatchDock

```
tar -xvf patchdock.tar.gz

tree PatchDock/

PatchDock/
├── buildParams.pl
├── buildParamsXlinks.pl
├── cdr
│   ├── cdr.Linux
│   ├── igvh.pc
│   ├── igvkl.pc
│   └── igvll.pc
├── chem.lib
├── patch_dock.Linux
├── pdb_trans
└── transOutput.pl

1 directory, 10 files

```
