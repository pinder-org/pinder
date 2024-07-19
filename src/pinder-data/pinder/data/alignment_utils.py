from __future__ import annotations
import json
import multiprocessing
import pickle
from typing import Tuple, Set
from pathlib import Path
from concurrent import futures
from itertools import repeat
import math

import biotite.structure as struc
import numpy as np
from biotite.structure.atoms import AtomArray
from numpy.typing import NDArray
from dataclasses import dataclass
from tqdm import tqdm

from pinder.core.index.system import PinderSystem
from pinder.core.structure.contacts import pairwise_contacts
from pinder.core.structure.atoms import atom_array_from_pdb_file, backbone_mask
from pinder.core.utils import setup_logger
from pinder.core.utils.paths import empty_file
from pinder.data.system import get_dev_systems
from pinder.data.config import get_config_hash, ContactConfig
from pinder.data.foldseek_utils import FOLDSEEK_FIELDS


log = setup_logger(__name__)


@dataclass
class DomainInfo:
    pdb: str
    chain: str
    ecod_domain_id: str
    t_name: str


@dataclass
class Domain:
    pdb_id: str
    chain: str
    domain_id: str
    t_name: str
    pdb_from: int
    pdb_to: int

    @classmethod
    def from_domain_info(
        cls, domain_info: DomainInfo, start: int, end: int
    ) -> "Domain":
        pdb_id, chain, domain_id, domain, pdb_from, pdb_to = (
            domain_info.pdb,
            domain_info.chain,
            domain_info.ecod_domain_id,
            domain_info.t_name,
            start,
            end,
        )
        return cls(pdb_id, chain, domain_id, domain, int(pdb_from), int(pdb_to))


@dataclass
class Alignment:
    pdbid1: str
    pdbid2: str
    alntmscore: float
    qstart: int
    qend: int
    qlen: int
    tstart: int
    tend: int
    tlen: int
    alnlen: int

    @classmethod
    def from_line(cls, line: str) -> "Alignment":
        (
            pdbid1,
            pdbid2,
            alntmscore,
            qstart,
            qend,
            qlen,
            tstart,
            tend,
            tlen,
            alnlen,
        ) = line.split("\t")
        if alntmscore == "nan":
            log.warning(
                f"Alignment score for {pdbid1} {pdbid2} is nan! Using default value of 0.5...."
            )
            alntmscore = "0.5"
        return cls(
            pdbid1.replace(".pdb", ""),
            pdbid2.replace(".pdb", ""),
            float(alntmscore),
            int(qstart),
            int(qend),
            int(qlen),
            int(tstart),
            int(tend),
            int(tlen),
            int(alnlen),
        )

    @classmethod
    def from_foldseek_line(cls, line: str) -> "Alignment":
        fields = FOLDSEEK_FIELDS.format("alntmscore").split(",")
        assert len(fields) == len(line.split("\t"))
        field_name_map = {"query": "pdbid1", "target": "pdbid2"}
        aln_fields = cls.__dataclass_fields__
        aln_kwargs: dict[str, str | float | int] = {}
        for field, val in zip(fields, line.split("\t")):
            if field in aln_fields:
                type_str = str(aln_fields[field].type)
                aln_val = eval(type_str)(val)
                aln_kwargs[field] = aln_val
            elif field in field_name_map:
                aln_field_name = field_name_map[field]
                type_str = str(aln_fields[aln_field_name].type)
                aln_val = eval(type_str)(val)
                if isinstance(aln_val, str):
                    aln_val = aln_val.replace(".pdb", "")
                aln_kwargs[aln_field_name] = aln_val
            else:
                log.warning(f"Unrecognized/unused alignment field: {field}")

        score_val = float(aln_kwargs["alntmscore"])
        if math.isnan(score_val):
            log.warning(
                f"Alignment score for {aln_kwargs['pdbid1']} {aln_kwargs['pdbid2']} is nan! "
                "Using default value of 0.5...."
            )
            score_val = 0.5

        return cls(
            pdbid1=str(aln_kwargs["pdbid1"]),
            pdbid2=str(aln_kwargs["pdbid2"]),
            alntmscore=float(score_val),
            qstart=int(aln_kwargs["qstart"]),
            qend=int(aln_kwargs["qend"]),
            qlen=int(aln_kwargs["qlen"]),
            tstart=int(aln_kwargs["tstart"]),
            tend=int(aln_kwargs["tend"]),
            tlen=int(aln_kwargs["tlen"]),
            alnlen=int(aln_kwargs["alnlen"]),
        )

    @classmethod
    def from_ecod_info(cls, ecod_info_pair: tuple[Domain, Domain]) -> "Alignment":
        ecod_info1, ecod_info2 = ecod_info_pair
        pdbid1, pdbid2, alntmscore, qstart, qend, qlen, tstart, tend, tlen, alnlen = (
            ecod_info1.pdb_id,
            ecod_info2.pdb_id,
            1.0,
            ecod_info1.pdb_from,
            ecod_info1.pdb_to,
            int(ecod_info1.pdb_to - ecod_info1.pdb_from),
            ecod_info2.pdb_from,
            ecod_info2.pdb_to,
            int(ecod_info1.pdb_to - ecod_info1.pdb_from),
            int(ecod_info1.pdb_to - ecod_info1.pdb_from),
        )
        return cls(
            pdbid1,
            pdbid2,
            float(alntmscore),
            int(qstart),
            int(qend),
            int(qlen),
            int(tstart),
            int(tend),
            int(tlen),
            int(alnlen),
        )

    def indices1(self) -> set[int]:
        return set(range(self.qstart, self.qend))

    def indices2(self) -> set[int]:
        return set(range(self.tstart, self.tend))

    def flip_query_and_target(self) -> "Alignment":
        return Alignment(
            self.pdbid2,
            self.pdbid1,
            self.alntmscore,
            self.tstart,
            self.tend,
            self.tlen,
            self.qstart,
            self.qend,
            self.qlen,
            self.alnlen,
        )


@dataclass
class Interface:
    pdbid1: str
    pdbid2: str
    indices1: Set[int]
    indices2: Set[int]
    alignments1: Set[Tuple[str, float]]
    alignments2: Set[Tuple[str, float]]

    @classmethod
    def from_line(cls, line: str) -> "Interface":
        pdbid1, pdbid2, indices1, indices2 = line.split("\t")
        return cls(
            pdbid1,
            pdbid2,
            set([int(x) for x in indices1.split(",")]),
            set([int(x) for x in indices2.split(",")]),
            set(),
            set(),
        )

    @classmethod
    def from_system(cls, system: PinderSystem, radius: float) -> "Interface":
        pdbid_R, pdbid_L = system.entry.id.split("--")
        pdbid_R += "-R"
        pdbid_L += "-L"
        chainR_res_foldseek_idx, chainL_res_foldseek_idx = get_foldseek_contacts(
            system, radius=radius, backbone_definition="dockq", return_calpha_only=True
        )
        return cls(
            pdbid_R,
            pdbid_L,
            set(chainR_res_foldseek_idx),
            set(chainL_res_foldseek_idx),
            set(),
            set(),
        )

    @classmethod
    def from_contact_info(
        cls, contact_info: dict[str, str | int | float | bool]
    ) -> "Interface":
        assert isinstance(contact_info["R_residues"], str)
        assert isinstance(contact_info["L_residues"], str)
        R_res = {int(r) for r in contact_info["R_residues"].split(",")}
        L_res = {int(r) for r in contact_info["L_residues"].split(",")}
        assert isinstance(contact_info["interface_id1"], str)
        assert isinstance(contact_info["interface_id2"], str)
        id1 = contact_info["interface_id1"]
        id2 = contact_info["interface_id2"]
        interface = Interface(
            id1,
            id2,
            indices1=R_res,
            indices2=L_res,
            alignments1=set(),
            alignments2=set(),
        )
        return interface

    def flip_interface(self) -> "Interface":
        return Interface(
            self.pdbid2,
            self.pdbid1,
            self.indices2,
            self.indices1,
            self.alignments2,
            self.alignments1,
        )


def get_foldseek_contacts(
    dimer: PinderSystem,
    radius: float = 10.0,
    backbone_definition: str = "dockq",
    return_calpha_only: bool = True,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    contacts = dimer.native.get_contacts(
        heavy_only=True, backbone_only=True, radius=radius
    )
    r_contacts = {x[2] for x in contacts}
    l_contacts = {x[3] for x in contacts}

    arr_ori = dimer.native.atom_array.copy()
    bb_mask = backbone_mask(arr_ori, backbone_definition)

    R_arr = arr_ori[(arr_ori.chain_id == "R") & bb_mask].copy()

    L_arr = arr_ori[(arr_ori.chain_id == "L") & bb_mask].copy()

    R_renum = R_arr.copy()
    L_renum = L_arr.copy()
    R_renum = struc.renumber_res_ids(R_renum, start=1)
    L_renum = struc.renumber_res_ids(L_renum, start=1)

    R_map = {at1.res_id: at2.res_id for at1, at2 in zip(R_arr, R_renum)}
    L_map = {at1.res_id: at2.res_id for at1, at2 in zip(L_arr, L_renum)}
    if return_calpha_only:
        R_list = sorted(
            list(
                {
                    R_map[resi]
                    for resi in R_arr[np.isin(R_arr.res_id, list(r_contacts))].res_id
                }
            )
        )
        L_list = sorted(
            list(
                {
                    L_map[resi]
                    for resi in L_arr[np.isin(L_arr.res_id, list(l_contacts))].res_id
                }
            )
        )
        R_cont_filter = np.array(R_list)
        L_cont_filter = np.array(L_list)
    else:
        R_cont_filter = np.array(
            [
                R_map[resi]
                for resi in R_arr[np.isin(R_arr.res_id, list(r_contacts))].res_id
            ]
        )
        L_cont_filter = np.array(
            [
                L_map[resi]
                for resi in L_arr[np.isin(L_arr.res_id, list(l_contacts))].res_id
            ]
        )
    return R_cont_filter, L_cont_filter


def get_foldseek_numbering(arr: AtomArray) -> dict[int, int]:
    arr_renum = arr.copy()
    arr_renum = struc.renumber_res_ids(arr_renum, start=1)
    res_map = {at1.res_id: at2.res_id for at1, at2 in zip(arr, arr_renum)}
    return res_map


def get_foldseek_dimer_contacts(
    dimer_pdb: Path,
    contact_config: ContactConfig = ContactConfig(),
) -> dict[str, str | int | float | bool] | None:
    structure = atom_array_from_pdb_file(dimer_pdb, backend="fastpdb")
    res_conts = pairwise_contacts(
        structure,
        radius=contact_config.radius,
        heavy_only=contact_config.heavy_only,
        backbone_only=contact_config.backbone_only,
        chain1=["R"],
        chain2=["L"],
        atom_and_residue_level=False,
    )
    if not len(res_conts):
        log.error(f"Dimer {dimer_pdb.name} had zero contacts!")
        return None
    if len(res_conts):
        r_contacts = {x[2] for x in res_conts}
        l_contacts = {x[3] for x in res_conts}

    arr_ori = structure.copy()
    R_arr = arr_ori[(arr_ori.chain_id == "R")].copy()
    L_arr = arr_ori[(arr_ori.chain_id == "L")].copy()

    R_map = get_foldseek_numbering(R_arr)
    L_map = get_foldseek_numbering(L_arr)
    if contact_config.only_unique_resi:
        R_list = sorted(
            list(
                {
                    R_map[resi]
                    for resi in R_arr[np.isin(R_arr.res_id, list(r_contacts))].res_id
                }
            )
        )
        L_list = sorted(
            list(
                {
                    L_map[resi]
                    for resi in L_arr[np.isin(L_arr.res_id, list(l_contacts))].res_id
                }
            )
        )
        R_cont_filter = np.array(R_list)
        L_cont_filter = np.array(L_list)
    else:
        R_cont_filter = np.array(
            [
                R_map[resi]
                for resi in R_arr[np.isin(R_arr.res_id, list(r_contacts))].res_id
            ]
        )
        L_cont_filter = np.array(
            [
                L_map[resi]
                for resi in L_arr[np.isin(L_arr.res_id, list(l_contacts))].res_id
            ]
        )

    pdbid_R, pdbid_L = dimer_pdb.stem.split("--")
    pdbid_R += "-R"
    pdbid_L += "-L"
    interface_id1 = pdbid_R
    interface_id2 = pdbid_L

    R_residues = ",".join(map(str, R_cont_filter))
    L_residues = ",".join(map(str, L_cont_filter))
    contact_info: dict[str, str | float | int | bool] = {
        "id": dimer_pdb.stem,
        "interface_id1": pdbid_R,
        "interface_id2": pdbid_L,
        "R_residues": R_residues,
        "L_residues": L_residues,
        "radius": contact_config.radius,
        "R_length": len(R_cont_filter),
        "L_length": len(L_cont_filter),
        "heavy_only": contact_config.heavy_only,
        "backbone_only": contact_config.backbone_only,
        "only_unique_resi": contact_config.only_unique_resi,
    }
    return contact_info


def generate_dimer_foldseek_contacts(
    dimer_pdb: Path,
    contact_config: ContactConfig = ContactConfig(),
    use_cache: bool = True,
) -> tuple[dict[str, str | int | float | bool], Path] | None:
    contact_fp = dimer_pdb.parent / "foldseek_contacts"
    config_hash = get_config_hash(contact_config)
    config_dir = contact_fp / config_hash
    if not config_dir.is_dir():
        config_dir.mkdir(exist_ok=True, parents=True)
    contact_json = config_dir / f"{dimer_pdb.stem}.json"
    if use_cache and contact_json.is_file():
        return None
    contact_info = get_foldseek_dimer_contacts(dimer_pdb, contact_config)
    if isinstance(contact_info, dict):
        return contact_info, contact_json
    else:
        return None


def populate_foldseek_contacts(
    dimer_pdbs: list[Path],
    contact_config: ContactConfig = ContactConfig(),
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    """Process batch of Dimer PDBs to store contacts with different configurations for foldseek.

    Parameters:
    dimer_pdbs (list[Path]): List of dimer PDBs to get contacts for.
    use_cache (bool): Whether to skip generation of contacts if the contacts.json corresponding to the config hash exists.
    parallel (bool): Whether to populate entries in parallel.
    max_workers (int, optional): Limit number of parallel processes spawned to `max_workers`.
    """

    log.info(f"Getting foldseek contacts for {len(dimer_pdbs)} dimer PDBs")
    dimer_contacts: list[tuple[dict[str, str | int | float | bool], Path] | None] = []
    if len(dimer_pdbs) > 0:
        # process all files in parallel
        if parallel:
            # max_workers=param_n_cores
            try:
                with futures.ProcessPoolExecutor(
                    mp_context=multiprocessing.get_context("spawn"),
                    max_workers=max_workers,
                ) as exe:
                    dimer_contacts = list(
                        exe.map(
                            generate_dimer_foldseek_contacts,
                            dimer_pdbs,
                            repeat(contact_config),
                        )
                    )
            except Exception as e:
                log.error(f"Failed to calculate foldseek contacts: {e}")
        else:
            dimer_contacts = []
            for dimer_pdb in tqdm(dimer_pdbs):
                result = generate_dimer_foldseek_contacts(dimer_pdb, contact_config)
                dimer_contacts.append(result)

        for c in tqdm(dimer_contacts):
            if c is not None and len(c) == 2:
                cinfo, cjson = c
                with open(cjson, "w") as f:
                    json.dump(cinfo, f)


def safe_read_contact_json(
    contact_json: Path,
) -> dict[str, str | int | float | bool] | None:
    contact_info: dict[str, str | int | float | bool] | None = None
    if not empty_file(contact_json):
        try:
            with open(contact_json) as f:
                contact_info = json.load(f)
        except Exception as e:
            log.warning(
                f"JSON is not empty, but its not readable: {contact_json.name}, {e}"
            )
            contact_json.unlink()
    return contact_info


def load_contact_info(
    pdb_file: Path,
    contact_json: Path,
    config: ContactConfig,
) -> dict[str, str | int | float | bool] | None:
    # Attempt to load the contact info from json. If the file exists and is corrupt,
    # attempt to re-create up to 5 times.
    contact_info = safe_read_contact_json(contact_json)
    attempts = 0
    while contact_info is None and attempts < 5:
        # Attempt to re-generate the contact json for the PDB file.
        generate_dimer_foldseek_contacts(
            pdb_file,
            config,
        )
        attempts += 1
        contact_info = safe_read_contact_json(contact_json)
    return contact_info


def collect_contact_jsons(
    data_dir: Path,
    dimer_pdbs: list[Path],
    config: ContactConfig,
    config_hash: str,
    use_cache: bool = True,
) -> None:
    """Return a mapping from monomer pairs to Interface objects

    Collects json files storing foldseek-formatted contacts (assumed to consist of two monomers),
    creates Interface objects using these data, filters those Interfaces based on criteria, and
    constructs a map between pairs of monomer identifiers (str) and Interface objects that
    describe the interface between the monomer pair.

    Any skipped systems are written to a log file.

    Parameters
    ----------
    data_dir: Path
        Path to the data ingestion directory.
    dimer_pdbs: list[Path]
        List of dimer PDBs which should have contact jsons on disk.
    config: ContactConfig
        Config object used to determine the directory with a hash name based on config used for contacts.
    config_hash: str
        MD5 hash of the config object used.
    use_cache: bool
        Whether to skip creation of the interface dictionary if the output pickle file exists on disk.

    Returns
    -------
    None
        The interface dictionary is written to a pickle file called `interfaces.pkl`,
        with contents of type dict[tuple[str, str], Interface].

    """

    foldseek_data_dir = data_dir.parent / "foldseek_contacts"
    hash_fp = foldseek_data_dir / config_hash
    hash_fp.mkdir(exist_ok=True, parents=True)
    interface_pkl = hash_fp / "interfaces.pkl"
    if use_cache and interface_pkl.is_file():
        return
    interface_dict = {}
    too_short: set[tuple[str, str]] = set()
    skip: set[tuple[str, str]] = set()
    failed_read: set[str] = set()
    for pdb_file in tqdm(dimer_pdbs):
        contact_fp = pdb_file.parent / "foldseek_contacts"
        config_dir = contact_fp / config_hash
        contact_json = config_dir / f"{pdb_file.stem}.json"
        if not contact_json.is_file():
            # We can assume there were no contacts or it failed upstream
            continue

        contact_info = load_contact_info(pdb_file, contact_json, config)
        if contact_info is None:
            failed_read.add(contact_json.stem)
            continue

        # TODO: create a TypedDict / pydantic model to skip these
        id1 = contact_info["interface_id1"]
        id2 = contact_info["interface_id2"]
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        interface_id = (
            id1,
            id2,
        )
        R_len = contact_info["R_length"]
        L_len = contact_info["L_length"]
        assert isinstance(R_len, int)
        assert isinstance(L_len, int)
        short = R_len < config.min_length or L_len < config.min_length
        if short or interface_id in skip:
            too_short.add(interface_id)
            continue

        interface_dict[interface_id] = Interface.from_contact_info(contact_info)

    log.info(f"Skipped {len(too_short)} interfaces with length < {config.min_length}.")
    # write skipped interfaces to file
    with open(hash_fp / "too_short_interfaces.txt", "w") as f:
        for iface in too_short:
            f.write("\t".join(iface) + "\n")
    with open(hash_fp / "corrupt_contact_json.txt", "w") as f:
        for file_name in failed_read:
            f.write(file_name + "\n")
    with open(interface_pkl, "wb") as f:
        pickle.dump(interface_dict, f)

    log.info(
        f"Found {len(interface_dict)} interfaces with minimum length {config.min_length} residues"
    )


def load_interface_pkl(interface_pkl: Path) -> dict[tuple[str, str], Interface]:
    with open(interface_pkl, "rb") as f:
        interfaces: dict[tuple[str, str], Interface] = pickle.load(f)
        return interfaces


def write_interface_dict(
    interface_dict: dict[tuple[str, str], Interface],
    filepath: Path,
) -> None:
    """Write the interface dictionary to pkl file.

    Parameters
    ----------
    interface_dict: Dict[Tuple[str, str], Interface]
        Dictionary mapping dimers (tuples of monomer IDs) to Interface objects
    filepath: Path
        The path to which to write "interfaces.pkl"
    """
    with open(filepath / "interfaces.pkl", "wb") as f:
        pickle.dump(interface_dict, f)


def get_interfaces_from_config(
    contact_root: Path,
    config_hash: str | None = None,
) -> dict[tuple[str, str], Interface]:
    if isinstance(config_hash, str):
        hash_str = config_hash
    else:
        hash_str = get_config_hash(ContactConfig())
    hash_fp = contact_root / hash_str
    interface_pkl = hash_fp / "interfaces.pkl"
    return load_interface_pkl(interface_pkl)
