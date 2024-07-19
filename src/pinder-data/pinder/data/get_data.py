from __future__ import annotations
import multiprocessing
from itertools import repeat
from pathlib import Path
from typing import Any

import biotite.structure as struc
import gemmi
import gzip
from concurrent import futures
import numpy as np
import pandas as pd
import os
import re
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure
from tqdm import tqdm

from pinder.core.index.id import Dimer, Monomer, Protein
from pinder.core.index.utils import set_mapping_column_types
from pinder.core.utils import setup_logger
from pinder.core.utils.paths import empty_file
from pinder.core.structure.atoms import get_buried_sasa
from pinder.core.structure.contacts import pairwise_contacts
from pinder.core.structure.atoms import get_resolved_resi_from_atom_array
from pinder.data.config import PinderDataGenConfig
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.pipeline.constants import UNIPROT_UNDEFINED

log = setup_logger(__name__)
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


def ingest_rscb_files(
    data_dir: Path = Path("."),
    two_char_code: str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    config: PinderDataGenConfig = PinderDataGenConfig(),
) -> None:
    """
    Process the downloaded RCSB files by globs on the data directory.

    Parameters
    ----------
    data_dir : Path
        The directory where the downloaded files are stored.
    two_char_code : Optional[str]
        A two character code representing the batch of files to process.
        If not provided, all files will be processed.
    parallel : bool
        If True, files will be processed in parallel.
    max_workers : int | None
        If specified, limits number of processes to spawn in parallel mode.
    config : PinderDataGenConfig
        Configuration parameters for dataset generation.

    Returns
    -------
    None
    """
    path_to_pdb = Path(data_dir)
    # collect all source CIF files to process
    processing_queue = [
        f
        for two_char_subdir in path_to_pdb.iterdir()
        for pdb_subdir in two_char_subdir.iterdir()
        for f in pdb_subdir.glob("*.cif.gz")
        if (
            (not two_char_code or two_char_subdir.name == two_char_code)
            and not f.name.startswith(".")
            and not two_char_subdir.name.startswith(".")
        )
    ]
    # print(path_to_pdb, processing_queue)
    ingest_mmcif_list(
        processing_queue, parallel=parallel, max_workers=max_workers, config=config
    )


def ingest_mmcif_list(
    mmcif_list: list[Path],
    parallel: bool = True,
    max_workers: int | None = None,
    config: PinderDataGenConfig = PinderDataGenConfig(),
    use_cache: bool = True,
) -> None:
    """Process a list of downloaded RCSB mmcif files.

    Parameters
    ----------
    mmcif_list : Path
        The list of mmcif files to process in a batch.
    parallel : bool
        If True, files will be processed in parallel.
    max_workers : int | None
        If specified, limits number of processes to spawn in parallel mode.
    config : PinderDataGenConfig
        Configuration parameters for dataset generation.

    Returns
    -------
    None
    """
    log.info(f"Processing queue {len(mmcif_list)}")
    if len(mmcif_list) > 0:
        # process all files in parallel
        if parallel:
            # max_workers=param_n_cores
            with futures.ProcessPoolExecutor(
                mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers
            ) as exe:
                exe.map(process_mmcif, mmcif_list, repeat(config), repeat(use_cache))
        else:
            for mmcif_file in tqdm(mmcif_list):
                process_mmcif(mmcif_file, config, use_cache)


def generate_bio_assembly(mmcif_filename: Path) -> tuple[Any, pd.DataFrame]:
    """Generate biological assemblies for the given mmCIF file"""
    # read in the CIF structure
    block = gemmi.cif.read(str(mmcif_filename))[0]

    structure = gemmi.make_structure_from_block(block)

    # Potentially better way to interrogate stoichometry and transforms
    # assembly = structure.assemblies[0]
    # gen = assembly.generators[0]
    # gen.chains
    # gen.subchains
    # len(gen.operators)

    # generate assembly with duplicate chains handled with copy number added
    asm_addnum = gemmi.make_assembly(
        structure.assemblies[0], structure[0], gemmi.HowToNameCopiedChain.AddNumber
    )
    # generate WITHOUT handling duplicates - will be in same order as AddNumber
    asm_dup = gemmi.make_assembly(
        structure.assemblies[0], structure[0], gemmi.HowToNameCopiedChain.Dup
    )
    # This modifies structure in-place and loses lineage to copy number
    # structure.transform_to_assembly(
    #     assembly_name="1", how=gemmi.HowToNameCopiedChain.AddNumber
    # )

    renum_map = []
    for renum_subch, dup_subch in zip(asm_addnum.subchains(), asm_dup.subchains()):
        # renum_subch is the chain_id that will be used for final assembly
        # dupe_subch is the chain_id before adding a copy number.
        # We need this to safely map entities, asym_id, instance number
        # and author / pdb_strand_id in downstream use-cases.
        renum_map.append(
            {
                "chain_id": renum_subch.subchain_id(),
                "asym_id": dup_subch.subchain_id(),
            }
        )
    renum_map = pd.DataFrame(renum_map)

    # While we have original structure, lets get entity_id <-> asym_id mappings
    entity_chain_map = []
    for entity in structure.entities:
        entity_chain_map.extend(
            [{"entity_id": entity.name, "asym_id": ch} for ch in entity.subchains]
        )
    entity_chain_map = pd.DataFrame(entity_chain_map)
    entity_chain_map = pd.merge(renum_map, entity_chain_map, how="left")

    # do not depend on parent directory name
    entry_id = mmcif_filename.stem.split("_")[1][-4:].lower()
    new_fname = mmcif_filename.parent / f"{entry_id}-assembly.cif"

    # Transform the Structure object to assembly. make_assembly produces Model,
    # not Structure which can't be converted to a CIF document.
    structure.transform_to_assembly(
        assembly_name="1", how=gemmi.HowToNameCopiedChain.AddNumber
    )
    structure.make_mmcif_document().write_file(str(new_fname))
    # read and return the assembly in biotite format
    asm_pdbx_file = read_mmcif_file(new_fname)
    bio_asm = get_structure(
        asm_pdbx_file,
        model=1,
        use_author_fields=False,
        extra_fields=["b_factor", "charge"],
    )
    return bio_asm, entity_chain_map


def read_mmcif_file(mmcif_filename: Path) -> PDBxFile:
    """Read a PDBx/mmCIF file."""
    if mmcif_filename.suffix == ".gz":
        with gzip.open(mmcif_filename, "rt") as mmcif_file:
            pdbx_file = PDBxFile.read(mmcif_file)
    else:
        pdbx_file = PDBxFile.read(mmcif_filename)
    return pdbx_file


def convert_category(
    category: dict[str, np.ndarray[Any, Any]],
) -> dict[int, dict[str, Any]]:
    """Convert a PDBx/mmCIF category to a dictionary indexed by sequential ids.
    with keys and values taken from the original value arrays.
    """
    category_dict: dict[int, dict[str, Any]] = {}
    if category is not None:
        for i in range(len(category[list(category.keys())[0]])):
            category_dict[i] = {}
            for key, value in category.items():
                category_dict[i][key] = value[i]
    return category_dict


def replace_with_nan(value: Any) -> Any:
    if value == "?" or value == "NaN":
        return np.nan
    return value


def get_mmcif_category(
    pdbx_file: PDBxFile, category_name: str
) -> dict[int, dict[str, Any]]:
    """Get a PDBx/mmCIF category as a dictionary"""
    cat = convert_category(pdbx_file.get_category(category_name, expect_looped=True))
    return cat


def infer_uniprot_from_mapping(mapping_df: pd.DataFrame) -> str:
    """Assign uniprot based on largest number of residues in mapping (in case of chimera)"""
    if "uniprot_acc" not in mapping_df.columns:
        uniprot: str = UNIPROT_UNDEFINED
        return uniprot
    uniprot_sizes = mapping_df.dropna().groupby("uniprot_acc").size()
    pdb_id = mapping_df.iloc[0]["entry_id"]
    asym_id = mapping_df.iloc[0]["asym_id"]
    pdb_strand_id = mapping_df.iloc[0]["pdb_strand_id"]
    if uniprot_sizes.shape[0] > 1:
        log.warning(
            f"Multiple uniprots found for {pdb_id} {asym_id} ({pdb_strand_id}). It is likely a chimera protein"
        )
    uniprot = uniprot_sizes.idxmax() if uniprot_sizes.shape[0] else UNIPROT_UNDEFINED
    assert isinstance(uniprot, str)
    return uniprot


def sequence_mapping(
    pdbx_file: PDBxFile, entry_id: str, entity_id: str
) -> pd.DataFrame:
    """Get sequence mapping from a PDBx/mmCIF file for all chains."""
    f = pdbx_file
    cat = get_mmcif_category(f, "entity_poly")
    for entity in cat.values():
        if entity["type"] == "polypeptide(L)" and entity["entity_id"] == entity_id:
            canonical_sequence = entity["pdbx_seq_one_letter_code_can"]

    structure = get_structure(pdbx_file, model=1, use_author_fields=False)
    chain_resolved_resi = get_resolved_resi_from_atom_array(structure)
    residue_mapping = []
    cat = get_mmcif_category(f, "pdbx_poly_seq_scheme")
    for residue in cat.values():
        if residue["entity_id"] == entity_id:
            resolved_resi: list[int] = chain_resolved_resi.get(residue["asym_id"], [])
            residue_mapping.append(
                {
                    "entity_id": int(residue["entity_id"]),
                    "asym_id": residue["asym_id"],
                    "pdb_strand_id": residue["pdb_strand_id"],
                    "resi": int(residue["seq_id"]),
                    "resi_pdb": int(residue["pdb_seq_num"]),
                    "resi_auth": replace_with_nan(residue["auth_seq_num"]),
                    "resn": residue["mon_id"],
                    "one_letter_code_can": canonical_sequence[
                        int(residue["seq_id"]) - 1
                    ],
                    "resolved": int(int(residue["seq_id"]) in resolved_resi),
                }
            )
    df = pd.DataFrame(residue_mapping)

    # load up uniprot residue mapping
    cat = get_mmcif_category(f, "atom_site")
    df_uniprot = (
        pd.DataFrame(cat.values())
        .query(f'label_entity_id == "{entity_id}"')
        .query("label_atom_id == 'CA'")
    )

    if "pdbx_sifts_xref_db_res" in df_uniprot.columns:
        df_uniprot = df_uniprot[
            [
                # "label_entity_id", "label_comp_id",
                "label_seq_id",
                "pdbx_sifts_xref_db_res",
                "pdbx_sifts_xref_db_num",
                "pdbx_sifts_xref_db_acc",
            ]
        ]
        df_uniprot = (
            df_uniprot.rename(
                columns={
                    # "label_entity_id": "entity_id",
                    "label_seq_id": "resi",
                    # "label_comp_id": "resn",
                    "pdbx_sifts_xref_db_res": "one_letter_code_uniprot",
                    "pdbx_sifts_xref_db_num": "resi_uniprot",
                    "pdbx_sifts_xref_db_acc": "uniprot_acc",
                }
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df_uniprot = df_uniprot.map(replace_with_nan)
        df_uniprot["resi"] = df_uniprot["resi"].astype(int)
        df_uniprot["resi_uniprot"] = df_uniprot["resi_uniprot"]
        df = df.merge(
            df_uniprot, left_index=False, left_on="resi", right_on="resi", how="left"
        )

    df.insert(0, "entry_id", entry_id)
    return df


def get_entities(pdbx_file: PDBxFile, entry_id: str) -> pd.DataFrame | None:
    """Get entities from a PDBx/mmCIF file."""
    f = pdbx_file
    protein_entities = []

    try:
        cat = get_mmcif_category(f, "entity_poly")
        for entity in cat.values():
            if entity.get("type") == "polypeptide(L)":
                protein_entities.append(
                    {
                        "entity_id": entity["entity_id"],
                        "sequence": entity["pdbx_seq_one_letter_code_can"],
                        "length": len(entity["pdbx_seq_one_letter_code_can"]),
                        "organism": None,
                        "tax_id": None,
                        "type": entity.get("type"),
                        # "asym_id": entity["pdbx_strand_id"],
                    }
                )
    except Exception:
        log.error(f"error in entity_poly for {entry_id}")
        return None

    try:
        cat = get_mmcif_category(f, "entity_src_gen")
        for entity in cat.values():
            for protein_entity in protein_entities:
                if protein_entity["entity_id"] == entity["entity_id"]:
                    protein_entity["organism"] = entity["pdbx_gene_src_scientific_name"]
                    protein_entity["tax_id"] = entity["pdbx_gene_src_ncbi_taxonomy_id"]
    except Exception:
        # log.error(f"error in entity_src_gen for {entry_id}")
        pass

    try:
        cat = get_mmcif_category(f, "entity_src_nat")
        for entity in cat.values():
            for protein_entity in protein_entities:
                if protein_entity["entity_id"] == entity["entity_id"]:
                    protein_entity["organism"] = entity["pdbx_organism_scientific"]
                    protein_entity["tax_id"] = entity["pdbx_ncbi_taxonomy_id"]
    except Exception:
        # log.error(f"error in entity_src_nat for {entry_id}")
        pass

    # either entity_src_gen or entity_src_nat or entity_src_syn should be present
    try:
        cat = get_mmcif_category(f, "pdbx_entity_src_syn")
        for entity in cat.values():
            for protein_entity in protein_entities:
                # print(protein_entity)
                if protein_entity["entity_id"] == entity["entity_id"]:
                    protein_entity["organism"] = entity["organism_scientific"]
                    protein_entity["tax_id"] = entity["ncbi_taxonomy_id"]
    except Exception:
        log.error(f"error in entity_src_nat for {entry_id}")
        pass

    df_protein_entities = pd.DataFrame(protein_entities)
    if df_protein_entities.shape[0] == 0:
        log.error(f"no protein entities for {entry_id}")
        return None

    try:
        cat = get_mmcif_category(f, "pdbx_poly_seq_scheme")
        dict_entities: dict[str, list[str | int]] = {
            "entity_id": [],
            "asym_id": [],
            "pdb_strand_id": [],
        }
        for residue in cat.values():
            dict_entities["entity_id"].append(residue["entity_id"])
            dict_entities["asym_id"].append(residue["asym_id"])
            dict_entities["pdb_strand_id"].append(residue["pdb_strand_id"])

        df_entities = (
            pd.DataFrame(dict_entities).drop_duplicates().reset_index(drop=True)
        )
        if df_entities.shape[0] == 0:
            return None

        df_entities = df_protein_entities.merge(
            df_entities, on=["entity_id"], how="left"
        ).reset_index(drop=True)
        df_entities.insert(0, "entry_id", entry_id)
        # print(df_entities.to_string())
        return df_entities
    except Exception:
        return None


def get_metadata(pdbx_file: PDBxFile) -> pd.DataFrame | None:
    """Get metadata from a PDBx/mmCIF file.

    Beware of special cases,
    e.g. who would have thought there are entries with multiple methods? https://www.rcsb.org/structure/7a0l
    """
    f = pdbx_file
    meta: dict[str, float | str | int | None] = {}

    meta["entry_id"] = None
    meta["method"] = None
    meta["date"] = f["pdbx_database_status"]["recvd_initial_deposition_date"]
    try:
        revisions = f["pdbx_audit_revision_history"]["revision_date"]
        if isinstance(revisions, np.ndarray):
            # Take the earliest release date
            meta["release_date"] = revisions[0]
        else:
            # There is only one revision date associated with the entry (initial release)
            meta["release_date"] = revisions
    except (KeyError, IndexError):
        # Fallback, there shouldn't be any entries without the release date.
        # It's 100% according to https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/pdbx_audit_revision_history.html
        meta["release_date"] = meta["date"]
    meta["resolution"] = None
    meta["assembly"] = None
    meta["assembly_details"] = None
    meta["oligomeric_details"] = None
    meta["oligomeric_count"] = None
    meta["biol_details"] = None
    meta["complex_type"] = None
    meta["status"] = None

    try:
        cat = get_mmcif_category(f, "exptl")
        if len(cat) > 0:
            meta["entry_id"] = cat[0]["entry_id"]
            meta["method"] = cat[0]["method"]
    except Exception:
        pass

    try:
        cat = get_mmcif_category(f, "refine")
        resolution = None
        if len(cat) > 0:
            resolution = cat[0]["ls_d_res_high"]
        if resolution is None or resolution in ["?", "."]:
            cat = get_mmcif_category(f, "em_3d_reconstruction")
            if len(cat) > 0:
                resolution = cat[0]["resolution"]
        meta["resolution"] = resolution
    except Exception:
        pass

    try:
        cat = get_mmcif_category(f, "pdbx_struct_assembly")
        # consider the first assembly only
        if len(cat) > 0:
            meta["assembly"] = cat[0]["id"]
            meta["assembly_details"] = cat[0]["details"]
            meta["oligomeric_details"] = cat[0]["oligomeric_details"]
            meta["oligomeric_count"] = cat[0]["oligomeric_count"]
    except Exception:
        pass

    try:
        cat = get_mmcif_category(f, "struct_biol")
        if len(cat) > 0:
            meta["biol_details"] = cat[0].get("details")
    except Exception:
        pass

    # consider the first experimental method only
    df = pd.DataFrame([meta], index=[0])
    return df


def get_structure_chains(
    structure: struc.AtomArrayStack | struc.AtomArray,
) -> list[str]:
    """Get all chains in a structure ordered by decreasing size (in residues)
    if size is the same, order by chain id alphabetically (1 11 12 2 3 4 ...)
    the logic is that we want to assign the largest chains as Receptors and smaller chains as Ligands
    such that in a R::L dimer, R is the largest chain and L is the smallest chain
    """
    chain_sizes = {
        chain: struc.get_residue_count(structure[structure.chain_id == chain])
        for chain in set(structure.chain_id)
    }
    chains = sorted(chain_sizes.items(), key=lambda x: (-x[1], x[0]))
    return [chain[0] for chain in chains]


def get_interacting_chains(
    structure: struc.AtomArray,
    entities: pd.DataFrame,
    contact_threshold: float = 10.0,
    backbone_only: bool = True,
) -> pd.DataFrame:
    """Identify interacting chains in a structure.

    `backbone_only`:
        The method focuses on protein backbone atoms as defined by DockQ
        atom names (C, CA, N, O). Due to the focus on backbone atoms, only
        residue-level contact information is returned.

    """

    chains = get_structure_chains(structure)
    log.info(f"chains {chains}")
    interacting_chains: list[dict[str, str | int | float]] = []
    if structure.coord.shape[0] == 0:
        return pd.DataFrame(interacting_chains)

    for i, chain_1 in enumerate(chains):
        protein_1_mask = structure.chain_id == chain_1
        for j, chain_2 in enumerate(chains):
            if j <= i:
                continue
            if chain_1 == chain_2:
                continue
            protein_2_mask = structure.chain_id == chain_2
            # NOTE! chain_1 is generally bigger receptor chain.
            # If chain_1 and chain_2 are the same size, it is alphabetical.
            res_conts = pairwise_contacts(
                structure,
                radius=contact_threshold,
                heavy_only=False,
                backbone_only=backbone_only,
                chain1=chain_1,
                chain2=chain_2,
                atom_and_residue_level=False,
            )
            if len(res_conts):
                dsasa = get_buried_sasa(
                    structure[protein_1_mask], structure[protein_2_mask]
                )
                chain1_res = {cp[2] for cp in res_conts}
                chain2_res = {cp[3] for cp in res_conts}

                interacting_chains.append(
                    {
                        "entry_id": entities.entry_id.values[0],
                        "chain_1": chain_1,
                        "asym_id_1": entities.query("chain == @chain_1").asym_id.values[
                            0
                        ],
                        "pdb_strand_id_1": entities.query(
                            "chain == @chain_1"
                        ).pdb_strand_id.values[0],
                        "chain_2": chain_2,
                        "asym_id_2": entities.query("chain == @chain_2").asym_id.values[
                            0
                        ],
                        "pdb_strand_id_2": entities.query(
                            "chain == @chain_2"
                        ).pdb_strand_id.values[0],
                        # "n_atom_pairs": len(at_conts),
                        "n_residue_pairs": len(res_conts),
                        "n_residues": len(chain1_res) + len(chain2_res),
                        "buried_sasa": dsasa,
                        "chain_1_residues": ",".join(
                            map(str, list(sorted(chain1_res)))
                        ),
                        "chain_2_residues": ",".join(
                            map(str, list(sorted(chain2_res)))
                        ),
                    }
                )
    return pd.DataFrame(interacting_chains)


def save(meta: pd.DataFrame, metadata_file: Path) -> None:
    meta.to_csv(metadata_file, index=False, sep="\t")


def save_mapping_checkpoint(checkpoint_file: Path) -> None:
    with open(checkpoint_file, "w") as f:
        f.write("complete")
    try:
        assert not empty_file(checkpoint_file)
    except AssertionError:
        log.debug(f"Failed to write {checkpoint_file}, retrying...")
        return save_mapping_checkpoint(checkpoint_file)


def process_mmcif(
    mmcif_file: Path,
    config: PinderDataGenConfig = PinderDataGenConfig(),
    use_cache: bool = True,
) -> None:
    """Process a single mmCIF file from the next generation PDB archive.

    Parameters
    ----------
    mmcif_file : Path
        The mmCIF file to be processed.
    config : PinderDataGenConfig
        Configuration parameters for dataset generation.
    use_cache : bool
        Whether to skip processing if the metadata file exists and status is set
        to one of the PROCESSED_STATUS_CODES
        [
            'complete', 'no metadata', 'assembly failed', 'entities failed',
            'non-protein assembly', 'too many chains'
        ]

    Returns
    -------
    None

    Notes
    -----
    - Saves metadata as a text file.
    - Saves biological assembly as an mmCIF file.
    - Saves interacting chains along with the interface metrics as a text file.
    - Saves all pairs of interacting chains as PDB files.
    - Saves residue numbers of interacting chains as a text file along with Uniprot IDs and numbering.
    """
    log.info(mmcif_file.parent)

    pdbx_file = read_mmcif_file(mmcif_file)
    pdb_id = mmcif_file.stem.split("_")[1][-4:].lower()
    log.info(f"PDB ID {pdb_id}")
    mapping_checkpoint = mmcif_file.parent / "checkpoint-mapping.txt"
    metadata_file = mmcif_file.parent / f"{pdb_id}-metadata.tsv"
    interacting_chains_tsv = mmcif_file.parent / f"{pdb_id}-interacting_chains.tsv"
    entities_pqt = mmcif_file.parent / f"{pdb_id}-entities.parquet"

    # Check cached results
    PROCESSED_STATUS_CODES = [
        "complete",
        "no metadata",
        "assembly failed",
        "entities failed",
        "non-protein assembly",
        "too many chains",
    ]
    if use_cache and not empty_file(mapping_checkpoint):
        log.info(
            f"Skipping patching of mapping file, {mmcif_file.stem}, checkpoint exists"
        )
        return

    if metadata_file.is_file() and use_cache:
        try:
            cached_metadata = pd.read_csv(metadata_file, sep="\t")
        except Exception as e:
            log.error(f"Failed to read existing metadata file. Removing...")
            metadata_file.unlink()
            cached_metadata = None

        if cached_metadata is not None and "status" in cached_metadata.columns:
            meta_status = cached_metadata.status.values[0]
            if meta_status in PROCESSED_STATUS_CODES and not empty_file(
                mapping_checkpoint
            ):
                log.info(
                    f"Skipping {pdb_id}, metadata status is {meta_status}, mapping patch checkpoint exists"
                )
                return None

    metadata = get_metadata(pdbx_file)

    if metadata is None or metadata.entry_id[0] is None:
        log.warning(f"no metadata for {mmcif_file}")
        if metadata is not None:
            metadata["entry_id"] = pdb_id
            metadata["status"] = "no metadata"
            save(metadata, metadata_file)
        save_mapping_checkpoint(mapping_checkpoint)
        return None

    # expect to have metadata data structure in order to proceed
    if pdb_id != metadata.entry_id[0].lower():
        log.warning(f"entry_id mismatch {pdb_id} {metadata.entry_id[0].lower()}")
        save_mapping_checkpoint(mapping_checkpoint)
        return None

    metadata.entry_id = metadata.entry_id.str.lower()
    metadata["status"] = "metadata"
    save(metadata, metadata_file)
    try:
        bio_asm, entity_chain_map = generate_bio_assembly(mmcif_file)
        # TODO: consume entity_chain_map and fix the flaky logic used the first time
        all_atom_asm_shape = bio_asm.shape[0]
        # keep only the protein in the assembly. DNA/RNA are gone
        # do not use HET as a filter. some non-standard amino acids are marked as HET
        bio_asm = bio_asm[struc.filter_amino_acids(bio_asm)].copy()
        protein_atom_asm_shape = bio_asm.shape[0]
        if protein_atom_asm_shape == 0 and all_atom_asm_shape > 0:
            log.info(f"All-atom bio-assembly had {all_atom_asm_shape} atoms")
            log.info("There are no protein atoms in the assembly. Skipping...")
            metadata["status"] = "non-protein assembly"
            save(metadata, metadata_file)
            save_mapping_checkpoint(mapping_checkpoint)
            return None
    except Exception as e:
        log.error(f"assembly failed for {mmcif_file}. Error: {str(e)}")
        metadata["status"] = "assembly failed"
        save(metadata, metadata_file)
        save_mapping_checkpoint(mapping_checkpoint)
        return None

    # attn: reading from the original file, not from the assembly file
    entities = get_entities(pdbx_file, entry_id=pdb_id)
    if entities is None:
        metadata["status"] = "entities failed"
        save(metadata, metadata_file)
        save_mapping_checkpoint(mapping_checkpoint)
        return None
    structure_chains = get_structure_chains(bio_asm)

    metadata["n_chains"] = len(structure_chains)
    metadata["status"] = "bio assembly"
    save(metadata, metadata_file)

    chain_mapping = {}
    unique_uniprot_acc = set()
    chains = []
    for entity_id in entities.entity_id.unique():
        entity_mapping = sequence_mapping(
            pdbx_file, entry_id=pdb_id, entity_id=entity_id
        )
        # print("Entity mapping", entity_mapping.shape)
        # print("Asyms", entities.query("entity_id == @entity_id").asym_id.unique())
        for asym_id in entities.query("entity_id == @entity_id").asym_id.unique():
            # Get the mapping for this asym_id and cast entity to str
            # (gemmi makes a note that entity_id is NOT strictly int, though should be)
            # They cast to string for safety, but our entity_mapping is int.
            mapping = entity_mapping.query("asym_id == @asym_id").reset_index(drop=True)
            mapping.entity_id = mapping.entity_id.astype(str)
            uniprot = infer_uniprot_from_mapping(mapping)
            unique_uniprot_acc.add(uniprot)
            # Merge the entity mapping for this asym_id with the entity-asym-chain mapping
            # we got from gemmi during chain renumbering
            mapping = pd.merge(mapping, entity_chain_map)
            for chain in structure_chains:
                # TODO: this for loop should be revisited,
                # we can probably avoid and just do some merges upstream
                chain_map = (
                    mapping.query("chain_id == @chain")
                    .reset_index(drop=True)
                    .rename({"chain_id": "chain"}, axis=1)
                )
                if not chain_map.shape[0]:
                    continue

                chain_mapping[chain] = chain_map.copy()
                chains.append(
                    {
                        "asym_id": asym_id,
                        "chain": chain,
                        "uniprot": uniprot,
                        "length_resolved": chain_mapping[chain].resolved.sum(),
                    }
                )

    # Simply add the actual assembly chain that we have in the structure
    # This chain has a number added by gemmi to handle duplicate instances
    entities = pd.merge(entities, pd.DataFrame(chains), on="asym_id", how="right")

    # Remove any chains that are non-polypeptide that may have been removed in mapping
    bio_asm = bio_asm[np.isin(bio_asm.chain_id, list(entities.chain.drop_duplicates()))]
    entities.to_parquet(entities_pqt, index=False)
    structure_chains = get_structure_chains(bio_asm)

    if len(unique_uniprot_acc) > 1:
        metadata["complex_type"] = "heteromer"
    elif len(unique_uniprot_acc) == 1:
        metadata["complex_type"] = "homomer"
    else:
        metadata["complex_type"] = "unknown"

    metadata["status"] = "entities"
    save(metadata, metadata_file)

    if len(structure_chains) > config.max_assembly_chains:
        metadata["status"] = "too many chains"
        save(metadata, metadata_file)
        log.error(f"Skipping {mmcif_file}, too many chains: {len(structure_chains)}")
        save_mapping_checkpoint(mapping_checkpoint)
        return None

    # INTERACTIONS
    interacting_chains = get_interacting_chains(
        bio_asm,
        entities,
        contact_threshold=config.interacting_chains_radius,
        backbone_only=config.interacting_chains_backbone_only,
    )
    interacting_chains.to_csv(
        interacting_chains_tsv,
        index=False,
        sep="\t",
    )
    metadata["status"] = "interacting chains"
    save(metadata, metadata_file)

    if interacting_chains.shape[0] == 0:
        # show a warning if more than 1 chain but they don't interact
        # save monomer(s) as true monomers
        if len(structure_chains) > 1:
            log.warning(
                f"multiple chains detected but no interacting chains for {pdb_id}"
            )
            metadata["complex_type"] = "monomer"
            metadata["status"] = "monomer"
            save(metadata, metadata_file)

        # in case there are multiple non-interacting (apo) chains
        for chain in structure_chains:
            uniprot = entities.query(f'chain == "{chain}"').uniprot.values[0]
            protein = Protein(source=pdb_id, chain=chain, uniprot=uniprot)
            monomer = Monomer(
                [protein],
            )
            monomer_pdb_file = PDBFile()
            monomer_structure = bio_asm[bio_asm.chain_id == chain].copy()
            monomer_structure.chain_id[:] = "A"
            monomer_pdb_file.set_structure(monomer_structure)
            monomer_filename = Path(mmcif_file.parent / f"{monomer}.pdb")
            monomer_pdb_file.write(monomer_filename)

            # save mapping
            monomer_mapping_dest = mmcif_file.parent / f"{monomer}.parquet"
            chain_mapping[chain] = set_mapping_column_types(chain_mapping[chain]).copy()
            chain_mapping[chain].to_parquet(monomer_mapping_dest, index=False)

    else:
        # save dimer complexes in the PDB format
        for chain_1, chain_2 in interacting_chains[["chain_1", "chain_2"]].values:
            # save two interacting chains in the PDB format
            dimer_pdb_file = PDBFile()
            two_chains = bio_asm[np.isin(bio_asm.chain_id, (chain_1, chain_2))]
            mask1 = two_chains.chain_id == chain_1
            mask2 = two_chains.chain_id == chain_2
            two_chains.chain_id[mask1] = "R"
            two_chains.chain_id[mask2] = "L"
            # Format identifiers for individual interacting chains
            chain1_uniprot = entities.query("chain == @chain_1").uniprot.values[0]
            chain2_uniprot = entities.query("chain == @chain_2").uniprot.values[0]
            protein1 = Protein(source=pdb_id, chain=chain_1, uniprot=chain1_uniprot)
            protein2 = Protein(source=pdb_id, chain=chain_2, uniprot=chain2_uniprot)
            dimer = Dimer(Monomer([protein1]), Monomer([protein2]))
            try:
                dimer_pdb_file.set_structure(two_chains)
                if two_chains.shape[0] > 99999:
                    log.error(f"too many atoms in dimer {pdb_id} {chain_1} {chain_2}")
                    continue

                dimer_dest_pdb = mmcif_file.parent / f"{dimer}.pdb"
                dimer_pdb_file.write(dimer_dest_pdb)
            except Exception:
                log.error(f"failed to write {pdb_id} {chain_1} {chain_2}")
                # don't write single chains when the dimer failed to write
                continue

            # save individual chains as holo monomers in the PDB format
            for chain_id, protein, side, mapping in (
                (chain_1, protein1, "R", chain_mapping[chain_1]),
                (chain_2, protein2, "L", chain_mapping[chain_2]),
            ):
                single_chain = bio_asm[bio_asm.chain_id == chain_id]
                # always assign chain A to the single chain protein
                single_chain.chain_id[single_chain.chain_id == chain_id] = side
                monomer = Monomer([protein], side=side)

                if single_chain.shape[0] > 99999:
                    log.error(f"too many atoms in monomer {pdb_id} {chain_1} {chain_2}")
                    continue

                if single_chain.shape[0] == 0:
                    log.error(f"no atoms in monomer {pdb_id} {chain_1} {chain_2}")
                    continue

                single_chain_file = PDBFile()
                try:
                    single_chain_file.set_structure(single_chain, hybrid36=False)
                    monomer_dest_pdb = mmcif_file.parent / f"{monomer}.pdb"
                    single_chain_file.write(monomer_dest_pdb)
                except Exception:
                    log.error(f"failed to write {pdb_id} {chain_id}")
                    continue
                # save mapping
                monomer_mapping_dest = mmcif_file.parent / f"{monomer}.parquet"
                mapping = set_mapping_column_types(mapping).copy()
                mapping.to_parquet(monomer_mapping_dest, index=False)

    metadata["status"] = "complete"
    save(metadata, metadata_file)
    save_mapping_checkpoint(mapping_checkpoint)
    log.info(f"File {mmcif_file} processed successfully")
    return None


if __name__ == "__main__":
    # for each file in /tmp/pinder-debug/*-enrich.cif.gz
    # run process_mmcif
    for mmcif_file in Path("/tmp/pinder-debug").glob("*-enrich.cif.gz"):
        process_mmcif(mmcif_file)
