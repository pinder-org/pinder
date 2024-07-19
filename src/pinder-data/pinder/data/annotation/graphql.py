from __future__ import annotations
import json
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any

import requests
import pandas as pd
from python_graphql_client import GraphqlClient
from tqdm import tqdm

from pinder.core.utils.log import setup_logger
from pinder.core.utils.process import process_starmap
from pinder.core.utils.retry import exponential_retry
from pinder.data.annotation.constants import (
    GRAPHQL_ANNOTATION_QUERY,
    RCSB_GRAPHQL_ENDPOINT,
    TYPE_PATTERNS,
)
from pinder.data.csv_utils import parallel_read_csvs


LOG = setup_logger(__name__)
CLIENT = GraphqlClient(endpoint=RCSB_GRAPHQL_ENDPOINT)


@exponential_retry(max_retries=10, exceptions=(requests.exceptions.HTTPError,))
def run_graphql_annotation_query(pdb_id: str) -> dict[str, Any]:
    """Fetch annotations for an entry by PDB entry ID.

    The data returned is identical to the data used to generate webpages for entries.
    For example: https://www.rcsb.org/annotations/2A79
    The query is taken from the Data API widget.

    Parameters
    ----------
    pdb_id : str
        The PDB entry ID to fetch.

    Returns
    -------
    data: dict[str, Any]
        A dictionary of data for the entry. If the entry is not found, the response will
        be {'data': {'entry': None}}.

    """
    data: dict[str, Any] = CLIENT.execute(
        query=GRAPHQL_ANNOTATION_QUERY, variables={"id": pdb_id}
    )
    return data


def fetch_entry_annotations(
    pdb_id: str,
    data_json: Path,
    use_cache: bool = True,
) -> None:
    """Fetch annotations for an entry by PDB entry ID and store in json.

    If the entry is not found, the json is still saved with status: empty.
    If the query fails, e.g. due to ratelimit exceeded or network connectivity,
    the json is saved with status: failed. The contents of the json are used to
    enable cached results without requiring a new query unless specified.

    Parameters
    ----------
    pdb_id : str
        The PDB entry ID to fetch.
    data_json : Path
        The Path to the json file to write data to.
    use_cache : bool
        Whether to skip the query if valid results exist on disk.

    Returns
    -------
    None

    """
    if use_cache and data_json.is_file():
        try:
            with open(data_json) as f:
                data = json.load(f)
                if data.get("data") or data.get("status") == "empty":
                    # Its a valid data json
                    return None
        except Exception as e:
            LOG.warning(f"PDB ID {pdb_id} had unreadable json data. Retrying...")

    try:
        data = run_graphql_annotation_query(pdb_id)
        if not data.get("data", {}).get("entry"):
            data = {"pdb_id": pdb_id, "status": "empty"}
    except Exception as e:
        print(f"Failed to fetch entry annotations for {pdb_id}")
        data = {"pdb_id": pdb_id, "status": "failed"}

    with open(data_json, "w") as f:
        json.dump(data, f)


def parse_pfam(polymer_entity: dict[str, Any]) -> pd.DataFrame:
    """Extract PFAM (protein family) annotations from a polymer entity."""
    try:
        pfams = pd.DataFrame(polymer_entity["pfams"])
    except Exception:
        pfams = pd.DataFrame()

    if not pfams.empty:
        # feature_id in polymer_entity_feats maps to rcsb_pfam_accession
        pfams.loc[:, "feature_id"] = pfams.rcsb_pfam_accession
        # polymer_entity_ids = pd.DataFrame([polymer_entity['rcsb_polymer_entity_container_identifiers']])
        polymer_entity_feats = pd.DataFrame(
            polymer_entity["rcsb_polymer_entity_feature"]
        )
        # There are cases where pfams dataframe HAS feature_positions, additional_properties, description, etc.
        # and others where its ONLY in the polymer_entity_feats. This causes a merge conflict. Lets rely on the
        # entity_feats for that info. We only want to merge on feature_id
        pfam_cols = set(pfams.columns)
        feat_cols = set(polymer_entity_feats.columns)
        common_cols = pfam_cols.intersection(feat_cols)
        common_cols = common_cols - {"feature_id"}
        if common_cols:
            pfams = pfams.drop(list(common_cols), axis=1).copy()
        pfams = pd.merge(pfams, polymer_entity_feats, how="left", on="feature_id")
        for k, v in polymer_entity["rcsb_polymer_entity_container_identifiers"].items():
            if isinstance(v, list):
                v = ",".join(map(str, v))
            pfams.loc[:, k] = v
    return pfams


def parse_ec(polymer_entity: dict[str, Any]) -> pd.DataFrame:
    """Extract PFAM (protein family) annotations from a polymer entity."""
    try:
        ec = pd.DataFrame(
            polymer_entity["rcsb_polymer_entity"]["rcsb_enzyme_class_combined"]
        )
    except Exception:
        ec = pd.DataFrame()
    if not ec.empty:
        for k, v in polymer_entity["rcsb_polymer_entity_container_identifiers"].items():
            if isinstance(v, list):
                v = ",".join(map(str, v))
            ec.loc[:, k] = v
    return ec


def parse_polymer_entity_instance(
    polymer_entity_instance: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract polymer entity instance features and annotations.

    Takes an instance of a polymer entity from a data entry and converts to
    dataframes corresponding to the `rcsb_polymer_instance_annotation` and `rcsb_polymer_instance_feature`
    keys. Also adds entity information to the dataframe containing identifiers like asym_id, auth_asym_id, entity_id.

    Parameters
    ----------
    polymer_entity_instance : dict[str, Any]
        The polymer entity instance coming from a polymer entity.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the annotation data and feature data for the entity instance.

    """
    entity_info = polymer_entity_instance[
        "rcsb_polymer_entity_instance_container_identifiers"
    ]
    normalized_info = cast_entity_info_lists(entity_info)
    parsed_tracks: dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
    parse_keys = ["annotation", "feature"]
    for key_suffix in parse_keys:
        data = polymer_entity_instance[f"rcsb_polymer_instance_{key_suffix}"]
        data = pd.DataFrame(data)
        data = add_entity_info(data, normalized_info)
        parsed_tracks[key_suffix] = data
    return parsed_tracks["annotation"], parsed_tracks["feature"]


def cast_entity_info_lists(
    entity_info: dict[str, str | int | float | list[str | int | float]],
) -> dict[str, str | int | float]:
    converted: dict[str, str | int | float] = {}
    for k, v in entity_info.items():
        if isinstance(v, list):
            v = ",".join(map(str, v))
        converted[k] = v
    return converted


def add_entity_info(
    df: pd.DataFrame,
    entity_info: dict[str, str | int | float],
) -> pd.DataFrame:
    if df.empty:
        return df
    for k, v in entity_info.items():
        df.loc[:, k] = v
    return df


def parse_annotation_data(
    data_json: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pdb_id = data_json.stem
    with open(data_json, "r") as f:
        data = json.load(f)
    if not data.get("data"):
        if data.get("status") == "failed":
            raise ValueError(f"{data_json} query failed to fetch, nothing to parse!")
        else:
            # We know that the query ran and there was no data in the API.
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pfam_dfs = []
    annotation_dfs = []
    feature_dfs = []
    ec_dfs = []
    for polymer_entity in data["data"]["entry"]["polymer_entities"]:
        pfams = parse_pfam(polymer_entity)
        if not pfams.empty:
            pfam_dfs.append(pfams)
        ec = parse_ec(polymer_entity)
        if not ec.empty:
            ec_dfs.append(ec)
        for polymer_entity_instance in polymer_entity["polymer_entity_instances"]:
            annot, feats = parse_polymer_entity_instance(polymer_entity_instance)
            if not annot.empty:
                annotation_dfs.append(annot)
            if not feats.empty:
                feature_dfs.append(feats)
    if len(feature_dfs):
        feature_df = pd.concat(feature_dfs).reset_index(drop=True)
        feature_df["pdb_id"] = pdb_id
    else:
        feature_df = pd.DataFrame()
    if len(annotation_dfs):
        annotation_df = pd.concat(annotation_dfs).reset_index(drop=True)
        annotation_df["pdb_id"] = pdb_id
    else:
        annotation_df = pd.DataFrame()

    if len(pfam_dfs):
        pfam_df = pd.concat(pfam_dfs).reset_index(drop=True)
        pfam_df["pdb_id"] = pdb_id
    else:
        pfam_df = pd.DataFrame()

    if len(ec_dfs):
        ec_df = pd.concat(ec_dfs).reset_index(drop=True)
        ec_df["pdb_id"] = pdb_id
    else:
        ec_df = pd.DataFrame()
    return pfam_df, feature_df, annotation_df, ec_df


def csv_format_pfam(pfam_df: pd.DataFrame, pdb_id: str) -> pd.DataFrame:
    if not pfam_df.empty:
        try:
            pfam_df = extract_feature_positions(pfam_df)
        except Exception as e:
            LOG.error(f"Failed to format {pdb_id} pfam DataFrame.")
    return pfam_df


def csv_format_features(feature_df: pd.DataFrame, pdb_id: str) -> pd.DataFrame:
    if not feature_df.empty:
        try:
            feature_df = cast_incompatible_feature_columns(feature_df)
        except Exception as e:
            LOG.error(f"Failed to format {pdb_id} feature DataFrame.")
    return feature_df


def csv_format_annotations(annotation_df: pd.DataFrame, pdb_id: str) -> pd.DataFrame:
    if not annotation_df.empty:
        try:
            annotation_df = cast_incompatible_annotation_columns(annotation_df)
        except Exception as e:
            LOG.error(f"Failed to format {pdb_id} annotation DataFrame.")
    return annotation_df


def safe_fetch_entry_annotations(
    pdb_id: str, pinder_dir: Path, use_cache: bool = True
) -> None:
    annotation_fp = pinder_dir / "rcsb_annotations"
    pfam_fp = annotation_fp / "pfam"
    ec_fp = annotation_fp / "enzyme_classification"
    feat_fp = annotation_fp / "features"
    annot_fp = annotation_fp / "annotations"
    json_fp = annotation_fp / "query_data"
    try:
        data_json = json_fp / f"{pdb_id}.json"
        fetch_entry_annotations(pdb_id, data_json, use_cache=use_cache)
        pfam_df, feature_df, annotation_df, ec_df = parse_annotation_data(data_json)
        # For empty dataframes, we write empty csv files to use for checkpointing to
        # indicate the query ran, but there were no matching annotations.
        # For non-empty df, we attempt to extract relevant data from columns containing
        # complex types. The original values are cast to json strings and preserved.
        # In case an exception is raised, we try to return existing dataframe anyways.
        # If an exception is raised during csv write, we know that the
        # schemas have changed or an unseen edge-case encountered.
        pfam_df = csv_format_pfam(pfam_df, pdb_id)
        pfam_df.to_csv(pfam_fp / f"{pdb_id}.csv.gz", index=False)
        ec_df.to_csv(ec_fp / f"{pdb_id}.csv.gz", index=False)
        feature_df = csv_format_features(feature_df, pdb_id)
        feature_df.to_csv(feat_fp / f"{pdb_id}.csv.gz", index=False)
        annotation_df = csv_format_annotations(annotation_df, pdb_id)
        annotation_df.to_csv(annot_fp / f"{pdb_id}.csv.gz", index=False)
    except Exception as e:
        LOG.error(f"Failed to fetch annotations for {pdb_id}: {e}")


def populate_rcsb_annotations(
    pinder_dir: Path,
    pdb_ids: list[str] | None = None,
    max_workers: int | None = None,
    use_cache: bool = True,
    parallel: bool = True,
) -> None:
    if not pdb_ids:
        index = pd.read_csv(pinder_dir / "index.1.csv.gz")
        pdb_ids = list(set(index.pdb_id))

    annotation_fp = pinder_dir / "rcsb_annotations"
    annotation_fp.mkdir(exist_ok=True, parents=True)
    pfam_fp = annotation_fp / "pfam"
    feat_fp = annotation_fp / "features"
    annot_fp = annotation_fp / "annotations"
    ec_fp = annotation_fp / "enzyme_classification"
    json_fp = annotation_fp / "query_data"
    for fp in [pfam_fp, feat_fp, annot_fp, ec_fp, json_fp]:
        fp.mkdir(exist_ok=True, parents=True)

    process_starmap(
        safe_fetch_entry_annotations,
        zip(pdb_ids, repeat(pinder_dir), repeat(use_cache)),
        parallel=parallel,
        max_workers=max_workers,
    )


def split_annotation_types(dfs: list[pd.DataFrame]) -> dict[str, list[pd.DataFrame]]:
    """Split list of annotation dataframes into their respective annotation categories.

    Reduces final dataframe size and used to write each category to its own file on disk.

    """
    group_dfs: dict[str, list[pd.DataFrame]] = {k: [] for k in TYPE_PATTERNS.keys()}
    known_types: set[str] = set()
    for k, v in TYPE_PATTERNS.items():
        known_types = known_types.union(set(v))
    for df in tqdm(dfs):
        if "type" not in df.columns:
            continue
        df_types = set(df["type"])
        for k, patterns in TYPE_PATTERNS.items():
            if k == "other":
                add_patterns = df_types - known_types
            else:
                add_patterns = {match for match in df_types if k in match}
            check_patterns = set(patterns).union(add_patterns)
            df_sub = df[df["type"].isin(check_patterns)].reset_index(drop=True)
            if not df_sub.empty:
                group_dfs[k].append(df_sub)
    return group_dfs


def extract_feature_positions(df: pd.DataFrame) -> pd.DataFrame:
    if "feature_positions" not in df.columns:
        return df
    max_entry_feats = max(
        [len(f) if isinstance(f, list) else 0 for f in df.feature_positions]
    )
    if max_entry_feats > 1:
        LOG.debug(
            f"Data frame contains some entries with more than one feature position! ({max_entry_feats})"
            " Only keeping the first feature position!"
        )
    # Take only the first position so we can extract single start+end seq ID
    df["beg_seq_id"] = df.feature_positions.apply(
        lambda x: x[0].get("beg_seq_id") if isinstance(x, list) else x
    )
    df["end_seq_id"] = df.feature_positions.apply(
        lambda x: x[0].get("end_seq_id") if isinstance(x, list) else x
    )
    # Cast feature positions column to json string
    df.loc[:, "feature_positions"] = df.feature_positions.apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    return df


def cast_incompatible_annotation_columns(annot_df: pd.DataFrame) -> pd.DataFrame:
    if "annotation_lineage" not in annot_df.columns:
        return annot_df
    annot_df.loc[:, "annotation_lineage"] = annot_df.annotation_lineage.apply(
        json.dumps
    )
    return annot_df


def cast_incompatible_feature_columns(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = extract_feature_positions(feat_df)
    if "additional_properties" not in feat_df.columns:
        return feat_df
    # list of dicts
    feat_df.loc[:, "additional_properties"] = feat_df.additional_properties.apply(
        json.dumps
    )
    return feat_df


def collate_csvs(
    csvs: list[Path],
    output_csv: Path,
    max_workers: int | None = None,
    use_cache: bool = True,
) -> None:
    if output_csv.is_file() and use_cache:
        LOG.info(f"Skipping collation, {output_csv} exists...")
        return
    dfs = parallel_read_csvs(csvs, max_workers=max_workers)
    if dfs:
        joined = pd.concat(dfs).reset_index(drop=True)
        joined.to_csv(output_csv, index=False)
        del joined


def collect_csvs_by_group(
    output_dir: Path,
    csv_files: list[Path],
    entity_prefix: str,
    max_workers: int | None = None,
    use_cache: bool = True,
) -> None:
    dfs = parallel_read_csvs(csv_files, max_workers=max_workers)
    if not dfs:
        LOG.warning("{entity_prefix} csv files were all empty...")
        return None
    groups = split_annotation_types(dfs)
    for annot_type, dfs in groups.items():
        output_csv = output_dir / f"{entity_prefix}_{annot_type}.csv.gz"
        if output_csv.is_file() and use_cache:
            LOG.info(f"Skipping collation, {output_csv} exists...")
            continue
        # Write separate csv.gz files per category of annotation/feature types
        if len(dfs):
            LOG.info(f"Collating {len(dfs)} {annot_type} {entity_prefix}")
            annot_df = pd.concat(dfs).reset_index(drop=True)
            LOG.info(f"Writing {annot_type} csv")
            annot_df.to_csv(output_csv, index=False)
            del annot_df
            del dfs


def collect_rcsb_annotations(
    pinder_dir: Path,
    max_workers: int | None = None,
    use_cache: bool = True,
) -> None:
    annotation_fp = pinder_dir / "rcsb_annotations"
    pfam_fp = annotation_fp / "pfam"
    feat_fp = annotation_fp / "features"
    annot_fp = annotation_fp / "annotations"
    ec_fp = annotation_fp / "enzyme_classification"
    collate_csvs(
        list(ec_fp.glob("*.csv.gz")),
        annotation_fp / "enzyme_classification.csv.gz",
        max_workers=max_workers,
        use_cache=use_cache,
    )
    collate_csvs(
        list(pfam_fp.glob("*.csv.gz")),
        annotation_fp / "pfam.csv.gz",
        max_workers=max_workers,
        use_cache=use_cache,
    )

    annot_csvs = list(annot_fp.glob("*.csv.gz"))
    collect_csvs_by_group(
        annotation_fp, annot_csvs, "annotations", max_workers, use_cache=use_cache
    )

    feat_csvs = list(feat_fp.glob("*.csv.gz"))
    collect_csvs_by_group(
        annotation_fp, feat_csvs, "features", max_workers, use_cache=use_cache
    )
