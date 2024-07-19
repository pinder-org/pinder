from __future__ import annotations

import re
from itertools import repeat
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import plotly.io as pio

from pinder.core.index.utils import get_index
from pinder.core.utils import setup_logger
from pinder.core.utils.process import process_starmap
from pinder.core.index.utils import (
    get_pinder_location,
    IndexEntry,
)
from pinder.core.structure.atoms import atom_array_from_pdb_file
from pinder.core.structure.models import DatasetName, MonomerName
from pinder.eval.dockq.biotite_dockq import BiotiteDockQ

pio.templates.default = "plotly_dark"

log = setup_logger(__name__)

subsets = [
    "pinder_xl",
    "pinder_s",
    "pinder_af2",
]

dataset_root = get_pinder_location()


class InvalidChainsError(Exception):
    def __init__(self, pdb_file: Path, chains: list[str]) -> None:
        self.pdb_file = pdb_file
        self.chains = chains


class TooFewChainsError(InvalidChainsError):
    def __str__(self) -> str:
        chain_str = "_".join(self.chains)
        msg = (
            f"PDB {self.pdb_file.stem} has too few chains for running eval:"
            f"{len(self.chains)} ({chain_str})"
            "There should be two chains: R (receptor) and L (ligand)."
        )
        return msg


class ExpectedChainsNotFoundError(InvalidChainsError):
    def __str__(self) -> str:
        chain_str = "_".join(self.chains)
        msg = (
            f"PDB {self.pdb_file.stem} is missing expected chains for running eval:"
            f"{chain_str} != R_L."
            "There should be two chains: R (receptor) and L (ligand)."
        )
        return msg


def _glob_dirs(parent: Path) -> list[Path]:
    return [d for d in parent.glob("*") if d.is_dir()]


def validate_system_id(system: Path, pinder_ids: set[str]) -> None:
    if system.stem not in pinder_ids:
        raise ValueError(
            f"Invalid directory structure! {system.stem} is not a pinder ID"
        )


def get_valid_eval_systems(eval_dir: Path) -> list[dict[str, Path | str]]:
    # Check that the eval_dir contains one of the acceptable formats
    # If it doesn't, it will be rejected
    pindex = get_index()
    pinder_ids = set(pindex.id)
    eval_methods_or_subset = _glob_dirs(eval_dir)
    eval_systems = []
    for subdir in eval_methods_or_subset:
        if subdir.stem in pinder_ids:
            # Subdir is itself a pinder_id
            eval_system: dict[str, Path | str] = {
                "system": subdir,
                "method_name": subdir.parent.stem,
            }
            eval_systems.append(eval_system)
        elif subdir.stem in subsets:
            for subset_subdir in _glob_dirs(subdir):
                if subset_subdir.stem in pinder_ids:
                    # Assumption is method_name contains pinder subset dirs
                    eval_system = {
                        "system": subset_subdir,
                        "method_name": subdir.parent.stem,
                    }
                    eval_systems.append(eval_system)
                    continue
                for system_subddir in _glob_dirs(subset_subdir):
                    try:
                        validate_system_id(system_subddir, pinder_ids)
                    except ValueError as e:
                        log.warning(f"{system_subddir.stem} not a valid ID, skipping")
                        continue
                    # Assumption is pinder subset dir contains method_name dirs
                    eval_system = {
                        "system": system_subddir,
                        "method_name": subset_subdir.stem,
                    }
                    eval_systems.append(eval_system)
        else:
            # Eval dir contains subdirectories of method names
            for system_subddir in _glob_dirs(subdir):
                try:
                    validate_system_id(system_subddir, pinder_ids)
                except ValueError as e:
                    log.warning(f"{system_subddir.stem} not a valid ID, skipping")
                    continue
                eval_system = {
                    "system": system_subddir,
                    "method_name": subdir.stem,
                }
                eval_systems.append(eval_system)
    if not eval_systems:
        raise FileNotFoundError(
            "Invalid directory structure! Unable to locate methods & systems!"
        )
    return eval_systems


class SubsetIds(TypedDict):
    count: int
    ids: set[str]


def get_expected_counts(
    pindex: pd.DataFrame, subsets: list[str]
) -> dict[tuple[str, str], SubsetIds]:
    expected_counts: dict[tuple[str, str], SubsetIds] = {}
    for subset in subsets:
        for monomer_name in ["holo", "apo", "predicted"]:
            subset_mono_ids = set(
                pindex.query(f"{subset} and ({monomer_name}_R and {monomer_name}_L)").id
            )
            n_ids: int = len(subset_mono_ids)
            subset_expected: SubsetIds = {
                "count": n_ids,
                "ids": subset_mono_ids,
            }
            expected_counts[(subset, monomer_name)] = subset_expected
    return expected_counts


def summarize_valid_systems(
    eval_dir: Path,
    confirm: bool = True,
    custom_index: str = "",
) -> None:
    eval_systems = get_valid_eval_systems(eval_dir)
    pindex = get_index(csv_name=custom_index) if custom_index != "" else get_index()
    counts_lst = []
    for eval_system in eval_systems:
        system = Path(eval_system["system"])
        method_name = eval_system["method_name"]
        row = pindex.query(f'id == "{system.stem}"')
        entry = IndexEntry(**row.to_dict(orient="records")[0])
        subdirs = _glob_dirs(Path(system))
        if "_decoys" in subdirs[0].stem:
            decoy_subdirs = subdirs
        else:
            decoy_subdirs = []
            for d in subdirs:
                decoy_subdirs.append(d / "models")

        for decoy_dir in decoy_subdirs:
            models = list(decoy_dir.glob("*.pdb"))
            # add which monomer kind and subset this is in
            if decoy_dir.stem == "models":
                monomer_name = decoy_dir.parent.stem
            else:
                monomer_name = decoy_dir.stem.split("_decoys")[0]

            # validate presence of two chains named R and L (only one decoy is checked)
            if len(models):
                test_decoy = models[0]
                decoy_chains = set(atom_array_from_pdb_file(test_decoy).chain_id)
                if len(decoy_chains) < 2:
                    raise TooFewChainsError(test_decoy, list(decoy_chains))
                if decoy_chains != {"R", "L"}:
                    raise ExpectedChainsNotFoundError(test_decoy, list(decoy_chains))
            counts_lst.append(
                {
                    "id": entry.id,
                    "method_name": method_name,
                    "pinder_xl": entry.pinder_xl,
                    "pinder_s": entry.pinder_s,
                    "pinder_af2": entry.pinder_af2,
                    "monomer_name": monomer_name,
                    "n_decoys": len(models),
                }
            )

    expected_counts = get_expected_counts(pindex, subsets)
    counts: pd.DataFrame = pd.DataFrame(counts_lst)
    for (subset, monomer), expected in expected_counts.items():
        matching = counts.query(f"{subset} and monomer_name == '{monomer}'")
        matching = matching[matching["id"].isin(expected["ids"])].reset_index(drop=True)
        actual = len(set(matching.id))
        n_expected = expected["count"]
        missing = n_expected - actual
        missing_ids = set(expected["ids"]) - set(matching.id)
        if actual < n_expected:
            lab = f"{subset} {monomer}"
            log.error(f"Method is missing {missing} systems from {lab}")
            print("#" * 100)
            print("The following systems are missing:")
            print("\n".join(missing_ids))
            print("#" * 100)

            print(
                "If you proceed with submission, these systems will receive metrics "
                f"that will penalize the method in {lab} evaluation."
            )
            if confirm:
                proceed = input("Do you wish to proceed? (y/n)\n")
                if proceed == "" or proceed.lower()[0] != "y":
                    raise RuntimeError("Method rejected! Missing systems!")


def download_entry(entry: IndexEntry) -> None:
    from pinder.core import PinderSystem

    # Will download single entry if not present
    PinderSystem(entry.id)


def get_eval_metrics(
    eval_system: dict[str, Path | str],
    parallel_io: bool = False,
    custom_index: str = "",
) -> pd.DataFrame:
    pindex = get_index(csv_name=custom_index) if custom_index != "" else get_index()
    system = Path(eval_system["system"])
    method_name = str(eval_system["method_name"])
    row = pindex.query(f'id == "{system.stem}"')
    entry = IndexEntry(**row.to_dict(orient="records")[0])
    native = dataset_root / f"pdbs/{entry.pinder_pdb}"
    if not native.is_file():
        native.parent.mkdir(exist_ok=True, parents=True)
        download_entry(entry)

    subdirs = _glob_dirs(Path(system))
    if "_decoys" in subdirs[0].stem:
        decoy_subdirs = subdirs
    else:
        decoy_subdirs = []
        for d in subdirs:
            decoy_subdirs.append(d / "models")

    all_metrics = []
    for decoy_dir in decoy_subdirs:
        print("#" * len(str(decoy_dir)))
        print(decoy_dir)
        print("#" * len(str(decoy_dir)))
        models = list(decoy_dir.glob("*.pdb"))
        if not models:
            log.error(f"{decoy_dir} is missing decoys!")
            continue
        default_combos = ("R", "L", "R", "L")
        inverse_default = ("L", "R", "L", "R")
        dockgpt_combos = ("R", "L", "A", "B")
        inverse_dockgpt = ("L", "R", "B", "A")

        # Check chains in decoy
        ab_chains = False
        try:
            test_decoy = models[0]
            decoy_chains = set(atom_array_from_pdb_file(test_decoy).chain_id)
            if "A" in decoy_chains:
                ab_chains = True
        except Exception as e:
            print(f"Unable to extract chains from decoy {decoy_dir}!")

        ab_method = any(
            alt_method in method_name for alt_method in ["dockgpt", "af2mm"]
        )
        if ab_method or ab_chains:
            chain_permutes = [dockgpt_combos]
            if entry.homodimer:
                chain_permutes.append(inverse_dockgpt)
        else:
            chain_permutes = [default_combos]
            if entry.homodimer:
                chain_permutes.append(inverse_default)
        permute_metrics = []
        for nat_R_chain, nat_L_chain, mod_R_chain, mod_L_chain in chain_permutes:
            try:
                bdq = BiotiteDockQ(
                    native,
                    models,
                    backbone_definition="dockq",
                    native_receptor_chain=[nat_R_chain],
                    native_ligand_chain=[nat_L_chain],
                    decoy_receptor_chain=[mod_R_chain],
                    decoy_ligand_chain=[mod_L_chain],
                    parallel_io=parallel_io,
                )
                metrics = bdq.calculate()
                metrics.loc[:, "receptor_chain"] = mod_R_chain
                metrics.loc[:, "ligand_chain"] = mod_L_chain
                metrics.loc[:, "native_receptor_chain"] = nat_R_chain
                metrics.loc[:, "native_ligand_chain"] = nat_L_chain
            except Exception as e:
                # What if R + L got flipped?
                bdq = BiotiteDockQ(
                    native,
                    models,
                    backbone_definition="dockq",
                    native_receptor_chain=[nat_R_chain],
                    native_ligand_chain=[nat_L_chain],
                    decoy_receptor_chain=[mod_L_chain],
                    decoy_ligand_chain=[mod_R_chain],
                    parallel_io=parallel_io,
                )
                metrics = bdq.calculate()
                metrics.loc[:, "receptor_chain"] = mod_L_chain
                metrics.loc[:, "ligand_chain"] = mod_R_chain
                metrics.loc[:, "native_receptor_chain"] = nat_R_chain
                metrics.loc[:, "native_ligand_chain"] = nat_L_chain
            permute_metrics.append(metrics)
        permute_df: pd.DataFrame = pd.concat(permute_metrics).reset_index(drop=True)
        # Get best permutation per model (R/L), only multiple if homodimer
        decoy_metrics = (
            permute_df.sort_values("DockQ", ascending=False)
            .drop_duplicates("model_name", keep="first")
            .reset_index(drop=True)
        )
        # add which monomer kind and subset this is in
        if decoy_dir.stem == "models":
            monomer_name = decoy_dir.parent.stem
        else:
            monomer_name = decoy_dir.stem.split("_decoys")[0]

        decoy_metrics.loc[:, "monomer_name"] = monomer_name
        decoy_metrics.loc[:, "method_name"] = method_name
        all_metrics.append(decoy_metrics)
    if len(all_metrics):
        metrics_df: pd.DataFrame = pd.concat(all_metrics).reset_index(drop=True)
        metrics_df.loc[:, "id"] = entry.id
        metrics_df.loc[:, "pinder_xl"] = entry.pinder_xl
        metrics_df.loc[:, "pinder_s"] = entry.pinder_s
        metrics_df.loc[:, "pinder_af2"] = entry.pinder_af2
        return metrics_df


def safe_get_eval_metrics(
    system: dict[str, Path | str],
    parallel_io: bool = False,
    custom_index: str = "",
) -> pd.DataFrame | None:
    try:
        return get_eval_metrics(system, parallel_io, custom_index)
    except Exception as e:
        if isinstance(system, dict):
            system_path = system.get("system", Path("."))
            method_name = system["method_name"]
            id = Path(system_path).stem
        else:
            id = method_name = system
        log.error(f"Failed to get metrics for {method_name}: {id}!")
        log.error(str(e))
        return None


def get_eval_metrics_all_methods(
    eval_dir: Path,
    parallel: bool = True,
    custom_index: str = "",
    max_workers: int | None = None,
) -> pd.DataFrame:
    eval_systems = get_valid_eval_systems(eval_dir)
    parallel_io = not parallel
    all_metrics = process_starmap(
        safe_get_eval_metrics,
        zip(eval_systems, repeat(parallel_io), repeat(custom_index)),
        parallel=parallel,
        max_workers=max_workers,
    )
    all_metrics = pd.concat(
        [df for df in all_metrics if isinstance(df, pd.DataFrame)]
    ).reset_index(drop=True)
    return all_metrics


def get_method_eval_metrics(
    run_dirs: list[Path],
    dataset_name: DatasetName,
    monomer_name: MonomerName,
    method_name: str | None = None,
    parallel: bool = True,
    custom_index: str = "",
    max_workers: int | None = None,
) -> pd.DataFrame:
    parallel_io = not parallel
    method_metrics = process_starmap(
        get_eval_metrics,
        zip(run_dirs, repeat(parallel_io), repeat(custom_index)),
        parallel=parallel,
        max_workers=max_workers,
    )
    method_df: pd.DataFrame = pd.concat(
        [df for df in method_metrics if isinstance(df, pd.DataFrame)]
    ).reset_index(drop=True)
    if not method_name:
        method_name = run_dirs[0].parent.stem
    method_df.loc[:, "method_name"] = method_name
    method_df.loc[:, "dataset_name"] = dataset_name
    method_df.loc[:, "monomer_name"] = monomer_name
    return method_df


def add_pinder_set(
    metrics: pd.DataFrame,
    allow_missing: bool = False,
    custom_index: str = "",
) -> pd.DataFrame | None:
    pindex = get_index(csv_name=custom_index) if custom_index != "" else get_index()
    expected_counts = get_expected_counts(pindex, subsets)
    method_counts = (
        metrics[["id", "method_name", "monomer_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    patched_metrics = []
    for method_name, counts in method_counts.groupby("method_name"):
        counts.reset_index(drop=True, inplace=True)
        method_metrics = metrics.query(f"method_name == '{method_name}'").reset_index(
            drop=True
        )
        for (subset, monomer), expected in expected_counts.items():
            matching = counts.query(f"monomer_name == '{monomer}'").copy()
            matching = matching[matching["id"].isin(expected["ids"])].reset_index(
                drop=True
            )
            actual = len(set(matching.id))
            n_expected = expected["count"]
            missing = n_expected - actual
            expected_ids = expected["ids"]
            missing_ids = set(expected_ids) - set(matching.id)
            monomer_metrics = method_metrics.query(
                f"monomer_name == '{monomer}'"
            ).reset_index(drop=True)
            monomer_metrics = monomer_metrics[
                monomer_metrics["id"].isin(expected_ids)
            ].reset_index(drop=True)
            if monomer_metrics.shape[0]:
                monomer_metrics.loc[:, "pinder_set"] = subset
            if actual < n_expected:
                lab = f"{subset} {monomer}"
                log.error(f"{method_name} is missing {missing} systems from {lab}")
                if not allow_missing:
                    penalty_metrics = pd.DataFrame(
                        [
                            get_missing_system_penalty_metrics(
                                id, monomer, method_name, subset
                            )
                            for id in missing_ids
                        ]
                    )
                    penalty_metrics = pd.merge(
                        penalty_metrics,
                        pindex[["id", "pinder_xl", "pinder_s", "pinder_af2"]],
                    )
                    monomer_metrics = pd.concat(
                        [monomer_metrics, penalty_metrics]
                    ).reset_index(drop=True)
            patched_metrics.append(monomer_metrics)

    if not patched_metrics:
        raise ValueError("No valid systems from pinder subsets present in metrics!")
    patched_metrics = pd.concat(patched_metrics).reset_index(drop=True)
    return patched_metrics


def get_missing_system_penalty_metrics(
    id: str,
    monomer_name: str,
    method_name: str,
    pinder_set: str,
    penalty_rmsd: float = 100.0,
) -> pd.DataFrame:
    penalty_record = {
        "model_name": "missing_decoy_1",
        "native_name": id,
        "model_folder": f"{monomer_name}_decoys",
        "iRMS": penalty_rmsd,
        "LRMS": penalty_rmsd,
        "Fnat": 0.0,
        "DockQ": 0.0,
        "CAPRI": "Incorrect",
        "receptor_chain": "R",
        "ligand_chain": "L",
        "monomer_name": monomer_name,
        "method_name": method_name,
        "id": id,
        "rank": 1,
        "CAPRI_rank": 0,
        "decoy": "missing_decoy_1.pdb",
        "pinder_set": pinder_set,
    }
    return penalty_record


class CapriClass:
    def __getitem__(self, key: str | int) -> str | int:
        capri_ranks = {
            "Undef": 0,
            "Incorrect": 0,
            "Acceptable": 1,
            "Medium": 2,
            "High": 3,
        }
        if isinstance(key, str):
            return capri_ranks.get(key, 0)
        else:
            inverse = {v: k for k, v in capri_ranks.items()}
            return inverse.get(key, "Incorrect")


class MethodMetrics:
    def __init__(
        self,
        eval_dir: Path,
        parallel: bool = True,
        allow_missing_systems: bool = False,
        custom_index: str = "",
        max_workers: int | None = None,
    ) -> None:
        """Evaluate method using reference-based metrics.

        Parameters
        ----------
        eval_dir : Path
            Path containing one of two valid formats for eval directories.

        Returns
        -------
        None

        """

        self.eval_dir = eval_dir
        self.parallel = parallel
        self.allow_missing = allow_missing_systems
        self.custom_index = custom_index
        self.oracle_agg = {
            "CAPRI_rank": "max",
            "DockQ": "max",
            "Fnat": "max",
            "LRMS": "min",
            "iRMS": "min",
        }
        self.display_columns = {
            "iRMS": "Interface RMSD (Å)",
            "LRMS": "Ligand RMSD(Å)",
            "method_name": "Method",
            "pinder_set": "Dataset",
            "monomer_name": "Monomer",
            "CAPRI_hit_rate": "CAPRI Hit Rate (%)",
        }
        self.metadata_cols = ["method_name", "monomer_name", "pinder_set"]
        self._metrics = None
        self.rank_regex = r"\d+(?=\D*$)"
        self.max_workers = max_workers

    @property
    def metrics(self) -> pd.DataFrame:
        if not isinstance(self._metrics, pd.DataFrame):
            self._metrics = self.get_metrics()
        return self._metrics

    @staticmethod
    def add_rank_column(metrics: pd.DataFrame, rank_regex: str) -> pd.DataFrame:
        ranks = []
        for decoy_name in list(metrics.model_name):
            m = re.findall(rank_regex, decoy_name)
            assert m, f"`{decoy_name}` file name is unparsable"
            try:
                rank = int(m[-1])
            except ValueError:
                raise ValueError(
                    f"Unable to infer decoy rank from decoy name! {decoy_name}"
                    f"Note the regex pattern used to infer is {rank_regex}"
                )
            ranks.append(rank)
        metrics.loc[:, "rank"] = ranks
        return metrics

    def get_metrics(self) -> pd.DataFrame:
        metrics = get_eval_metrics_all_methods(
            self.eval_dir,
            parallel=self.parallel,
            custom_index=self.custom_index,
            max_workers=self.max_workers,
        )
        metrics = self.add_rank_column(metrics, self.rank_regex)
        cc = CapriClass()
        metrics.loc[:, "CAPRI_rank"] = metrics.CAPRI.apply(lambda x: cc[x])
        metrics.loc[:, "decoy"] = metrics.model_name + ".pdb"
        # BiotiteDockQ columns that are superceded by MethodMetrics
        metrics.drop(["system", "method"], axis=1, inplace=True)
        metrics = add_pinder_set(
            metrics,
            allow_missing=self.allow_missing,
            custom_index=self.custom_index,
        )
        if not metrics.shape[0]:
            raise ValueError("Insufficient systems to calculate leaderboard!")
        return metrics

    def oracle_median_summary(self) -> pd.DataFrame:
        oracle = self.system_oracle()
        groups = self.metadata_cols
        metric_cols = set(self.oracle_agg.keys()).intersection(set(oracle.columns))
        agg = {metric: "median" for metric in metric_cols}
        summary = self.metrics.groupby(groups, as_index=False).agg(agg)
        hit_rate = (oracle.CAPRI_rank > 0).sum() / oracle.shape[0] * 100
        summary.loc[:, "CAPRI_hit_rate"] = hit_rate
        return summary.rename(self.display_columns, axis=1).round(2)

    def get_leaderboard_entry(self) -> pd.DataFrame:
        top1 = self.top_k(1)
        top5 = self.top_k(5)
        q25 = self.top_percentile(q=0.25)
        q50 = self.top_percentile(q=0.50)
        system_oracle = self.system_oracle()

        oracle_hit_rates = self.get_hit_rates(system_oracle, "oracle")
        top1_hit_rates = self.get_hit_rates(self.top_k_oracle(top1), "Max(Top 1)")
        top5_hit_rates = self.get_hit_rates(self.top_k_oracle(top5), "Max(Top 5)")

        # Get median metrics of oracle across all systems
        oracle_med = self.median_oracle(system_oracle)
        # Get median metrics of the max(top K) across all systems
        top5_med = self.median_oracle(top5)
        top1_med = self.median_oracle(top1)
        # Get median metrics of the max(top X percentile) across all systems
        q25_med = self.median_oracle(q25)
        q50_med = self.median_oracle(q50)

        top1_med.rename(
            {
                "iRMS": "I-RMSD Median(Top 1)",
                "LRMS": "L-RMSD Median(Top 1)",
                "Fnat": "Fnat Median(Top 1)",
                "DockQ": "DockQ Median(Top 1)",
            },
            axis=1,
            inplace=True,
        )

        top5_med.rename(
            {
                "iRMS": "I-RMSD Median(Max(Top 5))",
                "LRMS": "L-RMSD Median(Max(Top 5))",
                "Fnat": "Fnat Median(Max(Top 5))",
                "DockQ": "DockQ Median(Max(Top 5))",
            },
            axis=1,
            inplace=True,
        )

        q25_med.rename(
            {
                "iRMS": "I-RMSD Median(Max(25%))",
                "LRMS": "L-RMSD Median(Max(25%))",
                "Fnat": "Fnat Median(Max(25%))",
                "DockQ": "DockQ Median(Max(25%))",
            },
            axis=1,
            inplace=True,
        )
        q50_med.rename(
            {
                "iRMS": "I-RMSD Median(Max(50%))",
                "LRMS": "L-RMSD Median(Max(50%))",
                "Fnat": "Fnat Median(Max(50%))",
                "DockQ": "DockQ Median(Max(50%))",
            },
            axis=1,
            inplace=True,
        )

        oracle_med.rename(
            {
                "iRMS": "I-RMSD Median(Max(100%)) (oracle)",
                "LRMS": "L-RMSD Median(Max(100%)) (oracle)",
                "Fnat": "Fnat Median(Max(100%)) (oracle)",
                "DockQ": "DockQ Median(Max(100%)) (oracle)",
            },
            axis=1,
            inplace=True,
        )

        leaderboard = pd.merge(oracle_hit_rates, top1_hit_rates, how="left")
        leaderboard = pd.merge(leaderboard, top5_hit_rates, how="left")
        leaderboard = pd.merge(
            leaderboard, top1_med.drop("CAPRI_rank", axis=1), how="left"
        )
        leaderboard = pd.merge(
            leaderboard, top5_med.drop("CAPRI_rank", axis=1), how="left"
        )
        leaderboard = pd.merge(
            leaderboard, q25_med.drop("CAPRI_rank", axis=1), how="left"
        )
        leaderboard = pd.merge(
            leaderboard, q50_med.drop("CAPRI_rank", axis=1), how="left"
        )
        leaderboard = pd.merge(
            leaderboard, oracle_med.drop("CAPRI_rank", axis=1), how="left"
        )
        leaderboard.rename(self.display_columns, axis=1, inplace=True)
        col_order = [
            "Method",
            "Dataset",
            "Monomer",
            "I-RMSD Median(Top 1)",
            "I-RMSD Median(Max(Top 5))",
            "I-RMSD Median(Max(25%))",
            "I-RMSD Median(Max(50%))",
            "I-RMSD Median(Max(100%)) (oracle)",
            "I-RMSD % ≤ 2Å (oracle)",
            "I-RMSD % ≤ 2Å (Max(Top 1))",
            "I-RMSD % ≤ 2Å (Max(Top 5))",
            "L-RMSD Median(Top 1)",
            "L-RMSD Median(Max(Top 5))",
            "L-RMSD Median(Max(25%))",
            "L-RMSD Median(Max(50%))",
            "L-RMSD Median(Max(100%)) (oracle)",
            "L-RMSD % ≤ 5Å (oracle)",
            "L-RMSD % ≤ 5Å (Max(Top 1))",
            "L-RMSD % ≤ 5Å (Max(Top 5))",
            "Fnat Median(Top 1)",
            "Fnat Median(Max(Top 5))",
            "Fnat Median(Max(25%))",
            "Fnat Median(Max(50%))",
            "Fnat Median(Max(100%)) (oracle)",
            "Fnat % ≥ 0.8 (oracle)",
            "Fnat % ≥ 0.8 (Max(Top 1))",
            "Fnat % ≥ 0.8 (Max(Top 5))",
            "DockQ Median(Top 1)",
            "DockQ Median(Max(Top 5))",
            "DockQ Median(Max(25%))",
            "DockQ Median(Max(50%))",
            "DockQ Median(Max(100%)) (oracle)",
            "DockQ % acceptable (oracle)",
            "DockQ % medium (oracle)",
            "DockQ % high (oracle)",
            "DockQ % acceptable (Max(Top 1))",
            "DockQ % medium (Max(Top 1))",
            "DockQ % high (Max(Top 1))",
            "DockQ % acceptable (Max(Top 5))",
            "DockQ % medium (Max(Top 5))",
            "DockQ % high (Max(Top 5))",
        ]
        for col in col_order:
            if col not in ["Method", "Dataset", "Monomer"]:
                leaderboard[col] = leaderboard[col].astype(float)
        return leaderboard[col_order].round(2)

    def median_oracle(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pandas agg for getting median of each metric
        med_agg = {metric: "median" for metric in self.oracle_agg}
        method_groups = self.metadata_cols
        sys_groups = method_groups + ["id"]
        # Get "oracle" (max) within top 5 (by rank) per system
        sys_max = df.groupby(sys_groups, as_index=False).agg(self.oracle_agg)
        # Get median metrics of the oracle across all systems in df
        method_med = sys_max.groupby(method_groups, as_index=False).agg(med_agg)
        return method_med

    def get_hit_rates(self, sys_data: pd.DataFrame, pose_set: str) -> pd.DataFrame:
        hit_rates = []
        for cols, df in sys_data.groupby(self.metadata_cols):
            n_systems = df.shape[0]
            irmsd_2A = df.query("iRMS <= 2").shape[0] / n_systems
            lrmsd_5A = df.query("LRMS <= 5").shape[0] / n_systems
            fnat_08 = df.query("Fnat >= 0.8").shape[0] / n_systems

            acceptable = df.query("CAPRI_rank >= 1").shape[0] / n_systems
            med = df.query("CAPRI_rank >= 2").shape[0] / n_systems
            high = df.query("CAPRI_rank >= 3").shape[0] / n_systems

            row = {c: df[c].values[0] for c in self.metadata_cols}
            row[f"I-RMSD % ≤ 2Å ({pose_set})"] = irmsd_2A * 100
            row[f"L-RMSD % ≤ 5Å ({pose_set})"] = lrmsd_5A * 100
            row[f"Fnat % ≥ 0.8 ({pose_set})"] = fnat_08 * 100
            row[f"DockQ % acceptable ({pose_set})"] = acceptable * 100
            row[f"DockQ % medium ({pose_set})"] = med * 100
            row[f"DockQ % high ({pose_set})"] = high * 100
            hit_rates.append(row)
        hit_rates = pd.DataFrame(hit_rates)
        return hit_rates

    def top_k(self, k: int) -> pd.DataFrame:
        groups = self.metadata_cols + ["id"]
        return (
            self.metrics.sort_values("rank")
            .groupby(groups, as_index=False)
            .head(k)
            .reset_index(drop=True)
        )

    def top_k_oracle(self, df: pd.DataFrame) -> pd.DataFrame:
        groups = self.metadata_cols + ["id"]
        return df.groupby(groups, as_index=False).agg(self.oracle_agg)

    def top_percentile(self, q: float) -> pd.DataFrame:
        groups = self.metadata_cols + ["id"]
        top_q = []
        for group_cols, df in self.metrics.groupby(groups):
            q_cut = df["rank"].quantile(q, interpolation="linear")
            df = df.query(f"rank <= {q_cut}").reset_index(drop=True)
            top_q.append(df)
        top_q = pd.concat(top_q).reset_index(drop=True)
        return top_q

    def system_oracle(self) -> pd.DataFrame:
        groups = self.metadata_cols + ["id"]
        oracle = self.metrics.groupby(groups, as_index=False).agg(self.oracle_agg)
        cc = CapriClass()
        oracle.loc[:, "CAPRI"] = oracle.CAPRI_rank.apply(lambda x: cc[x])
        return oracle

    def system_median(self) -> pd.DataFrame:
        groups = self.metadata_cols + ["id"]
        metric_aggs = {metric: "median" for metric in self.oracle_agg}
        return self.metrics.groupby(groups, as_index=False).agg(metric_aggs)

    def method_median(self) -> pd.DataFrame:
        groups = self.metadata_cols
        metric_aggs = {metric: "median" for metric in self.oracle_agg}
        return self.metrics.groupby(groups, as_index=False).agg(metric_aggs)

    def method_oracle(self) -> pd.DataFrame:
        groups = self.metadata_cols
        oracle = self.metrics.groupby(groups, as_index=False).agg(self.oracle_agg)
        return oracle

    def system_capri_hit_rate(self) -> pd.DataFrame:
        groups = self.metadata_cols + ["id"]
        hit_rate = (
            self.metrics.groupby(groups, as_index=False)["CAPRI_rank"]
            .apply(lambda c: (c > 0).sum() / len(c))
            .rename({"CAPRI_rank": "CAPRI_hit_rate"}, axis=1)
        )
        return hit_rate

    def median_capri_hit_rate(self) -> pd.DataFrame:
        return np.median(self.system_capri_hit_rate().CAPRI_hit_rate)

    def histogram_plot(
        self, metric_df: pd.DataFrame, metric: str | None = None
    ) -> None:
        metric_df.loc[:, "label"] = [
            f"{method}<br>{monomer}, {dataset}"
            for method, monomer, dataset in zip(
                *(metric_df[c] for c in self.metadata_cols)
            )
        ]
        if metric:
            fig = px.histogram(
                metric_df,
                x=metric,
                color="label",
                barmode="overlay",
            )
        else:
            value_vars = set(metric_df.columns).intersection(
                set(self.oracle_agg.keys())
            )
            long = metric_df.melt(
                id_vars=["label"] + self.metadata_cols, value_vars=value_vars
            ).rename({"variable": "metric"}, axis=1)
            import plotly.express as px

            fig = px.histogram(
                long,
                x="value",
                facet_col="metric",
                facet_col_wrap=5,
                color="label",
                barmode="overlay",
            )
            fig.update_xaxes(matches=None, showticklabels=True)
            fig.update_yaxes(matches=None, showticklabels=True)
            fig.update_traces(bingroup=None)
        fig.show()
