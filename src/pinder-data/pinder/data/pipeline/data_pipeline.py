from __future__ import annotations
import json
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sized, TypeVar
from typing_extensions import ParamSpec

import pandas as pd
from pinder.core.utils.log import setup_logger
from pinder.core.utils.cloud import get_container_cpu_frac
from pinder.data import (
    alignment_utils,
    config,
    find_transitive_hits,
    foldseek_utils,
    get_alignment_similarity,
    get_annotations,
    get_apo,
    get_clusters,
    get_data,
    get_dimers,
    get_splits,
    get_test_set,
    graph_utils,
    rcsb_rsync,
)
from pinder.data.annotation import graphql
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.pipeline import cache, scatter
from pinder.data.pipeline.tasks import get_stage_tasks, run_task

P = ParamSpec("P")
T = TypeVar("T")


log = setup_logger(__name__)

stages: dict[str, list[dict[str, str | bool]]] = {
    "download_rcsb_files": [
        {"method_name": "generate_download_rcsb_files_tasks", "distributor": True},
        {
            "method_name": "download_rcsb_files",
            "distributed": True,
            "input_kwarg": "codes",
        },
    ],
    "ingest_rcsb_files": [
        {"method_name": "generate_cif_ingest_tasks", "distributor": True},
        {
            "method_name": "ingest_rcsb_files",
            "distributed": True,
            "input_kwarg": "mmcif_files",
        },
    ],
    "get_pisa_annotations": [
        {"method_name": "generate_pisa_annotation_tasks", "distributor": True},
        {
            "method_name": "get_pisa_annotations",
            "distributed": True,
            "input_kwarg": "mmcif_files",
        },
    ],
    "get_rcsb_annotations": [
        {"method_name": "generate_rcsb_annotation_tasks", "distributor": True},
        {
            "method_name": "get_rcsb_annotations",
            "distributed": True,
            "input_kwarg": "pdb_ids",
        },
        {"method_name": "join_rcsb_annotations", "join_step": True},
    ],
    "get_dimer_annotations": [
        {"method_name": "generate_dimer_annotation_tasks", "distributor": True},
        {
            "method_name": "get_dimer_annotations",
            "distributed": True,
            "input_kwarg": "dimer_pdbs",
        },
        {"method_name": "collect_dimer_annotations", "join_step": True},
    ],
    "get_dimer_contacts": [
        {"method_name": "generate_foldseek_contacts_tasks", "distributor": True},
        {
            "method_name": "get_dimer_contacts",
            "distributed": True,
            "input_kwarg": "dimer_pdbs",
        },
        {"method_name": "collect_foldseek_contacts", "join_step": True},
    ],
    "populate_entries": [
        {"method_name": "generate_populate_entry_tasks", "distributor": True},
        {
            "method_name": "populate_entries",
            "distributed": True,
            "input_kwarg": "entry_dirs",
        },
    ],
    "populate_predicted": [
        {"method_name": "generate_populate_predicted_tasks", "distributor": True},
        {
            "method_name": "populate_predicted",
            "distributed": True,
            "input_kwarg": "entry_dirs",
        },
    ],
    "index_dimers": [
        {"method_name": "index_dimers", "join_step": True},
    ],
    "add_predicted_monomers": [
        {"method_name": "add_predicted_monomers", "distributor": False},
    ],
    "get_apo": [
        {"method_name": "get_valid_apo_monomers", "distributor": False},
        {"method_name": "generate_apo_pairing_metric_tasks", "distributor": True},
        {
            "method_name": "get_apo_pairing_metrics",
            "distributed": True,
            "input_kwarg": "df_batch",
        },
        {"method_name": "join_apo_pairing_metrics", "join_step": True},
        {"method_name": "add_apo_pairings_to_index", "distributor": False},
    ],
    "foldseek": [
        {"method_name": "create_foldseek_dbs", "distributor": False},
        {"method_name": "generate_foldseek_tasks", "distributor": True},
        {
            "method_name": "run_foldseek",
            "distributed": True,
            "input_kwarg": "db_indices",
        },
        {"method_name": "join_foldseek", "join_step": True},
    ],
    "mmseqs": [
        {"method_name": "create_mmseqs_dbs", "distributor": False},
        {"method_name": "generate_mmseqs_tasks", "distributor": True},
        {"method_name": "run_mmseqs", "distributed": True, "input_kwarg": "db_indices"},
        {"method_name": "join_mmseqs", "join_step": True},
    ],
    "interface_graph": [
        {"method_name": "construct_interface_graph", "distributor": False},
    ],
    "foldseek_graph": [
        {"method_name": "construct_foldseek_graph", "distributor": False},
    ],
    "mmseqs_graph": [
        {"method_name": "construct_mmseqs_graph", "distributor": False},
    ],
    "cluster": [
        {"method_name": "cluster", "distributor": False},
    ],
    "deleak": [
        {"method_name": "generate_find_leakage_tasks", "distributor": True},
        {
            "method_name": "find_leakage",
            "distributed": True,
            "input_kwarg": "graph_batch",
        },
        {"method_name": "get_transitive_hits", "distributor": False},
        {
            "method_name": "get_af2_hard_difficulty_transitive_hits",
            "distributor": False,
        },
    ],
    "get_splits": [
        {"method_name": "get_splits", "distributor": False},
        {"method_name": "get_alignment_similarity", "distributor": False},
        {"method_name": "construct_final_index", "distributor": False},
    ],
    "get_test_set": [
        {"method_name": "get_test_set", "distributor": False},
    ],
}


def save_stage_metadata(
    pinder_mount: Path,
    step_name: str,
    version_metadata: dict[str, str],
) -> None:
    with open(pinder_mount / f"{step_name}-metadata.json", "w") as f:
        json.dump(version_metadata, f)


def year_month() -> str:
    pinder_version = datetime.now().strftime("%Y-%m")
    return pinder_version


def task_step(
    step_name: str,
    save_metadata: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    def task_decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @wraps(func)
        def wrapper(
            self: DataIngestPipeline, *args: P.args, **kwargs: P.kwargs
        ) -> T | None:
            if save_metadata:
                # Only save per-stage version metadata json if requested.
                # We want to avoid overwriting same file if the task is being scattered.
                save_stage_metadata(
                    self.pinder_mnt_dir, step_name, self.stage_version_metadata
                )
            # Pre-invocation logic, determine whether to skip the task
            skip = cache.skip_step(
                step_name, self.run_specific_step, self.skip_specific_step
            )
            if skip:
                return None
            return func(self, *args, **kwargs)

        return wrapper

    return task_decorator


def scatter_step(
    step_name: str,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    def scatter_decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @wraps(func)
        def wrapper(
            self: DataIngestPipeline, *args: P.args, **kwargs: P.kwargs
        ) -> T | None:
            # Auto-save per-stage version metadata json
            save_stage_metadata(
                self.pinder_mnt_dir, step_name, self.stage_version_metadata
            )
            # Pre-invocation logic, determine whether to skip the task
            skip = cache.skip_step(
                step_name, self.run_specific_step, self.skip_specific_step
            )
            if skip:
                # Set scatter_batches and short-circuit the steps scatter method
                self.scatter_batches = [[]]
                return None
            return func(self, *args, **kwargs)

        return wrapper

    return scatter_decorator


class DataIngestPipeline:
    def __init__(
        self,
        image: str = "local",
        pinder_mount_point: str = str(Path.home() / ".local/share/pinder"),
        pinder_release: str = "2024-02",
        ingest_config: config.PinderDataGenConfig = config.PinderDataGenConfig(),
        contact_config: config.ContactConfig = config.ContactConfig(),
        transient_interface_config: config.TransientInterfaceConfig = config.TransientInterfaceConfig(),
        foldseek_config: config.FoldseekConfig = config.FoldseekConfig(),
        mmseqs_config: config.MMSeqsConfig = config.MMSeqsConfig(),
        scatter_config: config.ScatterConfig = config.ScatterConfig(),
        graph_config: config.GraphConfig = config.GraphConfig(),
        cluster_config: config.ClusterConfig = config.ClusterConfig(),
        apo_config: config.ApoPairingConfig = config.ApoPairingConfig(),
        ialign_config: config.IalignConfig = config.IalignConfig(),
        two_char_code: str | None = None,
        run_specific_step: str = "",
        skip_specific_step: str = "",
        use_cache: bool = True,
        google_cloud_project: str = "vantai-analysis",
    ) -> None:
        """Parameters
        ----------
        image: str
            The fully-specified docker image to use for data ingestion. It is stored along with pinder_release in
            in a json file per pipeline stage as version metadata. If running locally, it default to `local`.
        pinder_mount_point: str
            The root directory in which to store data ingestion. Defaults to /pinder.
            Note: this parameter and the pipeline are compatible with NFS file mount locations.
        pinder_release: str
            The year-month formatted pinder version to use. Defaults to static release.
            Specify empty string to switch to current month.
        ingest_config: config.PinderDataGenConfig
            Configuration parameters used for data ingestion / generation.
        contact_config: config.ContactConfig
            Configuration parameters used for extracting dimer contacts in foldseek residue numbering conventions.
        transient_interface_config: config.TransientInterfaceConfig
            Configuration parameters used for assigning labels to potentially transient/weak interfaces.
        foldseek_config: config.FoldseekConfig
            Configuration parameters used for running foldseek on dimer chains.
        mmseqs_config: config.MMSeqsConfig
            Configuration parameters used for running mmseqs on dimer chains.
        scatter_config: config.ScatterConfig
            Batching parameters used for defining foreach scatter task sizes.
        graph_config: config.GraphConfig
            Config parameters for constructing graphs from alignment files.
        cluster_config: config.ClusterConfig
            Config parameters for clustering the pinder dimers and generating splits.
        apo_config: config.ApoPairingConfig
            Config parameters for evaluating and pairing apo monomers with holo dimer monomers.
        ialign_config: config.IalignConfig
            Configuration parameters for evaluating potential alignment leakage via iAlign.
        two_char_code: str | None
            A two character code representing the batch of files to download.
            If not provided, all files will be downloaded. Pattern is *{two_char_code}* for matching PDB IDs.
        run_specific_step: str
           Target a specific step of the workflow (only works if there has been a partial run previously).
           The value should be the name of a DAG step.
        skip_specific_step: str
           Skip a specific step of the workflow (only works if there has been a partial run previously).
           The value should be the name of a DAG step.
        """

        step_name = "start"
        self.run_specific_step = run_specific_step
        self.skip_specific_step = skip_specific_step
        self.config = ingest_config
        self.contact_config = contact_config
        self.transient_interface_config = transient_interface_config
        self.foldseek_config = foldseek_config
        self.mmseqs_config = mmseqs_config
        self.scatter_config = scatter_config
        self.graph_config = graph_config
        self.cluster_config = cluster_config
        self.apo_config = apo_config
        self.ialign_config = ialign_config
        self.use_cache = use_cache
        self.two_char_code = two_char_code
        self.google_cloud_project = google_cloud_project
        self.pinder_version = pinder_release if pinder_release != "" else year_month()
        self.pinder_mnt_dir = Path(pinder_mount_point)
        if not self.pinder_mnt_dir.is_dir():
            self.pinder_mnt_dir.mkdir(parents=True)
        self.data_dir = self.pinder_mnt_dir / "data"
        self.pinder_dir = self.pinder_mnt_dir / self.pinder_version
        self.pinder_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.stage_version_metadata = {
            "image": image,
            "pinder_version": self.pinder_version,
        }
        save_stage_metadata(self.pinder_mnt_dir, step_name, self.stage_version_metadata)
        # Save config collection to json with config dataclass name as file name.
        config_collection = [
            ingest_config,
            contact_config,
            transient_interface_config,
            foldseek_config,
            mmseqs_config,
            scatter_config,
            graph_config,
            cluster_config,
            apo_config,
            ialign_config,
        ]
        for cfg in config_collection:
            cfg_name: str = type(cfg).__name__
            with open(self.pinder_mnt_dir / f"{cfg_name}.json", "w") as f:
                json.dump(cfg.__dict__, f)

        self.contact_config_hash = config.get_config_hash(self.contact_config)
        self.graph_config_hash = config.get_config_hash(self.graph_config)
        self.cluster_config_hash = config.get_config_hash(self.cluster_config)
        with open(self.pinder_mnt_dir / f"{self.contact_config_hash}.json", "w") as f:
            json.dump(self.contact_config.__dict__, f)

    def run(self) -> None:
        for stage in stages:
            log.info(
                "\n".join(
                    ["\n", "#" * 20, f"# Starting PINDER-DATA stage {stage}", "#" * 20]
                )
            )
            self.run_stage(stage)

    def run_stage(self, stage_name: str, specific_method: str | None = None) -> None:
        assert stage_name in stages
        batches = None
        for step_spec in stages[stage_name]:
            method_name = step_spec["method_name"]
            assert isinstance(method_name, str)
            if specific_method and specific_method != method_name:
                log.info(
                    f"Skipping {stage_name}->{method_name}, specific_method was {specific_method}"
                )
                continue
            run_func = getattr(self, method_name)
            if step_spec.get("distributor"):
                run_func()
                batches = getattr(self, "scatter_batches")
            if step_spec.get("distributed"):
                if batches is not None and isinstance(batches, Sized):
                    for scatter_input in batches:
                        run_func(scatter_input)
            else:
                run_func()

    @scatter_step("download_rcsb_files")
    def generate_download_rcsb_files_tasks(self) -> None:
        if self.two_char_code:
            self.scatter_batches = [[self.two_char_code]]
        else:
            self.scatter_batches = get_stage_tasks(
                self.data_dir,
                input_type="two_char_codes",
                batch_size=self.scatter_config.two_char_batch_size,
            )

    @task_step("download_rcsb_files")
    def download_rcsb_files(self, codes: list[str]) -> None:
        run_task(
            rcsb_rsync.download_two_char_codes,
            task_input=dict(codes=codes, data_dir=self.data_dir, redirect_stdout=True),
            iterable_kwarg="codes",
        )

    @scatter_step("ingest_rcsb_files")
    def generate_cif_ingest_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="cif",
            batch_size=self.scatter_config.mmcif_batch_size,
            cache_func=cache.get_uningested_mmcif,
        )

    @task_step("ingest_rcsb_files")
    def ingest_rcsb_files(self, mmcif_files: list[Path]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.9)
        log.info(f"MMCIF FILES: {mmcif_files}")
        run_task(
            get_data.ingest_mmcif_list,
            task_input=dict(
                mmcif_list=mmcif_files,
                max_workers=workers,
                parallel=True,
                config=self.config,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="mmcif_list",
        )

    @scatter_step("get_pisa_annotations")
    def generate_pisa_annotation_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="cif",
            batch_size=self.scatter_config.mmcif_batch_size,
            cache_func=cache.get_pisa_unannotated,
        )

    @task_step("get_pisa_annotations")
    def get_pisa_annotations(self, mmcif_files: list[Path]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.9)
        run_task(
            get_annotations.get_pisa_annotations,
            task_input=dict(
                mmcif_list=mmcif_files,
                max_workers=workers,
                parallel=True,
                config=self.config,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="mmcif_list",
        )

    @scatter_step("get_rcsb_annotations")
    def generate_rcsb_annotation_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="pdb_ids",
            batch_size=self.scatter_config.graphql_batch_size,
            cache_func=cache.get_rcsb_unannotated,
            cache_kwargs={"pinder_dir": self.pinder_dir},
        )

    @task_step("get_rcsb_annotations")
    def get_rcsb_annotations(self, pdb_ids: list[str]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.9)
        run_task(
            graphql.populate_rcsb_annotations,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                pdb_ids=pdb_ids,
                max_workers=workers,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="pdb_ids",
        )

    @task_step("get_rcsb_annotations")
    def join_rcsb_annotations(self) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.8)
        run_task(
            graphql.collect_rcsb_annotations,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                max_workers=workers,
                use_cache=self.use_cache,
            ),
        )

    @scatter_step("get_dimer_annotations")
    def generate_dimer_annotation_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="dimers",
            batch_size=self.scatter_config.dimer_batch_size,
            cache_func=cache.get_unannotated_dimer_pdbs,
            cache_kwargs={"use_cache": self.use_cache},
        )

    @task_step("get_dimer_annotations")
    def get_dimer_annotations(self, dimer_pdbs: list[Path]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.9)
        run_task(
            get_annotations.get_dimer_annotations,
            task_input=dict(
                dimer_list=dimer_pdbs,
                parallel=True,
                max_workers=workers,
                config=self.config,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="dimer_list",
        )

    @task_step("get_dimer_annotations")
    def collect_dimer_annotations(self) -> None:
        run_task(
            get_annotations.collect,
            task_input=dict(data_dir=self.data_dir, pinder_dir=self.pinder_dir),
        )

    @scatter_step("get_dimer_contacts")
    def generate_foldseek_contacts_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="dimers",
            batch_size=self.scatter_config.dimer_batch_size,
            cache_func=cache.get_dimer_pdbs_missing_foldseek_contacts,
            cache_kwargs={"config_hash": self.contact_config_hash},
        )

    @task_step("get_dimer_contacts")
    def get_dimer_contacts(self, dimer_pdbs: list[Path]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.8)
        run_task(
            alignment_utils.populate_foldseek_contacts,
            task_input=dict(
                dimer_pdbs=dimer_pdbs,
                parallel=True,
                max_workers=workers,
                contact_config=self.contact_config,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="dimer_pdbs",
        )

    @task_step("get_dimer_contacts")
    def collect_foldseek_contacts(self) -> None:
        run_task(
            alignment_utils.collect_contact_jsons,
            task_input=dict(
                data_dir=self.data_dir,
                dimer_pdbs=list(self.data_dir.glob("*/*/*__*--*__*.pdb")),
                config=self.contact_config,
                config_hash=self.contact_config_hash,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="dimer_pdbs",
        )

    @scatter_step("populate_entries")
    def generate_populate_entry_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="entries",
            batch_size=self.scatter_config.dimer_batch_size,
        )

    @task_step("populate_entries")
    def populate_entries(self, entry_dirs: list[Path]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.25)
        run_task(
            get_dimers.populate_entries,
            task_input=dict(
                data_dir=self.data_dir,
                pinder_dir=self.pinder_dir,
                google_cloud_project=self.google_cloud_project,
                entry_dirs=[d for d in entry_dirs if d.is_dir()],
                use_cache=self.use_cache,
                populate_alphafold=False,
                parallel=True,
                max_workers=workers,
            ),
            iterable_kwarg="entry_dirs",
        )

    @scatter_step("populate_predicted")
    def generate_populate_predicted_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.data_dir,
            input_type="entries",
            batch_size=self.scatter_config.predicted_batch_size,
        )

    @task_step("populate_predicted")
    def populate_predicted(self, entry_dirs: list[Path]) -> None:
        run_task(
            get_dimers.populate_predicted_from_monomers,
            task_input=dict(
                data_dir=self.data_dir,
                pinder_dir=self.pinder_dir,
                google_cloud_project=self.google_cloud_project,
                entry_dirs=[d for d in entry_dirs if d.is_dir()],
                use_cache=self.use_cache,
            ),
            iterable_kwarg="entry_dirs",
        )

    @task_step("index_dimers", save_metadata=True)
    def index_dimers(self) -> None:
        run_task(
            get_dimers.get_populated_entries,
            task_input=dict(
                data_dir=self.data_dir,
                pinder_dir=self.pinder_dir,
                google_cloud_project=self.google_cloud_project,
                use_cache=self.use_cache,
                transient_interface_config=self.transient_interface_config,
            ),
        )

    @task_step("add_predicted_monomers", save_metadata=True)
    def add_predicted_monomers(self) -> None:
        run_task(
            get_dimers.add_predicted_monomers_to_index,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                use_cache=self.use_cache,
            ),
        )

    @task_step("get_apo", save_metadata=True)
    def get_valid_apo_monomers(self) -> None:
        run_task(
            get_apo.get_valid_apo_monomer_ids,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                config=self.apo_config,
                use_cache=self.use_cache,
                remove_chain_copies=True,
            ),
        )

    @scatter_step("get_apo")
    def generate_apo_pairing_metric_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.pinder_dir,
            input_type="putative_apo_pairings",
            batch_size=self.scatter_config.apo_pairing_id_batch_size,
            scatter_method=scatter.chunk_apo_pairing_ids,
            scatter_kwargs={"pinder_dir": self.pinder_dir},
        )

    @task_step("get_apo")
    def get_apo_pairing_metrics(self, pairing_batch: tuple[list[str], Path]) -> None:
        workers = get_container_cpu_frac(max_cpu_fraction=0.9)
        if len(pairing_batch) == 2:
            pairing_ids, output_pqt = pairing_batch
            run_task(
                get_apo.get_apo_pairing_metrics,
                task_input=dict(
                    pinder_dir=self.pinder_dir,
                    putative_pairs=pairing_ids,
                    config=self.apo_config,
                    max_workers=workers,
                    output_parquet=output_pqt,
                    use_cache=self.use_cache,
                ),
            )

    @task_step("get_apo")
    def join_apo_pairing_metrics(self) -> None:
        run_task(
            get_apo.collate_apo_metrics,
            task_input=dict(
                metric_dir=self.pinder_dir / "apo_metrics/pair_eval",
                output_parquet=self.pinder_dir
                / "apo_metrics/two_sided_apo_monomer_metrics.parquet",
            ),
        )

    @task_step("get_apo")
    def add_apo_pairings_to_index(self) -> None:
        run_task(
            get_apo.add_all_apo_pairings_to_index,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                config=self.apo_config,
                use_cache=self.use_cache,
            ),
        )

    @task_step("foldseek")
    def create_foldseek_dbs(self) -> None:
        run_task(
            foldseek_utils.setup_foldseek_dbs,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                foldseek_db_size=self.scatter_config.foldseek_db_size,
                use_cache=self.use_cache,
            ),
        )

    @scatter_step("foldseek")
    def generate_foldseek_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            data_dir=self.pinder_dir / "foldseek",
            input_type="foldseek",
            batch_size=self.scatter_config.foldseek_db_size,
            scatter_method=scatter.chunk_all_vs_all_indices,
        )

    @task_step("foldseek")
    def run_foldseek(self, db_indices: tuple[int, int]) -> None:
        run_task(
            foldseek_utils.run_foldseek_db_pair,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                db_indices=db_indices,
                foldseek_config=self.foldseek_config,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="db_indices",
        )

    @task_step("foldseek")
    def join_foldseek(self) -> None:
        run_task(
            foldseek_utils.collate_foldseek_alignments,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                foldseek_db_size=self.scatter_config.foldseek_db_size,
                use_cache=self.use_cache,
                alignment_filename=self.foldseek_config.alignment_filename,
            ),
        )

    @task_step("mmseqs")
    def create_mmseqs_dbs(self) -> None:
        run_task(
            foldseek_utils.setup_mmseqs_dbs,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                mmseqs_db_size=self.scatter_config.foldseek_db_size,
                use_cache=self.use_cache,
            ),
        )

    @scatter_step("mmseqs")
    def generate_mmseqs_tasks(self) -> None:
        # We just point to the foldseek PDB dir and use the same index-pairing as in foldseek
        self.scatter_batches = get_stage_tasks(
            data_dir=self.pinder_dir / "foldseek",
            input_type="foldseek",
            batch_size=self.scatter_config.foldseek_db_size,
            scatter_method=scatter.chunk_all_vs_all_indices,
        )

    @task_step("mmseqs")
    def run_mmseqs(self, db_indices: tuple[int, int]) -> None:
        run_task(
            foldseek_utils.run_mmseqs_db_pair,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                db_indices=db_indices,
                mmseqs_config=self.mmseqs_config,
                use_cache=self.use_cache,
            ),
            iterable_kwarg="db_indices",
        )

    @task_step("mmseqs")
    def join_mmseqs(self) -> None:
        run_task(
            foldseek_utils.collate_mmseqs_alignments,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                foldseek_db_size=self.scatter_config.foldseek_db_size,
                use_cache=self.use_cache,
                alignment_filename=self.mmseqs_config.alignment_filename,
            ),
        )

    @task_step("interface_graph", save_metadata=True)
    def construct_interface_graph(self) -> None:
        interface_pkl = (
            self.pinder_mnt_dir
            / "foldseek_contacts"
            / self.contact_config_hash
            / "interfaces.pkl"
        )
        run_task(
            graph_utils.construct_interface_graph,
            task_input=dict(
                interface_pkl=interface_pkl,
                output_dir=self.pinder_dir / "graphs",
                graph_config=self.graph_config,
            ),
        )

    @task_step("foldseek_graph", save_metadata=True)
    def construct_foldseek_graph(self) -> None:
        output_hash_fp = self.pinder_dir / "graphs" / self.graph_config_hash
        interface_pkl = output_hash_fp / "min_length_interfaces.pkl"
        run_task(
            graph_utils.construct_interface_alignment_graph,
            task_input=dict(
                interface_pkl=interface_pkl,
                alignment_file=self.pinder_dir / "foldseek/foldseek_dbs/alignment.txt",
                alignment_type="foldseek",
                output_dir=self.pinder_dir / "graphs",
                graph_config=self.graph_config,
                use_cache=self.use_cache,
            ),
        )

    @task_step("mmseqs_graph", save_metadata=True)
    def construct_mmseqs_graph(self) -> None:
        output_hash_fp = self.pinder_dir / "graphs" / self.graph_config_hash
        interface_pkl = output_hash_fp / "min_length_interfaces.pkl"
        run_task(
            graph_utils.construct_interface_alignment_graph,
            task_input=dict(
                interface_pkl=interface_pkl,
                alignment_file=self.pinder_dir / "mmseqs2/mmseqs_dbs/alignment.txt",
                alignment_type="mmseqs",
                output_dir=self.pinder_dir / "graphs",
                graph_config=self.graph_config,
                use_cache=self.use_cache,
            ),
        )

    @task_step("cluster", save_metadata=True)
    def cluster(self) -> None:
        graph_fp = self.pinder_dir / "graphs" / self.graph_config_hash
        mmseqs_pkl = graph_fp / "cleaned_mmseqs_alignment_graph.pkl"
        foldseek_pkl = graph_fp / "cleaned_foldseek_alignment_graph.pkl"
        interface_pkl = graph_fp / "min_length_interfaces.pkl"
        index = read_csv_non_default_na(
            self.pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
        )
        foldseek_graph = graph_utils.load_graph_pickle(foldseek_pkl)
        mmseqs_graph = graph_utils.load_graph_pickle(mmseqs_pkl)
        interfaces = alignment_utils.load_interface_pkl(interface_pkl)
        run_task(
            get_clusters.cluster,
            task_input=dict(
                index=index,
                foldseek_graph=foldseek_graph,
                mmseqs_graph=mmseqs_graph,
                interfaces_clean=interfaces,
                output_index_filename="index.2.csv.gz",
                checkpoint_dir=self.pinder_dir / "cluster" / self.cluster_config_hash,
                config=self.cluster_config,
                use_cache=self.use_cache,
            ),
        )

    @scatter_step("deleak")
    def generate_find_leakage_tasks(self) -> None:
        self.scatter_batches = get_stage_tasks(
            self.pinder_dir / "graphs" / self.graph_config_hash,
            input_type="graph_types",
            batch_size=1,
        )

    @task_step("deleak")
    def find_leakage(self, graph_batch: list[tuple[str, bool]]) -> None:
        # Scatter batch is a list of len 1 for each graph type
        assert len(graph_batch) == 1
        graph_spec = graph_batch[0]
        assert len(graph_spec) == 2
        graph_type, af2_transitive_hits = graph_spec
        run_task(
            find_transitive_hits.get_leakage_dict,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                graph_type=graph_type,
                config=self.cluster_config,
                graph_config=self.graph_config,
                use_cache=self.use_cache,
                af2_transitive_hits=af2_transitive_hits,
            ),
        )

    @task_step("deleak")
    def get_transitive_hits(self) -> None:
        run_task(
            find_transitive_hits.get_transitive_hits,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                config=self.cluster_config,
                graph_config=self.graph_config,
                use_cache=self.use_cache,
                af2_transitive_hits=False,
            ),
        )

    @task_step("deleak")
    def get_af2_hard_difficulty_transitive_hits(self) -> None:
        thresh_label = "{:.2f}".format(
            self.cluster_config.foldseek_af2_difficulty_threshold
        ).replace(".", "")
        run_task(
            find_transitive_hits.get_transitive_hits,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                config=self.cluster_config,
                graph_config=self.graph_config,
                test_systems_output=f"af2_lddt{thresh_label}_test_sys_table.csv",
                deleak_map_output=f"af2_lddt{thresh_label}_transitive_hits_mapping.csv",
                use_cache=self.use_cache,
                af2_transitive_hits=True,
            ),
        )

    @task_step("get_splits", save_metadata=True)
    def get_splits(self) -> None:
        run_task(
            get_splits.get_splits,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                config=self.cluster_config,
                use_cache=self.use_cache,
            ),
        )

    @task_step("get_splits", save_metadata=False)
    def get_alignment_similarity(self) -> None:
        run_task(
            get_alignment_similarity.get_alignment_similarity,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                cluster_config=self.cluster_config,
                ialign_config=self.ialign_config,
                use_cache=self.use_cache,
            ),
        )

    @task_step("get_splits", save_metadata=False)
    def construct_final_index(self) -> None:
        run_task(
            get_splits.construct_final_index,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                config=self.cluster_config,
                apo_config=self.apo_config,
                use_cache=self.use_cache,
            ),
        )

    @task_step("get_test_set", save_metadata=True)
    def get_test_set(self) -> None:
        run_task(
            get_test_set.curate_test_split,
            task_input=dict(
                pinder_dir=self.pinder_dir,
                use_cache=self.use_cache,
            ),
        )


if __name__ == "__main__":
    local_pinder_mount = Path("./").absolute() / "pinder-data-pipeline"
    if not local_pinder_mount.is_dir():
        local_pinder_mount.mkdir(parents=True)
    run_specific_step = ""  # "download_rcsb_files"
    pipe = DataIngestPipeline(
        image="local",
        pinder_mount_point=str(local_pinder_mount),
        run_specific_step=run_specific_step,
        two_char_code="bo",
        use_cache=True,
    )
    pipe.run()
    # pipe.run_stage(pipe.run_specific_step)
