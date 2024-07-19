from __future__ import annotations
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Callable, IO

from pinder.core.utils import setup_logger
from pinder.methods import SUPPORTED_ACTIONS, SUPPORTED_METHODS, SUPPORTED_PAIRS
from pinder.methods.preprocess import prepare_inference_inputs


log = setup_logger(__name__)

# Define a common function type for both yaml/json readers
ReaderType = Callable[[IO[Any]], dict[str, Any]]


def json_reader(file: IO[Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = json.load(file)
    return loaded


def yaml_reader(file: IO[Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(file)
    return loaded


def preprocess(
    method: str,
    data_dir: Path,
    subset: str | None = None,
    ids: list[str] | None = None,
    config: dict[str, Any] = {},
    pairs: SUPPORTED_PAIRS = SUPPORTED_PAIRS.ALL,
) -> None:
    if (
        method in SUPPORTED_METHODS
        and SUPPORTED_ACTIONS.PREPROCESS in SUPPORTED_METHODS[method]
    ):
        # Call the preprocess function of the method

        data_dir = Path(data_dir)

        inference_config = prepare_inference_inputs(
            data_dir, subset=subset, ids=ids, pairs=pairs
        )
        inference_config_df = pd.DataFrame(inference_config)
        inference_config_df["receptor"] = inference_config_df["receptor"].apply(
            lambda x: x.relative_to(data_dir)
        )
        inference_config_df["ligand"] = inference_config_df["ligand"].apply(
            lambda x: x.relative_to(data_dir)
        )
        inference_config_df.to_csv(data_dir / f"{method}_setup.csv", index=False)

    else:
        log.error(f"Method {method} does not support preprocessing.")
    return None


def train(
    method: str,
    data_dir: Path,
    subset: str | None = None,
    ids: list[str] | None = None,
    config: dict[str, Any] = {},
) -> None:
    if (
        method in SUPPORTED_METHODS
        and SUPPORTED_ACTIONS.TRAIN.value in SUPPORTED_METHODS[method]
    ):
        # Call the train function of the method
        pass
    else:
        log.error(f"Method {method} does not support training.")
    return None


def predict(
    method: str,
    results_dir: Path | None = None,
    data_dir: Path | None = None,
    subset: str | None = None,
    ids: list[str] | None = None,
    config: Path | None = None,
    pairs: SUPPORTED_PAIRS = SUPPORTED_PAIRS.ALL,
) -> None:
    """Run inference/prediction with the specified method on a subset of index `set` or on a list of PINDER IDs.

    Parameters:
        method (str): The method to use for prediction.
        results_dir (Path): Path to the directory where the results will be saved.
        data_dir (Path): Path to the directory where the preprocessed data is stored if available.
        subset (str): Optional subset of index to run prediction on (pinder-xl/s/af2).
        ids (list): A list of PINDER IDs to run prediction on. Takes priority over `subset` if provided.
        pairs (all/holo/apo/predicted): The type of monomer pairs to run prediction on.
        config (Path): An optional configuration file containing key-value parameters to pass to method.
            Supported formats are json and yaml.

    """
    if config:
        config_file = Path(config)
        with open(config_file) as f:
            if config_file.suffix == ".json":
                reader: ReaderType = json_reader
            elif config_file.suffix == ".yaml":
                reader = yaml_reader
            else:
                raise ValueError(
                    f"Invalid config file format {config_file.suffix}. Supported formats: [yaml, json]"
                )
            config_dict: dict[str, Any] = reader(f)
    else:
        config_dict = {}

    if (
        method in SUPPORTED_METHODS
        and SUPPORTED_ACTIONS.PREDICT.value in SUPPORTED_METHODS[method]
    ):
        # Call the predict function of the method
        import_module = __import__(
            f"pinder.methods.run.{method}.predict", fromlist=[""]
        )
        predict_method = getattr(import_module, f"predict_{method}")

        if not results_dir:
            results_dir = Path("./").absolute()

        config_dict["data_dir"] = data_dir
        config_dict["results_dir"] = results_dir

        inference_config = pd.from_csv(results_dir / f"{method}_setup.csv")

        for complex_config in inference_config.to_dict("records"):
            predict_method(complex_config, config_dict)
        log.info(f"Success! Finished docking {len(inference_config)} systems")
        return None
    else:
        log.error(f"Method {method} does not support prediction.")


def main() -> None:
    import fire

    fire.Fire(
        {
            SUPPORTED_ACTIONS.PREPROCESS.value: preprocess,
            SUPPORTED_ACTIONS.TRAIN.value: train,
            SUPPORTED_ACTIONS.PREDICT.value: predict,
        }
    )


if __name__ == "__main__":
    main()
