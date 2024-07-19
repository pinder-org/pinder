from __future__ import annotations
from itertools import repeat
from pathlib import Path
from typing import Any

import pandas as pd
from pandas._typing import DtypeArg
from tqdm import tqdm

from pinder.core.utils import setup_logger
from pinder.core.utils.process import process_starmap


log = setup_logger(__name__)


def read_csv_non_default_na(
    csv_file: Path,
    sep: str = ",",
    dtype: DtypeArg | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a csv file into pandas DataFrame without casting NA to NaN.

    Handle cases like asym_id = `NA`, which should NOT be cast to `NaN`!

    This method sets keep_default_na to False and passes `na_values` with all
    of the default values listed in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    except for `NA`.

    Parameters
    ----------
    csv_file : Path
        Path to tabular data to read.
    sep : str
        Character or regex pattern to treat as the delimiter. Defaults to ','.
        If sep=None, the C engine cannot automatically detect the separator,
        but the Python parsing engine can, meaning the latter will be used and
        automatically detect the separator from only the first valid row of the
        file by Python’s builtin sniffer tool, csv.Sniffer.
    dtype : dtype or dict of {Hashabledtype}, optional
        Data type(s) to apply to either the whole dataset or individual columns.
        E.g., {'a': np.float64, 'b': np.int32, 'c': 'Int64'}
        Use str or object together with suitable na_values settings to preserve
        and not interpret dtype.
    **kwargs : Any
        Any additional kwargs are passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame

    """
    if dtype is None:
        dtype = {"pdb_id": "str", "entry_id": "str"}

    na_values = [
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        # "NA",
        "NULL",
        "NaN",
        "n/a",
        "nan",
        "null",
    ]
    data = pd.read_csv(
        csv_file,
        na_values=na_values,
        keep_default_na=False,
        dtype=dtype,
        sep=sep,
        **kwargs,
    )
    return data


def safe_read_csv(csv_file: Path, sep: str = ",") -> pd.DataFrame:
    try:
        if csv_file.suffix == ".parquet":
            df = pd.read_parquet(csv_file)
        else:
            df = pd.read_csv(csv_file, sep=sep)
    except pd.errors.EmptyDataError as e:
        # It was an empty csv
        df = pd.DataFrame()
    except Exception as e:
        log.error(f"Encountered unexpected error reading {csv_file}...{e}")
        df = pd.DataFrame()
    return df


def parallel_read_csvs(
    csv_files: list[Path],
    max_workers: int | None = None,
    sep: str = ",",
    parallel: bool = True,
) -> list[pd.DataFrame] | None:
    dfs: list[pd.DataFrame] = process_starmap(
        safe_read_csv,
        zip(csv_files, repeat(sep)),
        parallel=parallel,
        max_workers=max_workers,
    )
    dfs = [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if len(dfs):
        return dfs
    else:
        return None
