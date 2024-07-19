"""Utilities for interacting with the RCSB nextgen rsync server."""

from __future__ import annotations
from itertools import repeat
from pathlib import Path
from subprocess import check_output, Popen, PIPE
from time import sleep
from tqdm import tqdm

from pinder.core.utils import setup_logger


log = setup_logger(__name__)


def download_rscb_files(
    data_dir: Path = Path("."),
    two_char_code: str | None = None,
    redirect_stdout: bool = True,
    retries: int = 5,
) -> None:
    """
    This function downloads RCSB files using rsync.

    Parameters
    ----------
    two_char_code : Optional[str]
        A two character code representing the batch of files to download.
        If not provided, all files will be downloaded.
    data_dir : str
        The directory where the downloaded files will be stored.
    redirect_stdout : bool
        Whether to silence stdout by redirecting to /dev/null. Default is True.

    Examples
    --------
    >>> download_rscb_files('./data', '1a') # doctest: +SKIP
    This will download the batch of files represented by the code '1a', such as 31ab, 51ac, etc.

    >>> download_rscb_files(data_dir='./data') # doctest: +SKIP
    This will download all files.
    """

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        log.warning(f"Directory: {data_dir} does not exist. Creating")
        data_dir.mkdir(parents=True, exist_ok=True)

    SERVER = "rsync-nextgen.wwpdb.org::rsync"
    PORT = "33444"

    redirect = "> /dev/null 2>/dev/null" if redirect_stdout else ""
    if two_char_code:
        command = (
            f"rsync -rlpt -z --delete --port={PORT} "
            f'--include "*/" --include "*-enrich.cif.gz" --exclude="*" '
            f"{SERVER}/data/entries/divided/{two_char_code}/ {data_dir}/{two_char_code}/ {redirect}"
        )
    else:
        command = (
            f"rsync -rlpt -z --delete --port={PORT} "
            f'--include "*/" --include "*-enrich.cif.gz" --exclude="*" '
            f"{SERVER}/data/entries/divided/ {data_dir}/ {redirect}"
        )
    try:
        log.info(command)
        proc = Popen(command, shell=True, stderr=PIPE, stdout=PIPE)
        stdout, stderr = proc.communicate()
        result = stdout.decode().strip().split("\n")
        if proc.returncode != 0:
            for ln in stderr.decode().splitlines():
                log.error(ln.strip())
            if retries > 0:
                retries -= 1
                log.info(f"Retrying rsync command... {retries} retries remaining")
                sleep_time = 2 ** (max([retries, 6]) - retries)
                log.info(f"Sleeping for {sleep_time} seconds")
                sleep(sleep_time)
                return download_rscb_files(
                    data_dir, two_char_code, redirect_stdout, retries
                )
        else:
            log.info("Command executed successfully.")
    except Exception as e:
        log.error(f"An error occurred while executing the command: {e}")


def download_two_char_codes(
    codes: list[str],
    data_dir: Path = Path("."),
    redirect_stdout: bool = True,
) -> None:
    """This function downloads RCSB files corresponding to a list of two-character codes using rsync.
    The two character codes map to the second two characters in a PDB ID.

    Parameters
    ----------
    two_char_codes : list[str]
        A list of two character code representing the batches of files to download.
    data_dir : str
        The directory where the downloaded files will be stored.
    redirect_stdout : bool
        Whether to silence stdout by redirecting to /dev/null. Default is True.

    Examples
    --------
    >>> download_two_char_codes(['1a', '1b'], './data') # doctest: +SKIP
    This will download the batch of files represented by the code '1a' and '1b', such as 31ab, 51ac, etc.
    """
    log.info(f"Downloading {len(codes)} two_char_codes")
    log.debug(f"{codes}")
    for two_char_code in tqdm(codes):
        download_rscb_files(
            data_dir=data_dir,
            two_char_code=two_char_code,
            redirect_stdout=redirect_stdout,
        )


def get_rsync_directories() -> list[str]:
    cmd = [
        "rsync",
        "--port=33444",
        "--list-only",
        "rsync-nextgen.wwpdb.org::rsync/data/entries/divided/",
    ]
    output = check_output(cmd).decode("utf-8").split("\n")
    # return output
    directories = [
        line.split()[-1]
        for line in output
        if line.startswith("d") and not line.endswith(".")
    ]
    return directories


def get_rsync_two_char_pdb_entries(two_char_code: str, retries: int = 3) -> list[str]:
    cmd = [
        "rsync",
        "--port=33444",
        "--list-only",
        f"rsync-nextgen.wwpdb.org::rsync/data/entries/divided/{two_char_code}/",
    ]
    try:
        output = check_output(cmd).decode("utf-8").split("\n")
        entries = [
            line.split()[-1]
            for line in output
            if line.startswith("d") and not line.endswith(".")
        ]
        return entries
    except Exception as e:
        log.warning(f"Failed to fetch entries for two_char_code: {two_char_code}")
        if retries > 0:
            retries -= 1
            log.warning(f"Retrying rsync command... {retries} retries remaining")
            sleep_time = 2 ** (max([retries, 6]) - retries)
            log.debug(f"Sleeping for {sleep_time} seconds")
            sleep(sleep_time)
            return get_rsync_two_char_pdb_entries(two_char_code, retries)
        else:
            log.error(
                f"Failed to fetch entries for {two_char_code}. Retries exhausted."
            )
            return []


def get_all_rsync_entries(two_char_codes: list[str]) -> list[str]:
    all_entries = []
    for code in tqdm(two_char_codes):
        entries = get_rsync_two_char_pdb_entries(code)
        if entries:
            all_entries.extend(entries)
    return all_entries


def get_two_char_codes_not_downloaded(
    data_dir: Path,
    two_char_codes: list[str],
) -> list[str]:
    missing = []
    for two_char_code in tqdm(two_char_codes):
        two_char_dir = data_dir / two_char_code
        if not two_char_dir.is_dir():
            missing.append(two_char_code)
            continue
        two_char_cifs = list(two_char_dir.glob("*/*enrich.cif.gz"))
        two_char_entries = get_rsync_two_char_pdb_entries(two_char_code)
        if len(two_char_cifs) != len(two_char_entries):
            delta = len(two_char_entries) - len(two_char_cifs)
            print(f"Two char code {two_char_code} if missing {delta} entries!")
            missing.append(two_char_code)
    return missing
