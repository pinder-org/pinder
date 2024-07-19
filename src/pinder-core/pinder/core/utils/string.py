from __future__ import annotations
import re


def natural_sort(list_to_sort: list[str]) -> list[str]:
    """Sorts the given iterable in the way that is expected.

    This function sorts the given list in a natural order. For example,
    the list ['A11', 'A9', 'A10'] will be sorted as ['A9', 'A10', 'A11'].

    Parameters
    ----------
    list_to_sort : List[str]
        The list to be sorted.

    Returns
    -------
    List[str]
        The sorted list.

    Examples
    --------
    >>> natural_sort(['A11', 'A9', 'A10'])
    ['A9', 'A10', 'A11']
    """

    def convert_to_integer(text: str) -> int | str:
        return int(text) if text.isdigit() else text.lower()

    def alphanumeric_key(key: str) -> list[int | str]:
        return [convert_to_integer(c) for c in re.split("([0-9]+)", key)]

    return sorted(list_to_sort, key=alphanumeric_key)
