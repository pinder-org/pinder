from functools import wraps
from time import time
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

from pinder.core.utils import setup_logger


P = ParamSpec("P")
T = TypeVar("T")


def timeit(func: Callable[P, T]) -> Callable[P, T]:
    """Simple function timer decorator"""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        log = setup_logger(".".join([func.__module__, func.__name__]))
        ts = time()
        result = None
        try:
            result = func(*args, **kwargs)
            log.info(f"runtime succeeded: {time() - ts:>9.2f}s")
        except Exception:
            log.error(f"runtime failed: {time() - ts:>9.2f}s")
            raise
        return result

    return wrapped
