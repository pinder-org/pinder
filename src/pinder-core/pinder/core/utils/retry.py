from __future__ import annotations
from functools import wraps
from time import sleep
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec
from pinder.core.utils.log import setup_logger

LOG = setup_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def exponential_retry(
    max_retries: int,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Exponential backoff retry decorator.

    Retries the wrapped function/method `max_retries` times if the exceptions listed
    in ``exceptions`` are thrown.

    Parameters
    ----------
    max_retries : int
        The max number of times to repeat the wrapped function/method.
    initial_delay : float
        Initial number of seconds to sleep after first failed attempt.
    multiplier : float
        Amount to multiply the delay by before the next attempt.
    exceptions : tuple[Exception]
        Tuple of exceptions that trigger a retry attempt.

    """

    def decorator_retry(func: Callable[P, T]) -> Callable[P, T]:
        # preserve information about the original function, or the func name will be "wrapper" not "func"
        @wraps(func)
        def wrapper_retry(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    LOG.warning(
                        f"Exception thrown when attempting to run {func.__name__}, {retries}/{max_retries} retries remaining"
                    )
                    retries += 1
                    if retries < max_retries:
                        LOG.info(f"Retrying in {delay} seconds...")
                        sleep(delay)
                        delay *= multiplier
                    else:
                        LOG.error("Exceeded maximum number of retries.")
                        raise Exception("Exceeded maximum number of retries.") from e

        return wrapper_retry

    return decorator_retry
