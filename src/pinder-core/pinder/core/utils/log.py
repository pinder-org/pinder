from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
T = TypeVar("T")


# change default logging level in the environment variable LOG_LEVEL
DEFAULT_LOGGING_LEVEL: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
LOGGING_FORMAT: str = "%(asctime)s | %(name)s:%(lineno)d | %(levelname)s : %(message)s"


class PinderLoggingError(Exception):
    pass


def setup_logger(
    logger_name: str | None = None,
    log_level: int = DEFAULT_LOGGING_LEVEL,
    log_file: str | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """Setup logger for the module name as the logger name by default
    for easy tracing of what's happening in the code

    Parameters:
    -----------
    logger_name : str
        Name of the logger
    log_level : int
        Log level
    log_file: str | None
        optional log file to write to
    propagate : bool
        propagate log events to parent loggers, default = False

    Returns:
    --------
    logging.Logger:
        logger object

    Examples
    --------
    >>> logger = setup_logger("some_logger_name")
    >>> logger.name
    'some_logger_name'
    >>> logger.level
    20
    >>> logger = setup_logger(log_level=logging.DEBUG)
    >>> logger.name
    'log.py'
    >>> logger.level
    10
    """

    if logger_name is None:
        # Get module name as the logger name
        # this is copied from:
        # https://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        file_path = __file__ if module is None else module.__file__
        logger_name = os.path.basename(file_path) if file_path is not None else "log"

    # set up logger with the given logger name
    logger = logging.getLogger(logger_name)
    # check if logging level has been set externally
    # otherwise first pass logger.level == 0 (NOTSET)
    set_level = not bool(logger.level)
    if set_level:
        logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if set_level:
        handler.setLevel(log_level)
    formatter = logging.Formatter(LOGGING_FORMAT)
    handler.setFormatter(formatter)
    if not len(logger.handlers):
        logger.addHandler(handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        if set_level:
            file_handler.setLevel(log_level)
        if not [h for h in logger.handlers if h.__class__ == logging.FileHandler]:
            logger.addHandler(file_handler)
    logger.propagate = propagate

    return logger


def inject_logger(
    log_level: int = DEFAULT_LOGGING_LEVEL, log_file: str | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """A decorator function that injects a logger into the function

    Parameters
    ----------
    log_level: integer representing the log level (default: DEFAULT_LOGGING_LEVEL)
    log_file: optional file path to write logs to

    Example
    -------

    >>> @inject_logger()
    ... def my_function(name, log):
    ...     log.info(f"hello {name}")
    ...     return name
    >>> my_function(name="pinder")
    'pinder'
    >>> # 2023-11-01 09:15:37,683 | __main__.my_function:3 | INFO : hello pinder
    >>> @inject_logger(log_file="my_log.txt") # this will write logs to my_log.txt
    ... def my_function_writing_to_file(name, log):
    ...     log.info(f"hello {name}")
    ...     return name
    >>> my_function_writing_to_file(name="pinder")
    'pinder'
    >>> # 2023-11-01 09:15:37,683 | __main__.my_function:3 | INFO : hello pinder
    >>>
    >>> @inject_logger()
    ... def my_bad(name):
    ...     log.info(f"hello {name}")
    ...     return name
    >>> my_bad(name="pinder")
    Traceback (most recent call last):
        ...
    core.utils.log.PinderLoggingError: The function 'core.utils.log.my_bad' should contain a variable named 'log'
    >>>
    >>> @inject_logger(log_level=logging.DEBUG)
    ... def my_function(name, log):
    ...     log.debug(f"hello {name}")
    ...     return name
    >>> my_function(name="pinder")
    'pinder'
    >>> # 2023-11-01 10:23:20,456 | __main__.my_function:3 | DEBUG : hello pinder
    """

    def decorator_function(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper_function(*args: P.args, **kwargs: P.kwargs) -> T:
            if "log" in kwargs:
                raise PinderLoggingError(
                    "variable 'log' is injected by inject_log decorator,"
                    "the log variable name should keep empty in:"
                    f"{func.__module__}"
                )

            if "log" not in inspect.getfullargspec(func).args:
                raise PinderLoggingError(
                    f"The function '{func.__module__}.{func.__name__}' "
                    "should contain a variable named 'log'"
                )

            logger_name = f"{func.__module__}.{func.__name__}"
            log = kwargs.pop(
                "log",
                setup_logger(
                    logger_name=logger_name, log_level=log_level, log_file=log_file
                ),
            )
            return func(*args, log=log, **kwargs)  # type: ignore

        return wrapper_function

    return decorator_function
