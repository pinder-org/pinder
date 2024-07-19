from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch
from typing import Any

import pytest
from pinder.core import log


def pytest_parametrize_dict(
    data: dict[str, dict[str, Any]], **kwargs
) -> pytest.mark.structures.MarkDecorator:
    """
    Formatted parameters for pytest mark parametrize

    adopted from:
        https://github.com/pytest-dev/pytest/issues/7568#issuecomment-1217328487

    :param dict[str, dict[str, Any]] data: _description_
    :param **kwargs: pytest.mark.parametrize key word arguments
    :return pytest.mark.structures.MarkDecorator: pytest.mark.parametrize decorator

    """
    ids = list(data.keys())
    arg_names = set(list(data.values())[0].keys())
    for test_params in data.values():
        if arg_names != set(test_params.keys()):
            raise ValueError("All test cases must have the same parameter names")

    formatted_data = [[item[a] for a in arg_names] for item in data.values()]
    return pytest.mark.parametrize(arg_names, formatted_data, ids=ids, **kwargs)


@pytest_parametrize_dict(
    {
        "Test case: no log file, with logger name": {
            "logger_name": "some_logger_name",
            "log_level": None,
            "log_file": None,
            "expected_logger_name": "some_logger_name",
            "expected_logger_level": logging.INFO,
        },
        "Test case: with log file, no logger name": {
            "logger_name": None,
            "log_level": logging.DEBUG,
            "log_file": "log_file.txt",
            "expected_logger_name": "test_log.py",
            "expected_logger_level": logging.DEBUG,
        },
    }
)
def test_setup_logger(
    logger_name,
    log_level,
    log_file,
    expected_logger_name,
    expected_logger_level,
    pinder_temp_dir,
):
    """
    System under testing: log.setup_logger
    Collaborators: None

    Parameters
    ----------
    logger_name : str
        the input logger name to the function
    log_level : Optional[int]
        the input log level to the function
    log_file: Optional[str]
        the input log file to the function
    expected_logger_name : str
        the expected logger's name
    expected_logger_level : int
        the expected logger's log level
    """
    if log_level is None:
        logger = log.setup_logger(logger_name, log_file=log_file)
    else:
        logger = log.setup_logger(
            logger_name, log_level, log_file=pinder_temp_dir / log_file
        )

    assert logger.name == expected_logger_name
    assert logger.level == expected_logger_level

    if log_file is not None:
        assert len(logger.handlers) == 2
        assert (
            logger.handlers[1].baseFilename == (pinder_temp_dir / log_file).as_posix()
        )
    else:
        assert len(logger.handlers) == 1


def test_setup_logger_level_already_set():
    logger = log.setup_logger("pinder.module.default")
    assert logger.level == log.DEFAULT_LOGGING_LEVEL

    logging.getLogger("pinder.module.critical").setLevel(logging.CRITICAL)
    logger = log.setup_logger("pinder.module.critical")
    assert logger.level == logging.CRITICAL


def test_setup_logger_non_duplicated():
    """
    System under testing: log.setup_logger
    Collaborators: None

    test that handlers won't be added multiple times
    """
    logger = log.setup_logger("some logger name")
    logger = log.setup_logger("some logger name")
    assert len(logger.handlers) == 1


@pytest.mark.skipif(sys.version_info < (3, 8), reason="no py37 support")
@patch("pinder.core.log.logging.getLogger")
@pytest.mark.parametrize(
    "log_level,log_file,expected_log_handler_count",
    [(logging.DEBUG, "log_file.txt", 2), (logging.INFO, None, 1)],
)
def test_inject_logger(
    mock_get_logger, log_level, log_file, expected_log_handler_count, pinder_temp_dir
):
    """
    System under testing: log.inject_logger
    Collaborators:
        - log.setup_logger

    Parameters
    ----------
    mock_get_logger :
        Patch object to mock the logging.getLogger method
    log_level : int
        the input log level integer
    log_file : _type_
        the input log file path
    expected_log_handler_count : _type_
        the expected call count for calling logging.addHandler
    """
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    # construct the function being tested
    if log_file is not None:
        log_file = pinder_temp_dir / log_file

    @log.inject_logger(log_level=log_level, log_file=log_file)
    def my_function(name, log):
        log.info(f"Hello {name}")
        return name

    # Actual invocation of the function being tested and
    # test that the function works properly as to returning the expected
    # result
    assert "Pinder" == my_function("Pinder")

    # test that logger is constructed as expected in the decorator
    mock_get_logger.assert_called_once_with("test_log.my_function")
    mock_get_logger.setLevel(log_level)

    # test that logger is called as expected in the test function
    mock_logger.info.assert_called_once_with("Hello Pinder")

    # test that log handlers are constructed as expected
    # 1 for stream handler and another one for file handler if a log file is specified
    assert mock_logger.addHandler.call_count == expected_log_handler_count

    # such that the stream handler is always there
    assert (
        mock_logger.addHandler.call_args_list[0].args[0].__class__
        == logging.StreamHandler
    )

    # and the file handler is constructed when log file is provided
    if log_file is not None:
        assert (
            mock_logger.addHandler.call_args_list[1].args[0].__class__
            == logging.FileHandler
        )
        assert (
            mock_logger.addHandler.call_args_list[1].args[0].baseFilename
            == (pinder_temp_dir / log_file).as_posix()
        )


def test_inject_logger__Exception_bad_function():
    """
    System under testing: log.inject_logger
    Collaborators: None

    Test that we can raise informative error when the input function
    does not conform to the expected signature.
    """

    @log.inject_logger()
    def my_bad_function(name):
        # missing a log parameter
        log.info(f"Hello {name}")
        return name

    # test that the function works properly
    with pytest.raises(
        log.PinderLoggingError,
        match=(
            "The function 'test_log.my_bad_function' "
            "should contain a variable named 'log'"
        ),
    ):
        my_bad_function("Pinder")


def test_inject_logger__Exception_bad_input():
    """
    System under testing: log.inject_logger
    Collaborators: None

    Test that we can raise informative error when the input function
    does not conform to the expected signature.
    """

    @log.inject_logger()
    def my_bad_function(name, log):
        # missing a log parameter
        log.info(f"Hello {name}")
        return name

    # test that the function works properly
    with pytest.raises(
        log.PinderLoggingError,
        match=(
            "variable 'log' is injected by inject_log decorator,"
            "the log variable name should keep empty in:"
        ),
    ):
        my_bad_function("Pinder", log=None)


@patch("pinder.core.log.logging.getLogger")
def test_inject_logger__Exception_in_inner_function(mock_get_logger):
    """
    System under testing: log.inject_logger
    Collaborators: None

    Test that exception within the input function is still
    probably raised
    """
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    @log.inject_logger()
    def my_bad_bad_function(log):
        # raise a TypeError
        return 1 + "a"

    expected_err_msg = "unsupported operand type(s) for +: 'int' and 'str'"
    with pytest.raises(Exception) as e:
        my_bad_bad_function()
        assert str(e) == expected_err_msg
