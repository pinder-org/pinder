import pytest


def test_data_run_main_exit():
    from pinder.data.run import main

    with pytest.raises(SystemExit):
        main()
