import pytest


def test_submission_empty():
    from pinder.eval.create_submission import main

    with pytest.raises(SystemExit):
        main([])


def test_submission_eval_dir(pinder_eval_dir):
    from pinder.eval.create_submission import main

    main(["--eval_dir", pinder_eval_dir.as_posix()])
    (pinder_eval_dir.parent / "eval_example.zip").unlink()
