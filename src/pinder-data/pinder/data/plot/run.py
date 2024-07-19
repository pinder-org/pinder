"""pinder_plot

pinder_plot --help
INFO: Showing help with the command 'pinder_plot -- --help'.

NAME
    pinder_plot

SYNOPSIS
    pinder_plot COMMAND

COMMANDS
    COMMAND is one of the following:

     annotations

     categories

     datasets

     difficulty

     families

     leaderboards

     leakage

     performance

     validity

pinder_plot annotations --help
INFO: Showing help with the command 'pinder_plot annotations -- --help'.

NAME
    pinder_plot annotations

SYNOPSIS
    pinder_plot annotations <flags>

FLAGS
    -f, --fig_dir=FIG_DIR
        Type: 'Path'
        Default: PosixPath('~/.local/share/pinder/2024-02/publication_figures')
    -o, --format=FORMAT
        Type: 'str'
        Default: 'pdf'
    -t, --theme_name=THEME_NAME
        Type: 'str'
        Default: 'light'

"""

from pinder.data.plot import (
    annotations,
    categories,
    datasets,
    difficulty,
    families,
    leaderboards,
    leakage,
    performance,
    validity,
)


def main() -> None:
    import fire

    fire.Fire(
        {
            "annotations": annotations.main,
            "categories": categories.main,
            "datasets": datasets.main,
            "difficulty": difficulty.main,
            "families": families.main,
            "leaderboards": leaderboards.main,
            "leakage": leakage.main,
            "performance": performance.main,
            "validity": validity.main,
        }
    )


if __name__ == "__main__":
    main()
