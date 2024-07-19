"""pinder_qc

NAME
    pinder_qc

SYNOPSIS
    pinder_qc COMMAND

COMMANDS
    COMMAND is one of the following:

     uniprot_leakage

     binding_leakage
       Extract ECOD paired binding site leakage for test and val splits.

     pfam_diversity
       Extract PFAM clan diversity and generate visualizations.

     sequence_leakage
       Extract sequence similarity / leakage by subsampling members in train split.

pinder_qc uniprot_leakage --help
NAME
    pinder_qc uniprot_leakage

SYNOPSIS
    pinder_qc uniprot_leakage <flags>

FLAGS
    -i, --index_path=INDEX_PATH
        Type: Optional['str | None']
        Default: None
    -s, --split=SPLIT
        Type: 'str'
        Default: 'test'

"""

from pinder.data.qc import (
    annotation_check,
    pfam_diversity,
    similarity_check,
    uniprot_leakage,
)


def main() -> None:
    import fire

    fire.Fire(
        {
            "uniprot_leakage": uniprot_leakage.uniprot_leakage_main,
            "binding_leakage": annotation_check.binding_leakage_main,
            "pfam_diversity": pfam_diversity.pfam_diversity_main,
            "sequence_leakage": similarity_check.sequence_leakage_main,
        }
    )


if __name__ == "__main__":
    main()
