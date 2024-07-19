from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Protein:
    """A class to represent a Protein.

    Attributes
    ----------
    source : str
        The source of the protein.
    uniprot : str
        The uniprot ID of the protein.
    chain : Optional[str]
        The chain of the protein, can be None.
    from_residue : Optional[int]
        The starting residue of the protein, can be None.
    to_residue : Optional[int]
        The ending residue of the protein, can be None.

    Methods
    -------
    __str__():
        Returns a string representation of the protein.

    Examples
    --------
    >>> protein = Protein(source='6q0r', chain='B', uniprot='Q66K64')
    >>> str(protein) == '6q0r__B_Q66K64'
    True

    >>> protein = Protein(source='af2', chain='A', uniprot='Q14498', from_residue=1, to_residue=100)
    >>> str(protein) == 'af2__A_Q14498_1_100'
    False
    """

    source: str
    uniprot: str
    chain: str | None = "A"
    from_residue: int | None = None
    to_residue: int | None = None

    def __str__(self) -> str:
        if len(self.source.lower()) == 4:
            return f"{self.source}__{self.chain}_{self.uniprot}"
        else:
            if self.from_residue is None or self.to_residue is None:
                return f"{self.source}__{self.uniprot}"
            return (
                f"{self.source}__{self.uniprot}_{self.from_residue}_{self.to_residue}"
            )


@dataclass
class Monomer:
    """A class to represent a Monomer.

    Attributes
    ----------
    proteins : List[Protein]
        A list of Protein objects that make up the monomer.
    side : Optional[str]
        The side of the monomer, can be None.

    Methods
    -------
    __str__():
        Returns a string representation of the monomer.

    from_string(monomer_str: str):
        Parses a monomer string into a Monomer object.

    Examples
    --------
    >>> filename = "6q0r__B_Q66K64.pdb"
    >>> parsed_monomer = Monomer.from_string(filename)
    >>> print(parsed_monomer)
    6q0r__B_Q66K64
    >>> assert filename == str(parsed_monomer) + '.pdb'

    >>> print(Monomer([Protein(source='af2', chain='A', uniprot='Q14498', from_residue=1, to_residue=100)]))
    af2__Q14498_1_100
    >>> print(Monomer([Protein(source='af2', chain='A', uniprot='Q14498', from_residue=1, to_residue=100)], side='L'))
    af2__Q14498_1_100-L
    >>> print(Monomer([Protein(source='6q0r', chain='B', uniprot='Q66K64')], side='R'))
    6q0r__B_Q66K64-R
    >>> print(Monomer.from_string("af2__Q14498_1_100-R.pdb"))
    af2__Q14498_1_100-R
    >>> print(Monomer([Protein(source='af2', chain='A', uniprot='Q14498')], side='R'))
    af2__Q14498-R
    >>> print(Monomer.from_string("af2__Q14498-R.pdb"))
    af2__Q14498-R
    """

    proteins: list[Protein]
    side: str | None = None

    def __str__(self) -> str:
        proteins_str = "-".join([str(p) for p in self.proteins])
        if self.side:
            proteins_str += "-" + self.side
        return proteins_str

    @classmethod
    def from_string(cls, monomer_str: str) -> "Monomer":
        """parse a monomer string into a Monomer object
        supports when monomer is a .pdb file or a part of a complex
        """
        monomer_str = monomer_str.rstrip(".pdb")
        isoform_pattern = r"-\d+$"
        side = None
        # Check for split dimer, remove -R/L in case uniprot is isoform with -
        if monomer_str.endswith("-R"):
            side = "R"
            monomer_str = monomer_str.split("-R")[0]
            protein_strs = [monomer_str]
        elif monomer_str.endswith("-L"):
            side = "L"
            monomer_str = monomer_str.split("-L")[0]
            protein_strs = [monomer_str]
        elif bool(re.search(isoform_pattern, monomer_str)):
            protein_strs = [monomer_str]
        else:
            protein_strs = monomer_str.split("-")

        proteins = []
        for protein_str in protein_strs:
            if len(protein_str) == 1:
                side = protein_str
                continue
            parts = protein_str.split("__")
            source = parts[0]
            qualifiers = parts[1].split("_")
            if len(source) != 4:
                if len(qualifiers) == 1:  # only the uniprot
                    uniprot = qualifiers[0]
                    protein = Protein(source=source, uniprot=uniprot)
                elif len(qualifiers) == 3:
                    uniprot, from_residue, to_residue = qualifiers
                    protein = Protein(
                        source=source,
                        uniprot=uniprot,
                        chain="A",
                        from_residue=int(from_residue),
                        to_residue=int(to_residue),
                    )
                else:
                    raise ValueError(
                        f"Invalid non-PDB protein string: {protein_str}. Either provide a uniprot or a uniprot and a range of residues"
                    )
            else:  # Assume it's a PDB entry
                if len(qualifiers) == 2:
                    chain, uniprot = qualifiers
                else:
                    raise ValueError(
                        f"Invalid PDB protein string: {protein_str}. Provide a chain and a uniprot"
                    )
                protein = Protein(source, uniprot, chain)
            proteins.append(protein)
        return cls(proteins, side=side)


@dataclass
class Dimer:
    """A class used to represent a Dimer, which is a complex of two Monomers.

    Attributes
    ----------
    monomer1 : Monomer
        The first monomer in the dimer
    monomer2 : Monomer
        The second monomer in the dimer

    Methods
    -------
    __str__():
        Returns a string representation of the dimer.

    Examples
    --------
    >>> monomer1 = Monomer.from_string("6q0r__B_Q66K64")
    >>> monomer2 = Monomer.from_string("6q0r__D_Q14498")
    >>> dimer = Dimer(monomer1, monomer2)
    >>> assert str(dimer) == "6q0r__B_Q66K64--6q0r__D_Q14498"
    >>> print(dimer)
    6q0r__B_Q66K64--6q0r__D_Q14498
    """

    monomer1: Monomer
    monomer2: Monomer

    def __str__(self) -> str:
        return f"{str(self.monomer1)}--{str(self.monomer2)}"

    @classmethod
    def from_string(cls, dimer_str: str) -> "Dimer":
        dimer_str = dimer_str.rstrip(".pdb")
        monomer_strs = dimer_str.split("--")
        monomer1 = Monomer.from_string(monomer_strs[0])
        monomer2 = Monomer.from_string(monomer_strs[1])
        return cls(monomer1, monomer2)
