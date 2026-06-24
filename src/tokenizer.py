"""
Étape 1 — Tokenizer caractère par caractère.

Rôle : transformer du texte (str) en une séquence d'entiers (token IDs),
et inversement. Ici l'unité atomique est le *caractère*.

Analogies ingénierie logicielle :
    - encode / decode : sérialisation / désérialisation
    - stoi / itos     : deux HashMaps inverses (bijection caractère <-> entier)
"""

from __future__ import annotations


class CharTokenizer:
    """Tokenizer déterministe au niveau caractère.

    Le vocabulaire est figé à la construction à partir d'un corpus de
    référence : c'est l'ensemble *trié* des caractères uniques rencontrés.
    L'objet est immuable après __init__ (aucune méthode ne modifie son état).
    """

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError(
                "Corpus de référence vide : impossible de construire un vocabulaire."
            )

        # set()    -> caractères uniques
        # sorted() -> ordre déterministe (reproductibilité : même corpus => mêmes IDs)
        chars = sorted(set(text))

        self.vocab_size: int = len(chars)
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab(self) -> list[str]:
        """Caractères du vocabulaire, indexés par leur token ID."""
        return [self._itos[i] for i in range(self.vocab_size)]

    def encode(self, s: str) -> list[int]:
        """str -> list[int]. Lève KeyError si un caractère est hors-vocabulaire."""
        return [self._stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        """list[int] -> str. Inverse exact de encode()."""
        return "".join(self._itos[i] for i in ids)
