"""
Étape 1 — Chargement du dataset + construction du Tokenizer.

Pipeline de l'étape :
    fichier texte (str)
        -> CharTokenizer  (vocabulaire + bijection char<->int)
        -> tenseur d'entiers de shape [N]   (N = nombre total de caractères)

Ce tenseur 1D est la "matière première" que l'Étape 2 découpera en batchs (X, Y).
"""

from pathlib import Path

import torch

from tokenizer import CharTokenizer

DATA_PATH = Path(__file__).parent / "data" / "input.txt"


def load_text(path: Path) -> str:
    """Lit un fichier texte brut en UTF-8. Échoue tôt et clairement s'il manque."""
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {path}\n"
            "Place un fichier texte brut à cet emplacement."
        )
    return path.read_text(encoding="utf-8")


def main() -> None:
    text = load_text(DATA_PATH)
    print(f"Longueur du corpus      : {len(text):,} caractères")

    tokenizer = CharTokenizer(text)
    print(f"Taille du vocabulaire   : {tokenizer.vocab_size}")
    print(f"Vocabulaire             : {''.join(tokenizer.vocab)!r}")

    # --- Démonstration encode/decode : ce DOIT être une bijection parfaite ---
    sample = text[:32]
    ids = tokenizer.encode(sample)
    print(f"\nsample           : {sample!r}")
    print(f"encode(sample)   : {ids}")
    print(f"decode(...)      : {tokenizer.decode(ids)!r}")

    # Garde-fou : ré-encoder puis décoder tout le corpus doit redonner le texte.
    assert tokenizer.decode(tokenizer.encode(text)) == text, "encode/decode non bijectif !"

    # --- Encodage du corpus entier dans un tenseur PyTorch ---
    # dtype=torch.long (int64) : ces entiers serviront d'INDICES dans la table
    # d'embedding (nn.Embedding exige des indices entiers, pas des floats).
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"\nTenseur encodé   : shape={tuple(data.shape)}, dtype={data.dtype}")
    print(f"100 premiers IDs : {data[:100].tolist()}")


if __name__ == "__main__":
    main()
