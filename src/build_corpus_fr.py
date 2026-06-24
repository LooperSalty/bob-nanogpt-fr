"""
Construit un gros corpus FRANÇAIS propre pour entraîner Bob.

Source : Wikipédia FR (via HuggingFace `datasets`, en streaming -> pas de
téléchargement géant). On nettoie le texte pour garder un jeu de caractères
"français" raisonnable (sinon le vocab char-level exploserait avec des milliers
de symboles Unicode rares).

Usage :  python build_corpus_fr.py [taille_Mo]   (défaut : 25 Mo)
"""

from __future__ import annotations

import re
import sys

import config

OUT = config.DATA_PATH
TARGET_MB = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
TARGET = int(TARGET_MB * 1024 * 1024)

ALLOWED = set(
    " \n"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "àâäéèêëîïôöùûüçœæÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒÆ"
    ".,;:!?'\"()-«»…%/"
)


def clean(text: str) -> str:
    text = text.replace("’", "'").replace("\t", " ").replace("\r", "")
    text = "".join(ch if ch in ALLOWED else " " for ch in text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    from datasets import load_dataset

    print(f"Cible : {TARGET_MB} Mo de français propre -> {OUT}", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)

    size = 0
    next_mark = 5 * 1024 * 1024
    n_articles = 0
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for art in ds:
            t = clean(art.get("text", ""))
            if len(t) < 300:
                continue
            f.write(t)
            f.write("\n\n")
            size += len(t) + 2
            n_articles += 1
            if size >= next_mark:
                print(f"  ... {size / 1024 / 1024:.1f} Mo  ({n_articles} articles)", flush=True)
                next_mark += 5 * 1024 * 1024
            if size >= TARGET:
                break

    print(f"Terminé : {size / 1024 / 1024:.1f} Mo, {n_articles} articles écrits.", flush=True)


if __name__ == "__main__":
    main()
