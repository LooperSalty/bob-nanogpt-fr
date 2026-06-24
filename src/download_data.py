"""
Télécharge un corpus FRANÇAIS du domaine public (Project Gutenberg),
retire l'en-tête / pied de page Gutenberg (rédigés en anglais, qui
pollueraient le vocabulaire), puis écrit le résultat dans data/input.txt.

On essaie plusieurs œuvres jusqu'à en trouver une qui se télécharge et
qui « ressemble » bien à du français (présence d'accents + taille suffisante).
"""

import re
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "data" / "input.txt"

# Textes français, UTF-8, riches en dialogues (bon pour la génération).
CANDIDATES = [
    ("Les Misérables T1 - Fantine (Hugo)", "https://www.gutenberg.org/files/17489/17489-0.txt"),
    ("Le Comte de Monte-Cristo T1 (Dumas)", "https://www.gutenberg.org/cache/epub/17989/pg17989.txt"),
    ("Les Trois Mousquetaires (Dumas)", "https://www.gutenberg.org/cache/epub/13951/pg13951.txt"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (educational nanoGPT dataset fetch)"}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def strip_gutenberg(text: str) -> str:
    """Garde uniquement le corps de l'œuvre, entre les marqueurs *** START/END ***."""
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF.*?\*\*\*", text)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    return text[s:e].strip()


def looks_french(text: str) -> bool:
    accents = sum(text.count(c) for c in "éèêàçùôîâ")
    return len(text) > 150_000 and accents > 1000


def main() -> None:
    for name, url in CANDIDATES:
        try:
            print(f"Essai : {name} ...")
            body = strip_gutenberg(fetch(url))
            if looks_french(body):
                OUT.write_text(body, encoding="utf-8")
                print(f"OK -> {name} : {len(body):,} caractères écrits dans {OUT}")
                return
            print(f"  rejeté (longueur={len(body):,}, ne semble pas français)")
        except Exception as exc:
            print(f"  échec : {exc}")
    print("Aucun corpus français n'a pu être téléchargé.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
