"""
Tokenizer BPE (sous-mots) — étape "scaling".

Le char-level fait travailler le modèle lettre par lettre : il gaspille sa
capacité à apprendre l'orthographe. Le BPE (Byte-Pair Encoding) fusionne
itérativement les paires de symboles les plus fréquentes pour bâtir un
vocabulaire de SOUS-MOTS (ex: "guerre", "tion", "Premi"). Résultat : séquences
beaucoup plus courtes et mots mieux formés, à budget de paramètres égal.

Analogie SE : le BPE est littéralement un algorithme de COMPRESSION à dictionnaire.

Les tokens spéciaux de chat sont réservés directement dans le vocab BPE, donc le
même tokenizer sert au pré-entraînement ET au fine-tuning (pas de redimensionnement).

Usage : python src/bpe_tokenizer.py   # entraîne le BPE sur le corpus + démo
"""

from __future__ import annotations

import config

BPE_PATH = config.CHECKPOINT_DIR / "bpe.json"
SPECIALS = ["<|pad|>", "<|user|>", "<|bot|>", "<|end|>"]
DEFAULT_VOCAB = 8000


def train_bpe(text_path=config.DATA_PATH, vocab_size: int = DEFAULT_VOCAB, out=BPE_PATH):
    """Entraîne un BPE byte-level sur le corpus et le sauvegarde."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([str(text_path)], trainer)
    out.parent.mkdir(exist_ok=True)
    tok.save(str(out))
    return tok


class BPETokenizer:
    """Interface tokenizer (encode/decode/vocab_size) + tokens spéciaux de chat."""

    def __init__(self, path=BPE_PATH) -> None:
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size()
        self.PAD = self.tok.token_to_id("<|pad|>")
        self.USER = self.tok.token_to_id("<|user|>")
        self.BOT = self.tok.token_to_id("<|bot|>")
        self.END = self.tok.token_to_id("<|end|>")

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s).ids

    def encode_text(self, s: str) -> list[int]:  # alias utilisé par le code de chat
        return self.tok.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def render(self, conversation: list[dict]) -> tuple[list[int], list[bool]]:
        """Conversation -> (ids, trainable) ; loss masquée hors réponse du bot."""
        ids: list[int] = []
        trainable: list[bool] = []
        for turn in conversation:
            is_bot = turn["role"] == "bot"
            ids.append(self.BOT if is_bot else self.USER)
            trainable.append(False)
            tids = self.encode(turn["text"])
            ids.extend(tids)
            trainable.extend([is_bot] * len(tids))
            ids.append(self.END)
            trainable.append(is_bot)
        return ids, trainable


def _demo() -> None:
    if not BPE_PATH.exists():
        print("Entraînement du BPE sur le corpus...")
        train_bpe()
    tok = BPETokenizer()
    print(f"vocab_size : {tok.vocab_size}")
    print(f"spéciaux   : pad={tok.PAD} user={tok.USER} bot={tok.BOT} end={tok.END}")
    s = "La Première Guerre mondiale a commencé en 1914."
    ids = tok.encode(s)
    pieces = [tok.tok.id_to_token(i) for i in ids]
    print(f"\ntexte         : {s}")
    print(f"{len(ids)} sous-mots : {pieces}")
    print(f"(char-level   : {len(s)} tokens)")
    print(f"decode        : {tok.decode(ids)}")


if __name__ == "__main__":
    _demo()
