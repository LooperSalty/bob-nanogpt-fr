"""
Étape 8a — ChatTokenizer : ajoute TOKENS SPÉCIAUX (+ caractères « extra ») au
tokenizer caractère.

IDs :
    0 .. V-1       : les V caractères du corpus (INCHANGÉS -> réutilise les
                     poids du modèle pré-entraîné)
    V .. V+E-1     : caractères EXTRA absents du corpus mais utiles (ex: '_'
                     pour les pseudos comme Looper_Salty)
    V+E .. +S      : <|user|>, <|bot|>, <|end|>, <|pad|>

Les lignes nouvelles (extra + spéciaux) sont initialisées au hasard puis
apprises pendant le fine-tuning (le grossissement des embeddings s'en charge).

`render()` renvoie aussi un masque `trainable` : on n'entraîne la loss QUE sur
la réponse de l'assistant (+ son <|end|>), jamais sur la question utilisateur.
"""

from __future__ import annotations

from tokenizer import CharTokenizer


class ChatTokenizer:
    SPECIALS = ["<|user|>", "<|bot|>", "<|end|>", "<|pad|>"]
    EXTRA = ["_"]  # caractères utiles absents du corpus Wikipédia

    def __init__(self, base: CharTokenizer) -> None:
        self.base = base

        # Caractères de base : IDs identiques à la base.
        self._stoi = {ch: i for i, ch in enumerate(base.vocab)}
        n = base.vocab_size

        # Caractères extra : IDs juste après la base.
        for i, ch in enumerate(self.EXTRA):
            self._stoi[ch] = n + i
        n2 = n + len(self.EXTRA)

        # Tokens spéciaux : IDs après les extra.
        self.special_to_id = {tok: n2 + i for i, tok in enumerate(self.SPECIALS)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}

        self._itos = {i: ch for ch, i in self._stoi.items()}  # base + extra
        self.vocab_size = n2 + len(self.SPECIALS)

        self.USER = self.special_to_id["<|user|>"]
        self.BOT = self.special_to_id["<|bot|>"]
        self.END = self.special_to_id["<|end|>"]
        self.PAD = self.special_to_id["<|pad|>"]

    def encode_text(self, s: str) -> list[int]:
        """Encode du texte (base + extra) ; ignore les caractères inconnus."""
        return [self._stoi[c] for c in s if c in self._stoi]

    def decode(self, ids: list[int], keep_specials: bool = False) -> str:
        parts: list[str] = []
        for i in ids:
            if i in self.id_to_special:
                if keep_specials:
                    parts.append(self.id_to_special[i])
            elif i in self._itos:
                parts.append(self._itos[i])
        return "".join(parts)

    def render(self, conversation: list[dict]) -> tuple[list[int], list[bool]]:
        """Conversation -> (ids, trainable).

        trainable[k] = True si le token k contribue à la loss
                       (= texte de l'assistant + son <|end|>).
        """
        ids: list[int] = []
        trainable: list[bool] = []
        for turn in conversation:
            is_bot = turn["role"] == "bot"
            ids.append(self.BOT if is_bot else self.USER)
            trainable.append(False)

            text_ids = self.encode_text(turn["text"])
            ids.extend(text_ids)
            trainable.extend([is_bot] * len(text_ids))

            ids.append(self.END)
            trainable.append(is_bot)
        return ids, trainable
