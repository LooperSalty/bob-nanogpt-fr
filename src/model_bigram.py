"""
Étape 3 — Bigram Language Model.

Le modèle le plus simple possible : chaque token prédit le suivant via une
unique table de lookup, SANS aucun contexte long. D'où "bi-gramme" : la
prédiction ne dépend QUE du token courant (le précédent).

La table a une shape [vocab_size, vocab_size] : sa ligne i contient
directement les logits (scores bruts, non normalisés) du prochain token,
sachant que le token courant vaut i.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # nn.Embedding = table de lookup [vocab_size, vocab_size].
        # Ici les "embeddings" SONT les logits du token suivant.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx : [B, T] d'entiers (indices de tokens)
        logits = self.token_embedding_table(idx)  # [B, T, C] avec C = vocab_size

        if targets is None:
            return logits, None

        # F.cross_entropy attend des shapes [N, C] et [N] : on aplatit B et T.
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)   # [B*T, C]
        targets_flat = targets.view(B * T)    # [B*T]
        loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Génération autorégressive : on prédit, on échantillonne, on rajoute, on recommence."""
        # idx : [B, T] = contexte courant
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                                # [B, T, C]
            logits = logits[:, -1, :]                            # [B, C] : SEUL le dernier pas compte
            probs = F.softmax(logits, dim=-1)                    # [B, C] : logits -> probabilités
            idx_next = torch.multinomial(probs, num_samples=1)   # [B, 1] : tirage selon les probas
            idx = torch.cat((idx, idx_next), dim=1)              # [B, T+1] : on allonge la séquence
        return idx
