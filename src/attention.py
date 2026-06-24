"""
Étapes 4-6 — Self-Attention causale : tête unique (Head) et multi-têtes
(MultiHeadAttention), avec dropout (régularisation ajoutée à l'Étape 6).

Mécanique d'une tête :
    affinités = Q · Kᵀ        -> à quel point le token i s'intéresse au token j
    /sqrt(head_size)          -> mise à l'échelle (stabilise le softmax)
    masque causal             -> interdit de regarder le futur (j > i)
    softmax                   -> poids d'attention (chaque ligne somme à 1)
    sortie    = poids · V     -> moyenne des Values pondérée par l'attention

Analogie SE : un dictionnaire clé→valeur DIFFÉRENTIABLE et FLOU. Au lieu de
renvoyer UNE valeur pour une clé exacte (HashMap.get), on renvoie un mélange
pondéré de TOUTES les valeurs, selon la ressemblance query/clé.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Une tête de self-attention causale : [B, T, n_embd] -> [B, T, head_size]."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # tril : masque triangulaire constant (PAS un paramètre apprenable).
        # register_buffer -> suit le modèle sur GPU sans être optimisé.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]     # longueur de séquence courante (pour le masque causal)
        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5           # [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # [B, T, T]
        wei = F.softmax(wei, dim=-1)                                  # [B, T, T]
        wei = self.dropout(wei)                                       # régularisation

        v = self.value(x)  # [B, T, head_size]
        out = wei @ v      # [B, T, head_size]
        return out


class MultiHeadAttention(nn.Module):
    """Plusieurs têtes en parallèle, puis concaténation + projection + dropout."""

    def __init__(
        self, n_head: int, head_size: int, n_embd: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)  # mélange les têtes recollées
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, head_size * n_head]
        out = self.dropout(self.proj(out))                   # [B, T, n_embd]
        return out
