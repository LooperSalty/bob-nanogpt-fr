"""
Étape 5-6 — Feed-Forward (MLP par token), avec dropout (Étape 6).

    - attention    = COMMUNICATION : les tokens échangent de l'info (axe T)
    - feed-forward = CALCUL        : chaque token "réfléchit" seul (axe C)

Expansion interne ×4 (n_embd -> 4*n_embd -> n_embd), reprise de "Attention is All You Need".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # projection vers une dim plus large
            nn.ReLU(),                       # non-linéarité (sinon = simple matrice)
            nn.Linear(4 * n_embd, n_embd),   # retour à la dimension d'origine
            nn.Dropout(dropout),             # régularisation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, T, n_embd] -> [B, T, n_embd]
