"""
Étapes 5-7 — Le mini-GPT complet.

Étape 6 introduit le `Block` : un bloc Transformer = attention + feed-forward,
chacun entouré d'une RESIDUAL CONNECTION et précédé d'une LAYER NORM (pré-LN).
Le modèle empile N_LAYER de ces blocs.

Forward, étage par étage :
    idx [B, T] (entiers)
      -> token embedding        [B, T, n_embd]
      +  position embedding      [T,    n_embd]
      -> N_LAYER × Block         [B, T, n_embd]   (attention + FFN, résiduels + LN)
      -> LayerNorm finale        [B, T, n_embd]
      -> lm_head (linéaire)      [B, T, vocab]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import MultiHeadAttention
from feed_forward import FeedForward


class Block(nn.Module):
    """Un bloc Transformer : communication (attention) puis calcul (FFN).

    Deux idées clés de l'Étape 6 :
      - RESIDUAL : x = x + sous_couche(x). Le '+ x' crée une "autoroute à gradient"
        qui permet d'empiler des couches en profondeur sans que l'apprentissage casse.
      - PRÉ-LAYERNORM : on normalise l'entrée de chaque sous-couche (ln1/ln2), ce qui
        stabilise les statistiques d'activation et accélère/fiabilise l'entraînement.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # residual autour de l'attention
        x = x + self.ffwd(self.ln2(x))  # residual autour du feed-forward
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # LayerNorm finale (avant la projection)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                       # [B, T, n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # [T, n_embd]
        x = tok_emb + pos_emb                                           # [B, T, n_embd]
        x = self.blocks(x)                                              # [B, T, n_embd]
        x = self.ln_f(x)                                                # [B, T, n_embd]
        logits = self.lm_head(x)                                        # [B, T, vocab_size]

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]                # [B, <=block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                           # [B, vocab_size]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)             # [B, T+1]
        return idx
