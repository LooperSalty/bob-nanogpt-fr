"""
Étape 4 — Démonstrations pédagogiques du Self-Attention.

Partie A : "le truc matriciel" — agréger le passé d'un token via une
           multiplication matricielle triangulaire (3 formulations équivalentes,
           dont la dernière EST la forme utilisée par l'attention).
Partie B : une vraie tête d'attention (Q/K/V) sur un batch réel, avec
           affichage de la matrice d'attention obtenue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

import config
from attention import Head
from data_loader import CharDataset
from step1_data import load_text
from tokenizer import CharTokenizer


def partie_a_truc_matriciel() -> None:
    print("=" * 64)
    print("PARTIE A — Le truc matriciel : un token agrège son passé")
    print("=" * 64)
    torch.manual_seed(config.SEED)
    B, T, C = 1, 5, 2  # minuscule, pour TOUT voir à l'œil nu
    x = torch.randn(B, T, C)

    # V1 — boucle explicite : moyenne des tokens passés (lent, mais limpide)
    xbow1 = torch.zeros(B, T, C)
    for b in range(B):
        for t in range(T):
            xbow1[b, t] = x[b, : t + 1].mean(dim=0)

    # V2 — multiplication matricielle avec une triangulaire normalisée
    wei2 = torch.tril(torch.ones(T, T))         # [T, T] triangulaire inférieure (1 = "visible")
    wei2 = wei2 / wei2.sum(dim=1, keepdim=True)  # chaque ligne somme à 1
    xbow2 = wei2 @ x                             # [T,T] @ [B,T,C] -> [B,T,C]

    # V3 — softmax + masque (la forme exacte de l'attention)
    tril = torch.tril(torch.ones(T, T))
    wei3 = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))  # futur = -inf
    wei3 = F.softmax(wei3, dim=-1)                                   # -inf -> proba 0
    xbow3 = wei3 @ x

    torch.set_printoptions(precision=3, sci_mode=False)
    print(f"\nMatrice 'moyenne du passé' (V2) [T,T]=[{T},{T}] :\n{wei2}")
    print(f"\nMême matrice via softmax+masque (V3) :\n{wei3}")
    identiques = torch.allclose(xbow1, xbow2) and torch.allclose(xbow1, xbow3)
    print(f"\nLes 3 versions donnent le MÊME résultat ? {identiques}")


def partie_b_self_attention() -> None:
    print("\n" + "=" * 64)
    print("PARTIE B — Une vraie tête de Self-Attention (Q/K/V causale)")
    print("=" * 64)
    torch.manual_seed(config.SEED)

    text = load_text(config.DATA_PATH)
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(
        text, tokenizer,
        train_frac=config.TRAIN_FRAC, block_size=config.BLOCK_SIZE,
        batch_size=config.BATCH_SIZE, device=config.DEVICE,
    )

    n_embd = config.N_EMBD
    head_size = 16

    # Q/K/V opèrent sur des VECTEURS : il faut d'abord embarquer les entiers.
    token_embedding = nn.Embedding(tokenizer.vocab_size, n_embd).to(config.DEVICE)
    head = Head(n_embd, head_size, config.BLOCK_SIZE).to(config.DEVICE)

    x_ids, _ = dataset.get_batch("train")  # [B, T] entiers
    x = token_embedding(x_ids)             # [B, T, n_embd]
    out = head(x)                          # [B, T, head_size]

    print(f"\nx_ids (entiers)   : {tuple(x_ids.shape)}   [B, T]")
    print(f"x (embeddings)    : {tuple(x.shape)}   [B, T, n_embd]")
    print(f"sortie de la tête : {tuple(out.shape)}   [B, T, head_size]")

    # Recalcul de la matrice d'attention pour l'afficher (1er exemple du batch).
    k = head.key(x)
    q = head.query(x)
    wei = q @ k.transpose(-2, -1) * head_size ** -0.5
    wei = wei.masked_fill(head.tril[: config.BLOCK_SIZE, : config.BLOCK_SIZE] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)

    torch.set_printoptions(precision=3, sci_mode=False)
    print(f"\nMatrice d'attention du 1er exemple [T, T] :\n{wei[0]}")
    print("\n-> Triangulaire inférieure (poids 0 sur le futur), chaque ligne somme à 1,")
    print("   et les poids sont NON uniformes : ils dépendent des données (via Q·K).")
    print(f"   Somme de chaque ligne : {[round(s, 3) for s in wei[0].sum(dim=-1).tolist()]}")


if __name__ == "__main__":
    partie_a_truc_matriciel()
    partie_b_self_attention()
