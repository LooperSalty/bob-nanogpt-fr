"""
Étape 2 — Génération des batchs (X, Y).

À partir du tenseur 1D produit à l'Étape 1 :
    1. split train / validation
    2. échantillonnage aléatoire de B fenêtres de longueur T
    3. X [B, T] = contextes ; Y [B, T] = mêmes fenêtres décalées de +1 (les cibles)

Idée clé : une fenêtre de T tokens contient en réalité T exemples
d'entraînement imbriqués — prédire le token t+1 à partir du contexte [0..t],
et ce pour chaque t de 0 à T-1.
"""

from __future__ import annotations

import torch

import config
from step1_data import load_text
from tokenizer import CharTokenizer


class CharDataset:
    """Corpus tokenisé + split train/val + échantillonnage de batchs sur le bon device."""

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        *,
        train_frac: float,
        block_size: int,
        batch_size: int,
        device: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # [N]
        n_train = int(train_frac * len(data))
        self.train_data = data[:n_train]  # [N_train]
        self.val_data = data[n_train:]    # [N_val]

    def _data_for(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_data
        if split == "val":
            return self.val_data
        raise ValueError(f"split inconnu : {split!r} (attendu 'train' ou 'val').")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Renvoie (x, y), chacun de shape [B, T], déjà placés sur self.device."""
        data = self._data_for(split)

        # B positions de départ tirées au hasard dans [0, len(data) - block_size).
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))  # [B]

        # Pour chaque départ i : x = data[i : i+T] ; y = data[i+1 : i+1+T] (décalé de +1).
        x = torch.stack([data[i : i + self.block_size] for i in ix])          # [B, T]
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])  # [B, T]

        return x.to(self.device), y.to(self.device)


def _demo() -> None:
    torch.manual_seed(config.SEED)

    text = load_text(config.DATA_PATH)
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(
        text,
        tokenizer,
        train_frac=config.TRAIN_FRAC,
        block_size=config.BLOCK_SIZE,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
    )

    print(f"Device              : {config.DEVICE}")
    print(f"Tokens train / val  : {len(dataset.train_data):,} / {len(dataset.val_data):,}")
    print(f"block_size (T)      : {config.BLOCK_SIZE}")
    print(f"batch_size (B)      : {config.BATCH_SIZE}")

    x, y = dataset.get_batch("train")
    print(f"\nx.shape = {tuple(x.shape)}  (B, T)   device = {x.device}")
    print(f"y.shape = {tuple(y.shape)}  (B, T)   device = {y.device}")
    print(f"\nx =\n{x}")
    print(f"\ny =\n{y}")

    # --- Le point clé : une fenêtre de T tokens = T exemples imbriqués ---
    print("\n--- Déroulé de la 1re séquence du batch (contexte -> cible) ---")
    seq_x, seq_y = x[0].tolist(), y[0].tolist()
    for t in range(config.BLOCK_SIZE):
        context_ids = seq_x[: t + 1]
        target_id = seq_y[t]
        ctx_txt = tokenizer.decode(context_ids)
        tgt_txt = tokenizer.decode([target_id])
        print(f"  t={t}: contexte {str(context_ids):<28} ({ctx_txt!r:>14}) -> cible {target_id:>2} ({tgt_txt!r})")


if __name__ == "__main__":
    _demo()
