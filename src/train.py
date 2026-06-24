"""
Boucle d'entraînement du mini-GPT.

À chaque EVAL_INTERVAL : on mesure la loss train/val, et on SAUVEGARDE
  - checkpoints/model.pt   : les poids du modèle
  - checkpoints/meta.json  : config + vocabulaire + historique des loss + timings
Ces fichiers sont lus en direct par le serveur web (dashboard temps réel).

Écritures ATOMIQUES (tmp -> os.replace) pour que le serveur ne lise jamais un
fichier à moitié écrit pendant l'entraînement.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import torch

import config
from data_loader import CharDataset
from gpt import GPTLanguageModel
from step1_data import load_text
from tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataset: CharDataset, eval_iters: int) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def _atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Écrit via un fichier .tmp puis os.replace (atomique). Petite reprise sur Windows."""
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)  # dernier essai (laisse remonter en cas d'échec)


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    history: list[dict],
    elapsed: float,
    n_params: int,
    status: str,
) -> None:
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    _atomic_write(config.CHECKPOINT_DIR / "model.pt", lambda p: torch.save(model.state_dict(), p))

    meta = {
        "status": status,  # "training" | "done"
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": config.N_EMBD,
            "n_head": config.N_HEAD,
            "n_layer": config.N_LAYER,
            "block_size": config.BLOCK_SIZE,
            "batch_size": config.BATCH_SIZE,
            "dropout": config.DROPOUT,
            "learning_rate": config.LEARNING_RATE,
            "max_iters": config.MAX_ITERS,
        },
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "device": config.DEVICE,
        "elapsed_sec": round(elapsed, 1),
        "history": history,
    }
    _atomic_write(
        config.CHECKPOINT_DIR / "meta.json",
        lambda p: p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"),
    )


def main() -> None:
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

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=config.N_EMBD,
        block_size=config.BLOCK_SIZE,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device                 : {config.DEVICE}")
    print(f"Paramètres             : {n_params:,}")
    print(f"Loss attendue à l'init : ~{math.log(tokenizer.vocab_size):.4f}")
    print("-" * 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    history: list[dict] = []
    start = time.time()
    for it in range(config.MAX_ITERS + 1):
        if it % config.EVAL_INTERVAL == 0:
            losses = estimate_loss(model, dataset, config.EVAL_ITERS)
            elapsed = time.time() - start
            history.append(
                {
                    "iter": it,
                    "train": round(losses["train"], 4),
                    "val": round(losses["val"], 4),
                    "t": round(elapsed, 1),
                }
            )
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:6.1f}s")
            save_checkpoint(model, tokenizer, history, elapsed, n_params, status="training")

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start
    save_checkpoint(model, tokenizer, history, elapsed, n_params, status="done")
    print("-" * 64)
    print(f"Terminé en {elapsed:.1f}s. Checkpoint -> {config.CHECKPOINT_DIR}")

    context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
    sample = tokenizer.decode(model.generate(context, max_new_tokens=400)[0].tolist())
    print("\n--- Échantillon ---\n" + sample)


if __name__ == "__main__":
    main()
