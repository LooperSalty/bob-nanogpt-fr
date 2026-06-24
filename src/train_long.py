"""
train_long.py — Entraînement long de Bob (>= 2 h), pensé pour tourner SANS surveillance.

Phase 1 — Pré-entraînement CHRONOMÉTRÉ sur le gros corpus FR (Wikipédia), en
          bfloat16 (Tensor Cores), avec checkpoints + meta.json réguliers
          (le dashboard web suit en direct).
Phase 2 — Fine-tuning conversationnel automatique (SFT) -> Bob sait discuter
          et répond aux connaissances de base du dataset chat.

Robuste : sauvegarde à chaque éval, et un checkpoint final même si interrompu.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch

import config
import train_chat
from bpe_tokenizer import BPE_PATH, BPETokenizer, train_bpe
from data_loader import CharDataset
from gpt import GPTLanguageModel
from step1_data import load_text
from train import estimate_loss, save_checkpoint

MODEL_CKPT = config.CHECKPOINT_DIR / "model.pt"
STATE_PATH = config.CHECKPOINT_DIR / "train_state.pt"


def amp_ctx():
    """Autocast bfloat16 sur GPU (≈2× plus rapide, moins de VRAM)."""
    if config.DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def pretrain() -> None:
    torch.manual_seed(config.SEED)

    text = load_text(config.DATA_PATH)
    if not BPE_PATH.exists():
        train_bpe()
    tok = BPETokenizer()
    ds = CharDataset(
        text, tok,
        train_frac=config.TRAIN_FRAC, block_size=config.BLOCK_SIZE,
        batch_size=config.BATCH_SIZE, device=config.DEVICE,
    )
    model = GPTLanguageModel(
        tok.vocab_size, config.N_EMBD, config.BLOCK_SIZE,
        config.N_HEAD, config.N_LAYER, config.DROPOUT,
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"[PRETRAIN] device={config.DEVICE} | corpus={len(text):,} car. | "
          f"vocab={tok.vocab_size} | params={n_params:,}", flush=True)
    print(f"[PRETRAIN] cible : {config.TRAIN_SECONDS / 3600:.1f} h", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.1)

    # --- Reprise (resume) : on repart du dernier checkpoint si disponible ---
    history: list[dict] = []
    start_it = 0
    elapsed_offset = 0.0
    if MODEL_CKPT.exists():
        sd = torch.load(MODEL_CKPT, map_location=config.DEVICE, weights_only=True)
        if sd["token_embedding_table.weight"].shape[0] == tok.vocab_size:
            model.load_state_dict(sd)
            print(f"[RESUME] poids rechargés depuis {MODEL_CKPT.name}", flush=True)
            if STATE_PATH.exists():
                st = torch.load(STATE_PATH, map_location=config.DEVICE, weights_only=True)
                opt.load_state_dict(st["opt"])
                start_it = int(st["iter"])
                elapsed_offset = float(st["elapsed"])
                history = st.get("history", [])
                print(f"[RESUME] reprise exacte à iter {start_it} "
                      f"({elapsed_offset / 60:.1f} min déjà faites)", flush=True)
        else:
            print("[RESUME] vocab différent -> nouveau modèle (pas de reprise)", flush=True)

    start = time.time() - elapsed_offset
    it = start_it
    try:
        while it <= config.MAX_ITERS:
            if it % config.EVAL_INTERVAL == 0:
                with amp_ctx():
                    losses = estimate_loss(model, ds, config.EVAL_ITERS)
                elapsed = time.time() - start
                history.append({"iter": it, "train": round(losses["train"], 4),
                                "val": round(losses["val"], 4), "t": round(elapsed, 1)})
                print(f"[PRETRAIN] iter {it:6d} | train {losses['train']:.4f} | "
                      f"val {losses['val']:.4f} | {elapsed / 60:6.1f} min", flush=True)
                save_checkpoint(model, tok, history, elapsed, n_params, status="training")
                torch.save({"opt": opt.state_dict(), "iter": it, "elapsed": elapsed,
                            "history": history}, STATE_PATH)
                if elapsed >= config.TRAIN_SECONDS:
                    break

            xb, yb = ds.get_batch("train")
            with amp_ctx():
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            it += 1
    finally:
        elapsed = time.time() - start
        save_checkpoint(model, tok, history, elapsed, n_params, status="done")
        print(f"[PRETRAIN] terminé : {it} iters en {elapsed / 60:.1f} min", flush=True)


def main() -> None:
    pretrain()
    print("\n[SFT] fine-tuning conversationnel sur le nouveau modèle de base...\n", flush=True)
    train_chat.main()
    print("\n[DONE] Bob est prêt : modèle de base + assistant.", flush=True)


if __name__ == "__main__":
    main()
