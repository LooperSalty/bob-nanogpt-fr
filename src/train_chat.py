"""
Étape 8c — Fine-tuning conversationnel (SFT) du modèle pré-entraîné.

Étapes :
    1. charger le checkpoint Monte-Cristo (vocab 94)
    2. créer un modèle chat (vocab 98) et y COPIER les poids pré-entraînés,
       en AGRANDISSANT token_embedding_table et lm_head (94 -> 98 lignes)
    3. construire les batchs (X, Y) avec MASQUE de loss (Y = -100 hors réponse)
    4. fine-tuner avec un petit learning rate
    5. sauver checkpoints/chat_model.pt (+ chat_meta.json) et tester
"""

from __future__ import annotations

import json
import time

import torch

import config
from bpe_tokenizer import BPETokenizer
from chat_data import build_conversations
from chat_infer import chat_generate
from gpt import GPTLanguageModel

PRETRAINED = config.CHECKPOINT_DIR / "model.pt"
CHAT_MODEL = config.CHECKPOINT_DIR / "chat_model.pt"
CHAT_META = config.CHECKPOINT_DIR / "chat_meta.json"


def encode_example(tok: ChatTokenizer, conv: list[dict], block_size: int) -> tuple[list[int], list[int]]:
    """Conversation -> (x, y) de longueur block_size, avec y=-100 sur les positions masquées."""
    ids, trainable = tok.render(conv)
    ids = ids[: block_size + 1]
    trainable = trainable[: block_size + 1]
    pad = (block_size + 1) - len(ids)
    ids += [tok.PAD] * pad
    trainable += [False] * pad

    x = ids[:block_size]
    # y[t] = token à prédire (ids[t+1]) SI supervisé, sinon -100 (ignoré par la loss)
    y = [ids[t + 1] if trainable[t + 1] else -100 for t in range(block_size)]
    return x, y


@torch.no_grad()
def load_pretrained_into(chat_model: torch.nn.Module, path) -> list:
    """Copie les poids pré-entraînés ; agrandit les couches dont la 1re dim a grossi."""
    old = torch.load(path, map_location=config.DEVICE, weights_only=True)
    new = chat_model.state_dict()
    grown = []
    for k, v in new.items():
        if k not in old:
            continue
        ov = old[k]
        if ov.shape == v.shape:
            v.copy_(ov)
        else:
            sl = tuple(slice(0, min(a, b)) for a, b in zip(ov.shape, v.shape))
            v[sl].copy_(ov[sl])  # copie le bloc commun, laisse les nouvelles lignes au hasard
            grown.append(f"{k}: {tuple(ov.shape)} -> {tuple(v.shape)}")
    return grown


def main() -> None:
    torch.manual_seed(config.SEED)

    tok = BPETokenizer()

    # --- Données SFT ---
    convs = build_conversations()
    data = [encode_example(tok, c, config.BLOCK_SIZE) for c in convs]
    X = torch.tensor([d[0] for d in data], dtype=torch.long)
    Y = torch.tensor([d[1] for d in data], dtype=torch.long)
    supervised = int((Y != -100).sum())
    print(f"Exemples SFT       : {len(data)} | X {tuple(X.shape)} | Y {tuple(Y.shape)}")
    print(f"Tokens supervisés  : {supervised:,} / {Y.numel():,} "
          f"({100 * supervised / Y.numel():.1f} % — le reste est masqué)")

    # --- Modèle : on part du modèle pré-entraîné, agrandi de 94 à 98 tokens ---
    if not PRETRAINED.exists():
        raise FileNotFoundError("Checkpoint de base introuvable — lance d'abord train.py.")
    model = GPTLanguageModel(
        vocab_size=tok.vocab_size,  # 98
        n_embd=config.N_EMBD,
        block_size=config.BLOCK_SIZE,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)
    grown = load_pretrained_into(model, PRETRAINED)
    print("Poids pré-entraînés chargés. Couches agrandies :")
    for g in grown:
        print("    " + g)
    print("-" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.FT_LR)

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data), (config.BATCH_SIZE,))
        return X[ix].to(config.DEVICE), Y[ix].to(config.DEVICE)

    history = []
    start = time.time()
    model.train()
    for it in range(config.FT_MAX_ITERS + 1):
        if it % config.FT_EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(50)
                for k in range(50):
                    xb, yb = get_batch()
                    _, l = model(xb, yb)
                    losses[k] = l.item()
            model.train()
            lo = losses.mean().item()
            history.append({"iter": it, "loss": round(lo, 4), "t": round(time.time() - start, 1)})
            print(f"iter {it:5d} | loss {lo:.4f}")

        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- Sauvegarde ---
    torch.save(model.state_dict(), CHAT_MODEL)
    CHAT_META.write_text(
        json.dumps(
            {
                "status": "done",
                "vocab_size": tok.vocab_size,
                "specials": {"pad": tok.PAD, "user": tok.USER, "bot": tok.BOT, "end": tok.END},
                "n_examples": len(data),
                "history": history,
                "elapsed_sec": round(time.time() - start, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 60)
    print(f"Modèle chat sauvegardé -> {CHAT_MODEL}")

    # --- Test : on discute avec l'assistant fraîchement entraîné ---
    print("\n=== Test de l'assistant ===")
    model.eval()
    for prompt in [
        "Bonjour",
        "Comment tu t'appelles ?",
        "Quelle est la capitale de la France ?",
        "Quelle est la capitale de l'Italie ?",
        "Merci beaucoup",
        "Au revoir",
    ]:
        rep = chat_generate(model, tok, prompt, device=config.DEVICE, temperature=0.4)
        print(f"  [user] {prompt}")
        print(f"  [bot]  {rep}\n")


if __name__ == "__main__":
    main()
