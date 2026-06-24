"""
Étape 8c/8d — Génération conversationnelle.

On formate le prompt selon le gabarit de chat, puis on génère jusqu'à ce que le
modèle émette <|end|> (le token d'arrêt appris pendant le SFT).

    entrée envoyée au modèle : <|user|> {prompt} <|end|> <|bot|>
    le modèle complète         :                               {réponse} <|end|>
                                                                          ^^^^^^ STOP

- chat_generate : renvoie la réponse complète (utilisé en CLI / tests)
- chat_stream   : générateur qui renvoie la réponse caractère par caractère (web)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _prompt_ids(tok, messages: list[dict]) -> list[int]:
    """Sérialise un historique de conversation en IDs, prêt pour la génération."""
    ids: list[int] = []
    for m in messages:
        ids.append(tok.BOT if m.get("role") == "bot" else tok.USER)
        ids.extend(tok.encode_text(m.get("text", "")))
        ids.append(tok.END)
    ids.append(tok.BOT)  # on amorce le tour de l'assistant
    return ids


def _sample_next(model, idx, temperature: float, top_k: int | None) -> int:
    idx_cond = idx[:, -model.block_size:]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def chat_generate(
    model, tok, prompt: str, *, device: str = "cpu",
    max_new_tokens: int = 200, temperature: float = 0.8, top_k: int | None = None,
) -> str:
    """Réponse complète à un prompt simple (un seul tour utilisateur)."""
    ids = _prompt_ids(tok, [{"role": "user", "text": prompt}])
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    temperature = max(temperature, 1e-5)
    out: list[int] = []
    for _ in range(max_new_tokens):
        nid = _sample_next(model, idx, temperature, top_k)
        if nid == tok.END:
            break
        out.append(nid)
        idx = torch.cat((idx, torch.tensor([[nid]], device=device)), dim=1)
    return tok.decode(out)


def chat_stream(
    model, tok, messages: list[dict], *, device: str = "cpu",
    max_new_tokens: int = 200, temperature: float = 0.8, top_k: int | None = None,
):
    """Générateur : renvoie la réponse de l'assistant caractère par caractère.

    `messages` = historique [{"role": "user"|"bot", "text": ...}, ...].
    S'arrête dès que le modèle émet <|end|>.
    """
    ids = _prompt_ids(tok, messages)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    temperature = max(temperature, 1e-5)
    gen: list[int] = []
    prev = ""
    for _ in range(max(1, min(max_new_tokens, 1000))):
        with torch.no_grad():
            nid = _sample_next(model, idx, temperature, top_k)
        if nid == tok.END:
            return
        gen.append(nid)
        idx = torch.cat((idx, torch.tensor([[nid]], device=device)), dim=1)
        # Décodage incrémental : on redécode toute la réponse et on émet le
        # nouveau morceau (robuste au BPE byte-level, où 1 caractère peut
        # s'étaler sur plusieurs tokens).
        text = tok.decode(gen)
        if len(text) > len(prev):
            yield text[len(prev):]
            prev = text
