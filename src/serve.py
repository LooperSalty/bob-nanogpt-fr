"""
Interface web — backend FastAPI.

Endpoints :
    GET  /              -> page web (web/index.html)
    GET  /api/health    -> état serveur + device + modèles prêts ?
    GET  /api/stats     -> meta.json (config + historique de loss)   [dashboard]
    POST /api/generate  -> complétion en streaming (modèle de base)  [onglet Complétion]
    POST /api/chat      -> réponse de l'assistant en streaming       [onglet Assistant]

Les modèles sont chargés paresseusement et rechargés si le checkpoint change
sur disque (hot-reload) -> on peut discuter pendant que ça (ré)entraîne.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import config
from bpe_tokenizer import BPETokenizer
from chat_infer import chat_stream
from db import store
from gpt import GPTLanguageModel

WEB = config.WEB_DIR
CKPT = config.CHECKPOINT_DIR / "model.pt"
META = config.CHECKPOINT_DIR / "meta.json"
CHAT_CKPT = config.CHECKPOINT_DIR / "chat_model.pt"

app = FastAPI(title="nanoGPT FR")

# Tokenizer BPE — sert au modèle de base ET à l'assistant (même vocab).
_tokenizer = BPETokenizer()
_chat_tok = _tokenizer

# Caches de modèles : rechargés si le mtime du checkpoint change.
_state: dict = {"model": None, "mtime": 0.0}
_chat_state: dict = {"model": None, "mtime": 0.0}


def _load(cache: dict, ckpt: Path, vocab_size: int) -> GPTLanguageModel | None:
    if not ckpt.exists():
        return None
    mtime = ckpt.stat().st_mtime
    if cache["model"] is None or mtime != cache["mtime"]:
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            n_embd=config.N_EMBD,
            block_size=config.BLOCK_SIZE,
            n_head=config.N_HEAD,
            n_layer=config.N_LAYER,
            dropout=config.DROPOUT,
        ).to(config.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE, weights_only=True))
        model.eval()
        cache["model"] = model
        cache["mtime"] = mtime
    return cache["model"]


def get_model() -> GPTLanguageModel | None:
    return _load(_state, CKPT, _tokenizer.vocab_size)


def get_chat_model() -> GPTLanguageModel | None:
    return _load(_chat_state, CHAT_CKPT, _chat_tok.vocab_size)


class GenRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 400
    temperature: float = 0.8
    top_k: int | None = None


class FilePayload(BaseModel):
    name: str = ""
    size: int = 0
    content: str = ""


class ChatRequest(BaseModel):
    conversation_id: str
    text: str = ""
    file: FilePayload | None = None
    max_new_tokens: int = 220
    temperature: float = 0.5
    top_k: int | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "device": config.DEVICE,
        "model_ready": CKPT.exists(),
        "chat_ready": CHAT_CKPT.exists(),
        "db": store.backend,
        "torch": torch.__version__,
    }


@app.get("/api/stats")
def stats() -> JSONResponse:
    try:
        return JSONResponse(json.loads(META.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"status": "no_run", "history": []})


# --- Complétion (modèle de base) ---
def stream_generate(req: GenRequest):
    model = get_model()
    if model is None:
        yield "[Le modèle de base n'est pas encore prêt.]"
        return
    ids = _tokenizer.encode(req.prompt) or [_tokenizer.PAD]
    idx = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)
    yield req.prompt
    temperature = max(req.temperature, 1e-5)
    gen, prev = list(ids), _tokenizer.decode(ids)
    for _ in range(max(1, min(req.max_new_tokens, 2000))):
        with torch.no_grad():
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if req.top_k:
                v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            gen.append(int(idx_next.item()))
            text = _tokenizer.decode(gen)
        if len(text) > len(prev):
            yield text[len(prev):]
            prev = text


@app.post("/api/generate")
def generate(req: GenRequest) -> StreamingResponse:
    return StreamingResponse(stream_generate(req), media_type="text/plain; charset=utf-8")


# --- Assistant (modèle fine-tuné), avec persistance des conversations ---
def _fold(m: dict) -> str:
    """Replie une éventuelle pièce jointe dans le texte envoyé au modèle."""
    f = m.get("file")
    if f and f.get("content"):
        return f"{m.get('text', '')}\n\n[Fichier joint : {f.get('name', '')}]\n{f['content']}"
    return m.get("text", "")


def stream_chat(req: ChatRequest):
    model = get_chat_model()
    if model is None:
        yield "[L'assistant n'est pas encore entraîné — lance finetune (train_chat.py).]"
        return

    file = req.file.model_dump() if req.file else None
    store.append(req.conversation_id, "user", req.text, file)

    conv = store.get(req.conversation_id)
    history = conv["messages"] if conv else [{"role": "user", "text": req.text}]
    model_msgs = [{"role": m["role"], "text": _fold(m)} for m in history]

    chunks: list[str] = []
    for ch in chat_stream(model, _chat_tok, model_msgs, device=config.DEVICE,
                          max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_k=req.top_k):
        chunks.append(ch)
        yield ch
    store.append(req.conversation_id, "bot", "".join(chunks))


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(stream_chat(req), media_type="text/plain; charset=utf-8")


# --- Conversations (CRUD, stockées en base NoSQL) ---
class NewConversation(BaseModel):
    title: str = "Nouvelle discussion"


@app.post("/api/conversations")
def create_conversation(body: NewConversation | None = None) -> dict:
    return store.create(body.title if body else "Nouvelle discussion")


@app.get("/api/conversations")
def list_conversations() -> list:
    return store.list()


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str) -> JSONResponse:
    conv = store.get(cid)
    if conv is None:
        return JSONResponse({"error": "introuvable"}, status_code=404)
    return JSONResponse(conv)


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str) -> dict:
    store.delete(cid)
    return {"ok": True}
