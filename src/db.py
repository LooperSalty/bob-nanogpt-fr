"""
Stockage des conversations du chat — base NoSQL MongoDB.

Repli automatique sur un fichier JSON local si MongoDB est indisponible
(le site continue de fonctionner même sans Docker/Mongo).

Schéma d'un document :
    { id, title, messages: [{role, text, file?}], created, updated }
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import suppress

import config

MONGO_URI = "mongodb://localhost:27017/"
_FALLBACK = config.CHECKPOINT_DIR / "chats_fallback.json"


def _now() -> float:
    return time.time()


class ChatStore:
    """Persistance des conversations : MongoDB en priorité, sinon fichier JSON."""

    def __init__(self) -> None:
        self.coll = None
        self.backend = "fichier JSON"
        self._mem: dict = {}
        try:
            from pymongo import MongoClient

            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            self.coll = client["bob"]["conversations"]
            self.coll.create_index("updated")
            self.backend = "MongoDB"
        except Exception:
            self._mem = self._load()

    # ---- repli fichier ----
    def _load(self) -> dict:
        if _FALLBACK.exists():
            with suppress(Exception):
                return json.loads(_FALLBACK.read_text(encoding="utf-8"))
        return {}

    def _flush(self) -> None:
        config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        _FALLBACK.write_text(json.dumps(self._mem, ensure_ascii=False), encoding="utf-8")

    # ---- API ----
    def create(self, title: str = "Nouvelle discussion") -> dict:
        doc = {"id": uuid.uuid4().hex, "title": title, "messages": [],
               "created": _now(), "updated": _now()}
        if self.coll is not None:
            self.coll.insert_one(dict(doc))
        else:
            self._mem[doc["id"]] = doc
            self._flush()
        return {"id": doc["id"], "title": doc["title"], "updated": doc["updated"]}

    def list(self) -> list[dict]:
        if self.coll is not None:
            cur = self.coll.find({}, {"_id": 0, "id": 1, "title": 1, "updated": 1}).sort("updated", -1)
            return list(cur)
        items = [{"id": c["id"], "title": c["title"], "updated": c["updated"]} for c in self._mem.values()]
        return sorted(items, key=lambda c: c["updated"], reverse=True)

    def get(self, cid: str) -> dict | None:
        if self.coll is not None:
            return self.coll.find_one({"id": cid}, {"_id": 0})
        return self._mem.get(cid)

    def append(self, cid: str, role: str, text: str, file: dict | None = None) -> None:
        conv = self.get(cid)
        if conv is None:
            return
        msg = {"role": role, "text": text}
        if file:
            msg["file"] = file
        set_fields = {"updated": _now()}
        first_user = role == "user" and not any(m["role"] == "user" for m in conv.get("messages", []))
        if first_user and conv.get("title", "Nouvelle discussion") == "Nouvelle discussion":
            set_fields["title"] = text[:42] or "Nouvelle discussion"
        if self.coll is not None:
            self.coll.update_one({"id": cid}, {"$push": {"messages": msg}, "$set": set_fields})
        else:
            conv["messages"].append(msg)
            conv.update(set_fields)
            self._flush()

    def delete(self, cid: str) -> None:
        if self.coll is not None:
            self.coll.delete_one({"id": cid})
        else:
            self._mem.pop(cid, None)
            self._flush()


store = ChatStore()
