"""
Étape 8a + 8b — Démo : tokens spéciaux, rendu d'une conversation, et MASQUE DE LOSS.

Montre concrètement que le modèle n'apprend QUE sur la réponse de l'assistant.
"""

from __future__ import annotations

import config
from chat_data import build_conversations
from chat_tokenizer import ChatTokenizer
from step1_data import load_text
from tokenizer import CharTokenizer


def main() -> None:
    base = CharTokenizer(load_text(config.DATA_PATH))
    tok = ChatTokenizer(base)

    print(f"Vocab de base : {base.vocab_size}")
    print(f"Vocab chat    : {tok.vocab_size}  (+{tok.vocab_size - base.vocab_size} tokens spéciaux)")
    for s, i in tok.special_to_id.items():
        print(f"    {s:<10} -> id {i}")

    convs = build_conversations()
    print(f"\nConversations générées : {len(convs)}")

    # Vérif : aucun caractère hors-vocab (sinon le SFT serait corrompu).
    base_chars = set(base.vocab)
    used = {ch for c in convs for turn in c for ch in turn["text"]}
    oov = sorted(ch for ch in used if ch not in base_chars)
    print(f"Caractères hors-vocab  : {oov if oov else 'aucun ✓'}")

    # Rendu détaillé d'une conversation simple (un seul échange).
    conv = next(c for c in convs if len(c) == 2)
    print("\nConversation exemple :")
    for turn in conv:
        print(f"    [{turn['role']:>4}] {turn['text']}")

    ids, trainable = tok.render(conv)
    print(f"\nRendu (tokens spéciaux visibles) :\n    {tok.decode(ids, keep_specials=True)!r}")
    print(f"Longueur : {len(ids)} tokens")

    print("\n--- Masque de loss, token par token ---")
    print("    idx | token            | supervisé ?")
    for k, (i, t) in enumerate(zip(ids, trainable)):
        disp = tok.id_to_special.get(i) or repr(tok.base.decode([i]))
        print(f"    {k:3d} | {disp:<16} | {'✓ OUI' if t else '·  non'}")

    sup = "".join((tok.id_to_special.get(i) or tok.base.decode([i])) for i, t in zip(ids, trainable) if t)
    print(f"\nLe modèle n'apprend QUE sur : {sup!r}")
    print("(la question de l'utilisateur est masquée -> ignore_index=-100 dans la cross-entropy)")


if __name__ == "__main__":
    main()
