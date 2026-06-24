# Bob — un mini-GPT français, fait main

Bob est un **modèle de langage de type GPT, écrit from scratch en PyTorch**, entraîné
au niveau caractère sur du texte français, puis affiné (fine-tuning) pour discuter.
Le projet est avant tout **pédagogique** : chaque brique du Transformer a été
construite et comprise une par une (tokenizer → attention → blocs → entraînement →
SFT conversationnel), avec une interface web pour discuter et suivre l'entraînement.

> Créé par **Looper_Salty**.

---

## Démarrage rapide

Prérequis : Python 3.12, un GPU NVIDIA (optionnel mais conseillé), Docker (pour la base de données des chats).

```bash
# 1. Dépendances Python
pip install torch --index-url https://download.pytorch.org/whl/cu128   # build GPU (CUDA 12.8)
pip install -r requirements.txt                                        # fastapi, uvicorn, datasets, pymongo...

# 2. Base de données des conversations (NoSQL)
docker run -d --name bob-mongo --restart unless-stopped -p 27017:27017 mongo:7

# 3. (optionnel) Reconstruire le corpus français depuis Wikipédia
python src/build_corpus_fr.py 20        # ~20 Mo de français propre

# 4. Entraîner Bob (pré-entraînement long ~2h + fine-tuning auto)
python src/train_long.py

# 5. Lancer le site
python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000
# -> http://127.0.0.1:8000
```

La liste complète des commandes est dans [commands.md](commands.md).

---

## L'interface web

Trois onglets (http://127.0.0.1:8000) :

- **💬 Assistant** — chat à bulles avec Bob. Historique des conversations dans la
  barre latérale (stocké en **MongoDB**), nouvelle conversation à chaque visite,
  pièces jointes (📎), streaming caractère par caractère.
- **✍️ Complétion** — le modèle de base prolonge une amorce de texte (style brut).
- **📊 Dashboard** — statistiques d'entraînement en direct (paramètres, loss
  train/val, courbes), rafraîchies toutes les 4 s.

---

## Architecture du projet

Tout le code Python vit dans `src/` ; les données, modèles et la page web restent à la racine.

```
bob/
├── data/input.txt            # corpus d'entraînement (français)
├── checkpoints/              # model.pt, chat_model.pt, meta.json (écrits par l'entraînement)
├── web/index.html            # frontend (servi par FastAPI)
├── doc/                      # cette documentation
├── requirements.txt
└── src/
    ├── config.py             # hyperparamètres + chemins (PROJECT_ROOT)
    ├── tokenizer.py          # CharTokenizer (caractère -> entier)
    ├── chat_tokenizer.py     # ChatTokenizer (tokens spéciaux <|user|>… + '_')
    ├── step1_data.py         # load_text + démo Étape 1
    ├── data_loader.py        # CharDataset : génération des batchs (X, Y)
    ├── chat_data.py          # dataset conversationnel synthétique (SFT)
    ├── attention.py          # Head + MultiHeadAttention (self-attention causale)
    ├── feed_forward.py       # FeedForward (MLP par token)
    ├── gpt.py                # Block (residual + LayerNorm) + GPTLanguageModel
    ├── model_bigram.py       # modèle bigramme (référence Étape 3)
    ├── train.py              # boucle d'entraînement + checkpoints
    ├── train_chat.py         # fine-tuning conversationnel (SFT, loss masking)
    ├── train_long.py         # pré-entraînement chronométré (~2h) + SFT auto
    ├── chat_infer.py         # génération chat (arrêt sur <|end|>)
    ├── serve.py              # backend FastAPI (API + streaming)
    ├── db.py                 # stockage des chats (MongoDB, repli JSON)
    ├── build_corpus_fr.py    # construit le corpus depuis Wikipédia FR
    ├── download_data.py      # télécharge un livre Gutenberg (alternative)
    ├── step4_attention.py    # démo Étape 4 (self-attention)
    └── step8_chat_data.py    # démo Étape 8 (tokens spéciaux + masque de loss)
```

Détail des mécanismes du modèle : [architecture.md](architecture.md).

---

## La base de données des conversations

Les chats sont stockés dans **MongoDB** (NoSQL), conteneur Docker `bob-mongo`.
Si Mongo est indisponible, `db.py` bascule automatiquement sur un fichier JSON local
(`checkpoints/chats_fallback.json`) — le site reste fonctionnel.

Schéma d'un document : `{ id, title, messages: [{role, text, file?}], created, updated }`.

Routes API : `POST/GET /api/conversations`, `GET/DELETE /api/conversations/{id}`,
`POST /api/chat` (persiste le tour user + la réponse de Bob).

---

## Le parcours pédagogique (8 étapes)

1. **Dataset + Tokenizer** caractère par caractère.
2. **Dataloaders** : batchs `(X, Y)` `[B, T]`, Y décalé de +1.
3. **Bigramme** + boucle d'entraînement (loss, optimizer, backprop).
4. **Self-Attention** (Query/Key/Value) + masque causal.
5. **Multi-Head Attention** + Feed-Forward.
6. **Residual connections** + **LayerNorm** (blocs empilables).
7. **Scaling up** + interface web d'inférence.
8. **Assistant conversationnel** : tokens spéciaux, loss masking, fine-tuning (SFT).

---

## Le modèle actuel

- **~10,8 M paramètres** (n_embd 384, 6 couches, 6 têtes, contexte 256).
- Pré-entraîné ~2 h sur **~21 Mo de Wikipédia FR** (val loss ≈ 1,04).
- Fine-tuné (SFT) sur des dialogues français + connaissances de base (capitales, faits simples, poésie).
- Entraîné en **bfloat16** sur RTX 4070 SUPER.

> ⚠️ Bob reste un petit modèle char-level : il parle un français crédible et répond
> aux intentions apprises, mais ce n'est pas une encyclopédie. Pour un vrai assistant,
> voir les pistes « pour aller plus loin » ci-dessous.

## Pour aller plus loin

- **Tokenizer BPE** (sub-words) au lieu du caractère.
- **Modèle plus gros** + corpus plus large.
- **Alignement par préférences** (DPO) après le SFT.
