# 🤖 Bob — un mini-GPT français, écrit *from scratch*

> Un modèle de langage de type **GPT**, construit brique par brique en **PyTorch**,
> entraîné sur du français, puis affiné pour discuter — avec une interface web
> (chat, complétion, dashboard d'entraînement temps réel).

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA)-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Bob est un **projet pédagogique** : chaque composant du Transformer a été
implémenté et compris un par un. Rien n'est repris d'un modèle pré-entraîné —
**tout est entraîné à la maison**, sur une seule carte graphique (RTX 4070 SUPER).

---

## ✨ Fonctionnalités

- **Architecture GPT complète, codée à la main** : embeddings de tokens + positions,
  self-attention causale multi-têtes, feed-forward, residual connections, LayerNorm.
- **Tokenizer BPE** (sous-mots, vocab 8000) entraîné sur le corpus — après une
  première version au niveau caractère.
- **Pré-entraînement** sur ~52 Mo de Wikipédia FR (bfloat16, ~20 M paramètres),
  avec checkpoints, **reprise (resume)** et dashboard en direct.
- **Assistant conversationnel** par fine-tuning supervisé (SFT) : tokens spéciaux
  de chat + *loss masking* (on n'entraîne que la réponse de l'assistant).
- **Interface web** (FastAPI + JS vanilla, thème « encre & laiton ») à 3 onglets :
  - 💬 **Assistant** — chat à bulles, historique persistant en **MongoDB**, streaming, pièces jointes.
  - ✍️ **Complétion** — le modèle de base prolonge un texte.
  - 📊 **Dashboard** — courbes de loss et statistiques d'entraînement en temps réel.

## 🧠 Le parcours, de zéro à l'assistant

1. Dataset + tokenizer caractère
2. Génération des batchs `(X, Y)` `[B, T]`
3. Bigramme + boucle d'entraînement (loss, optimizer, backprop)
4. Self-attention (Query/Key/Value) + masque causal
5. Multi-Head Attention + Feed-Forward
6. Residual connections + LayerNorm (blocs empilables)
7. Scaling + interface web d'inférence
8. Assistant conversationnel (SFT : tokens spéciaux, loss masking)
9. **Scaling BPE** : passage aux sous-mots + modèle plus gros + corpus plus large

Détails techniques (shapes, attention, SFT) : dossier [`doc/`](doc/).

## 🚀 Démarrage rapide

> Prérequis : Python 3.12, un GPU NVIDIA (conseillé), Docker (pour la base des chats).

```bash
# 1. PyTorch GPU (CUDA 12.8) — NE PAS faire un simple `pip install torch` (= build CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Base NoSQL des conversations
docker run -d --name bob-mongo --restart unless-stopped -p 27017:27017 mongo:7

# 3. Construire le corpus français (Wikipédia FR, ~50 Mo)
python src/build_corpus_fr.py 50

# 4. Entraîner (pré-entraînement chronométré ~2h + fine-tuning auto)
python src/train_long.py

# 5. Lancer le site
python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000
#  -> http://127.0.0.1:8000
```

> Les dossiers `data/` (corpus) et `checkpoints/` (modèles) ne sont **pas** versionnés
> car volumineux et régénérables par les commandes ci-dessus.
> Aide-mémoire complet : [`doc/commands.md`](doc/commands.md).

## 🗂️ Structure

```
src/        tout le code Python (tokenizers, modèle, entraînement, serveur, BDD)
web/        interface web (index.html)
doc/        documentation (README, architecture, commands)
data/        corpus d'entraînement        (non versionné)
checkpoints/ modèles + tokenizer BPE      (non versionné)
```

## 📊 Le modèle

- **~20 M paramètres** (n_embd 384, 8 couches, 6 têtes, contexte 256, vocab BPE 8000).
- Pré-entraîné en **bfloat16** sur ~52 Mo de Wikipédia FR.
- Affiné (SFT) sur des dialogues français + connaissances de base (capitales, faits simples, poésie).

## ⚠️ Honnêteté

Bob est un **petit modèle entraîné à la maison**. Il parle un français crédible et
répond aux intentions apprises, mais ce n'est **pas** un assistant généraliste : il
ne rédige pas de dissertations sur commande et ne « raisonne » pas. L'objectif est
de **comprendre comment fonctionne un GPT**, pas de rivaliser avec ChatGPT.

## 🛠️ Stack

PyTorch · FastAPI + Uvicorn · HuggingFace `tokenizers` & `datasets` · MongoDB · Chart.js

## 📄 Licence

[MIT](LICENSE). Le corpus d'entraînement provient de Wikipédia (CC BY-SA) et n'est pas
inclus dans ce dépôt.

---

Construit par **Looper_Salty** comme projet d'apprentissage du deep learning. 🇫🇷
