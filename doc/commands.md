# Aide-mémoire des commandes

> Tout le code est dans `src/`. Les scripts se lancent avec `python src/<script>.py`
> (le dossier `src/` est automatiquement sur le `sys.path`). Les chemins data/
> checkpoints/ web/ sont résolus par rapport à la racine du projet (`config.PROJECT_ROOT`).

## Installation

```bash
# PyTorch GPU (CUDA 12.8) — NE PAS faire un simple `pip install torch` (= CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Base de données (MongoDB via Docker)

```bash
docker run -d --name bob-mongo --restart unless-stopped -p 27017:27017 mongo:7
docker ps --filter name=bob-mongo      # vérifier qu'elle tourne
docker stop bob-mongo                   # arrêter
docker start bob-mongo                  # relancer
```

## Données

```bash
python src/build_corpus_fr.py 20       # corpus Wikipédia FR (~20 Mo)
python src/download_data.py            # alternative : un livre Gutenberg
python src/step1_data.py               # démo tokenizer
```

## Entraînement

```bash
python src/train.py                    # pré-entraînement court (config MAX_ITERS)
python src/train_long.py               # pré-entraînement ~2h + fine-tuning auto
python src/train_chat.py               # fine-tuning conversationnel seul (SFT)
```

Pour un entraînement long sans surveillance (détaché, survit à la fermeture du terminal) :

```powershell
Start-Process python -ArgumentList "src/train_long.py" `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput train_long.log -RedirectStandardError train_long.err.log
```

Arrêter : `Stop-Process -Id <PID>`.

## Serveur web

```bash
python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000
```

## Démos pédagogiques

```bash
python src/step4_attention.py          # self-attention (Q/K/V + masque)
python src/step8_chat_data.py          # tokens spéciaux + masque de loss (SFT)
```

## API (résumé)

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/health` | état (device, modèles, base de données) |
| GET | `/api/stats` | métriques d'entraînement (dashboard) |
| POST | `/api/generate` | complétion en streaming (modèle de base) |
| POST | `/api/chat` | réponse de l'assistant + persistance |
| POST/GET | `/api/conversations` | créer / lister les conversations |
| GET/DELETE | `/api/conversations/{id}` | lire / supprimer une conversation |
