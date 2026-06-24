"""
Hyperparamètres centralisés du projet nanoGPT.

Version "entraînement long" (>= 2h) sur gros corpus français.
Adapté à une RTX 4070 SUPER (12 Go) en bfloat16.
"""

from pathlib import Path

import torch

# Racine du projet : ce fichier est dans src/, mais data/, checkpoints/ et web/
# vivent à la racine. On référence donc tout par rapport à PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Reproductibilité ---
SEED = 1337

# --- Données ---
DATA_PATH = PROJECT_ROOT / "data" / "input.txt"
TRAIN_FRAC = 0.9

# --- Échantillonnage des batchs ---
BLOCK_SIZE = 256  # T : fenêtre de contexte
BATCH_SIZE = 64   # B : séquences par batch

# --- Modèle (config "GPT" de Karpathy : ~10,7 M params) ---
N_EMBD = 384   # C : dimension des embeddings
N_HEAD = 6     # têtes (head_size = 384 // 6 = 64)
N_LAYER = 8    # blocs Transformer empilés (scaling BPE)
DROPOUT = 0.2

# --- Entraînement (pré-entraînement) ---
LEARNING_RATE = 3e-4
TRAIN_SECONDS = 7200   # durée cible du pré-entraînement (2 h)
MAX_ITERS = 100000     # plafond de sécurité (le temps prime, cf. train_long.py)
EVAL_INTERVAL = 1000   # éval + sauvegarde checkpoint à cette fréquence
EVAL_ITERS = 50        # batchs moyennés par estimation de loss

# --- Fine-tuning conversationnel (SFT) ---
FT_LR = 3e-4
FT_MAX_ITERS = 2000
FT_EVAL_INTERVAL = 250

# --- Checkpoints / logs (lus par le serveur web) ---
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WEB_DIR = PROJECT_ROOT / "web"

# --- Matériel ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
