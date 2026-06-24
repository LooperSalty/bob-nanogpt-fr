# Architecture du modèle

Bob est un **Transformer decoder-only** (comme GPT). Sa seule tâche apprise :
prédire le **caractère suivant**. Tout le reste (génération, dialogue) en découle.

Convention de dimensions : `[B, T, C]` = `[Batch, Time, Channels]`.

## Flux du forward

```
idx  [B, T]                          indices de caractères (entiers)
 ├─ token_embedding   [B, T, C]      sens du caractère
 └─ + position_embed  [T, C]         position dans la séquence
 = x  [B, T, C]
 → N_LAYER × Block    [B, T, C]      attention + feed-forward (résiduels + LayerNorm)
 → LayerNorm finale   [B, T, C]
 → lm_head (Linear)   [B, T, vocab]  logits du prochain caractère
```

## 1. Tokenizer (`tokenizer.py`, `chat_tokenizer.py`)

- `CharTokenizer` : bijection caractère ↔ entier, vocabulaire = caractères uniques
  triés du corpus. `encode` / `decode` sans perte.
- `ChatTokenizer` : étend le vocab de base avec des **tokens spéciaux**
  `<|user|>`, `<|bot|>`, `<|end|>`, `<|pad|>` (+ le caractère `_`), placés *après*
  les IDs de base pour réutiliser les poids pré-entraînés.

## 2. Génération des batchs (`data_loader.py`)

`get_batch` tire `B` fenêtres aléatoires de longueur `T` (`block_size`) :
`X = data[i:i+T]`, `Y = data[i+1:i+1+T]` (décalé de +1). Une fenêtre de `T` tokens
contient `T` exemples imbriqués (prédire t+1 à partir de [0..t]).

## 3. Self-Attention (`attention.py`)

Chaque token projette trois vecteurs : **Query** (ce que je cherche), **Key** (ce
que je contiens), **Value** (ce que je transmets).

```
wei = softmax( (Q · Kᵀ) / √head_size  + masque_causal )   [B, T, T]
out = wei · V                                              [B, T, head_size]
```

- Le `/√head_size` stabilise le softmax (évite la saturation).
- Le **masque causal** (triangulaire) interdit de regarder le futur — indispensable
  pour un modèle autorégressif.
- **Multi-têtes** : plusieurs `Head` en parallèle (chacune sur un sous-espace),
  concaténées puis projetées.

## 4. Feed-Forward (`feed_forward.py`)

MLP par token : `Linear(C, 4C) → ReLU → Linear(4C, C)`. L'attention fait
*communiquer* les tokens (axe T) ; le feed-forward les fait *calculer* (axe C).

## 5. Le Block (`gpt.py`)

```python
x = x + self.sa(self.ln1(x))    # residual + pré-LayerNorm autour de l'attention
x = x + self.ffwd(self.ln2(x))  # residual + pré-LayerNorm autour du feed-forward
```

- **Residual** (`+ x`) : autoroute à gradient, permet d'empiler en profondeur.
- **LayerNorm** : normalise les activations, stabilise l'entraînement.

## 6. Entraînement

- **Pré-entraînement** (`train.py`, `train_long.py`) : prédire le token suivant sur
  tout le corpus. Loss = cross-entropy. Optimizer = AdamW. `bfloat16` sur GPU.
- **Fine-tuning conversationnel / SFT** (`train_chat.py`) :
  - on part du modèle pré-entraîné, on **agrandit** `token_embedding` et `lm_head`
    pour accueillir les tokens spéciaux (ex. 112 → 117 lignes) ;
  - **loss masking** : on ne calcule la loss que sur la réponse de l'assistant
    (les positions de la question sont mises à `-100`, ignorées par `cross_entropy`).

## 7. Génération (`chat_infer.py`)

Autorégressive : on formate `<|user|> … <|end|> <|bot|>`, puis on échantillonne
caractère par caractère (softmax + `multinomial`, avec température et top-k
optionnels) **jusqu'au token `<|end|>`** — c'est ainsi que Bob sait s'arrêter.
