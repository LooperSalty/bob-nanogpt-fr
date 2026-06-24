"""
Étape 8b — Dataset de fine-tuning conversationnel (synthétique, en français).

Paires (question -> réponse) à partir de patrons. Inclut des "connaissances de
base" (capitales, faits simples, maths) pour que Bob réponde à des questions
factuelles après le SFT. Le français fluide vient du pré-entraînement Wikipédia ;
ces réponses-ci sont les comportements précis que le SFT fixe.
"""

from __future__ import annotations

import random

# Faits : (question(s), réponse modèle) — étoffe les "connaissances de base".
_CAPITALES = {
    "la France": "Paris", "l'Espagne": "Madrid", "l'Italie": "Rome",
    "l'Allemagne": "Berlin", "le Royaume-Uni": "Londres", "le Portugal": "Lisbonne",
    "la Belgique": "Bruxelles", "la Suisse": "Berne", "le Japon": "Tokyo",
    "les États-Unis": "Washington", "le Canada": "Ottawa", "la Russie": "Moscou",
    "la Chine": "Pékin", "la Grèce": "Athènes", "le Maroc": "Rabat",
    "le Brésil": "Brasilia", "les Pays-Bas": "Amsterdam", "l'Autriche": "Vienne",
}

INTENTS = [
    {"user": ["Bonjour", "Salut", "Coucou", "Bonsoir"],
     "bot": ["Bonjour ! Comment puis-je t'aider ?", "Salut ! Que puis-je faire pour toi ?",
             "Bonjour ! Ravi de te parler."]},
    {"user": ["Comment ça va ?", "Ça va ?", "Comment vas-tu ?"],
     "bot": ["Je vais très bien, merci ! Et toi ?", "Tout va bien, merci de demander."]},
    {"user": ["Comment tu t'appelles ?", "Quel est ton nom ?", "Tu t'appelles comment ?"],
     "bot": ["Je m'appelle Bob, un assistant créé par Looper_Salty.", "Mon nom est Bob."]},
    {"user": ["Qui es-tu ?", "Présente-toi.", "Tu es quoi ?"],
     "bot": ["Je suis Bob, un modèle de langage entraîné par Looper_Salty sur du texte français.",
             "Je suis Bob, un assistant qui apprend le français."]},
    {"user": ["Qui t'a créé ?", "Qui est ton créateur ?", "Qui t'a fait ?", "Qui t'a programmé ?"],
     "bot": ["J'ai été créé par Looper_Salty.", "C'est Looper_Salty qui m'a créé."]},
    {"user": ["Écris-moi une poésie", "Écris une poésie", "Fais-moi un poème",
              "Récite un poème", "Un poème s'il te plaît"],
     "bot": ["Le soleil brille sur la mer,\net le vent murmure tout bas ;\nmon coeur, léger comme l'air,\ns'envole et chante avec toi.",
             "La nuit étend son voile noir,\nles étoiles dansent là-haut ;\ndans le silence et dans le soir,\nje rêve tout bas au bord de l'eau."]},
    {"user": ["Merci", "Merci beaucoup", "Je te remercie"],
     "bot": ["De rien !", "Avec plaisir !", "Je t'en prie."]},
    {"user": ["Au revoir", "Bonne nuit", "À bientôt"],
     "bot": ["Au revoir !", "Bonne journée !", "À bientôt !"]},
    {"user": ["Aide-moi", "Peux-tu m'aider ?", "J'ai besoin d'aide"],
     "bot": ["Bien sûr, dis-moi ce dont tu as besoin.", "Avec plaisir, explique-moi ton problème."]},
    {"user": ["Quelle langue parles-tu ?", "Tu parles français ?"],
     "bot": ["Je parle français.", "Oui, je parle français !"]},
    # --- Connaissances de base ---
    {"user": ["Combien font deux plus deux ?", "Deux plus deux ?"],
     "bot": ["Deux plus deux font quatre.", "Cela fait quatre."]},
    {"user": ["Combien font trois fois quatre ?", "Trois fois quatre ?"],
     "bot": ["Trois fois quatre font douze.", "Cela fait douze."]},
    {"user": ["De quoi est composée l'eau ?", "Qu'est-ce que l'eau ?"],
     "bot": ["L'eau est composée d'hydrogène et d'oxygène.",
             "L'eau est faite de molécules d'hydrogène et d'oxygène."]},
    {"user": ["Combien y a-t-il de planètes dans le système solaire ?", "Combien de planètes ?"],
     "bot": ["Il y a huit planètes dans le système solaire.", "Le système solaire compte huit planètes."]},
    {"user": ["Quelle est la monnaie de la France ?", "Monnaie de la France ?"],
     "bot": ["La monnaie de la France est l'euro.", "C'est l'euro."]},
    {"user": ["Quelle est la plus grande planète ?", "La plus grosse planète ?"],
     "bot": ["La plus grande planète est Jupiter.", "C'est Jupiter."]},
    {"user": ["Qu'est-ce qu'un ordinateur ?"],
     "bot": ["Un ordinateur est une machine qui traite des informations.",
             "C'est une machine électronique qui exécute des programmes."]},
    {"user": ["Sur quelle planète vivons-nous ?", "Comment s'appelle notre planète ?"],
     "bot": ["Nous vivons sur la Terre.", "Notre planète s'appelle la Terre."]},
]

# Génère une intention par capitale.
for _pays, _cap in _CAPITALES.items():
    INTENTS.append({
        "user": [f"Quelle est la capitale de {_pays} ?", f"Capitale de {_pays} ?"],
        "bot": [f"La capitale de {_pays} est {_cap}.", f"C'est {_cap}."],
    })


def build_conversations(seed: int = 1337) -> list[list[dict]]:
    """Renvoie une liste de conversations (chacune = liste de tours user/bot)."""
    rng = random.Random(seed)

    singles: list[list[dict]] = []
    for intent in INTENTS:
        for u in intent["user"]:
            for b in intent["bot"]:
                singles.append([{"role": "user", "text": u}, {"role": "bot", "text": b}])

    multi = [rng.choice(singles) + rng.choice(singles) for _ in range(len(singles))]
    convs = singles + multi
    rng.shuffle(convs)
    return convs
