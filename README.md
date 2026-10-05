# Email Triage Agents – Classification d'emails par agents LLM

Un pipeline **multi-agents** (LangGraph + LLM OpenAI ou Mistral) qui trie des emails professionnels :
**intention**, **priorité** et **type d'expéditeur**. Un agent de contrôle qualité peut renvoyer un email
en reclassification ou l'escalader vers un humain. Le pipeline est comparé à une **baseline supervisée**
(TF-IDF + régression logistique), avec un protocole d'évaluation conçu pour éviter la **fuite de données**.

## Architecture

```
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  approuvé   ┌──────────┐
 email ───> │  Classifier  │──> │   Priority   │──> │   Reviewer   │ ──────────> │ Finalize │ ──> résultat
            │ intent+sender│    │ indices      │    │ cohérence    │             │          │
            │ + confiance  │    │ d'urgence    │    │ + correction │             └──────────┘
            └──────────────┘    └──────────────┘    └──────┬───────┘                  ^
                   ^                                       │ rejeté (1er essai)       │ rejeté (2e essai)
                   └──────────── retour du reviewer ───────┘──────────────────────────┘ → needs_human
```

| Agent | Rôle | Sortie (validée par Pydantic) |
|---|---|---|
| `ClassifierAgent` | Intention principale + type d'expéditeur | `intent`, `sender`, `confidence`, `rationale` |
| `PriorityAgent` | Priorité à partir d'indices explicites et implicites | `priority`, `urgency_cues` |
| `ReviewerAgent` | Contrôle qualité, correction éventuelle | `approved`, `corrected_*`, `comment` |

Choix de conception :
- **Sorties structurées** : chaque appel LLM renvoie du JSON validé par un schéma Pydantic, avec
  auto-correction si le JSON est invalide (`email_agents/llm.py`).
- **Boucle de correction bornée** : une seule reclassification, puis escalade humaine (`needs_human`).
  Les emails à faible confiance (< 0,6) sont aussi escaladés.
- **Traçabilité** : chaque résultat contient la trace des décisions des agents, utile pour l'analyse d'erreurs.
- **Fournisseur interchangeable** : OpenAI ou Mistral via `LLM_PROVIDER` (API compatible OpenAI).

## État des tests

 **9 tests unitaires passent avec succès** :
- JSON valide et auto-correction des JSON invalides
- Gestion des clés manquantes et retry jusqu'à l'abandon
- Chemin happy path : email approuvé directement par le reviewer
- Rejet du reviewer → reclassification avec feedback
- Double rejet → escalade vers humain avec correction
- Emails à faible confiance → escalade
- Baseline TF-IDF apprend correctement sur dataset simple

```bash
pytest -q  # 9 passed
```

## Données

`data/generate_dataset.py` génère 400 emails synthétiques en français (graine fixe, reproductible)
avec 6 intentions, 3 priorités et 3 types d'expéditeur. Pour éviter une tâche triviale :
20 % des emails mélangent deux sujets, et une partie des urgences est **implicite**
(« notre clôture comptable a lieu cette semaine »).

## Protocole d'évaluation et fuite de données

Les emails sont générés à partir de gabarits de phrases. Un split aléatoire met des quasi-doublons
dans le train et le test : la baseline semble presque parfaite, mais elle mémorise les gabarits.
Le split par défaut (`grouped`) réserve **un gabarit entier par intention au test**. Toutes les classes
restent présentes, mais avec des formulations jamais vues à l'entraînement.

| Baseline TF-IDF + LogReg | Accuracy intention | Macro-F1 intention | Accuracy priorité |
|---|---|---|---|
| Split aléatoire (fuite) | 0,980 | 0,978 | 0,980 |
| **Split groupé (sans fuite)** | **0,875** | **0,863** | **0,990** |

L'écart montre que la performance en split aléatoire est surestimée.

### Agents LLM

| Méthode (même sous-ensemble de test) | Accuracy intention | Macro-F1 intention | Accuracy priorité | Appels LLM / email | Taux d'escalade |
|---|---|---|---|---|---|
| Baseline TF-IDF | __ | | | 0 | – |
| Agents LLM (`gpt-4o-mini`) | _ | | | | |

> Lancer `python -m email_agents.evaluate --agents --limit 60` puis reporter les valeurs de `reports/metrics.json`.

## Installation et utilisation

```bash
git clone https://github.com/leg234-png/email-triage-agents.git
cd email-triage-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # puis renseigner la clé API (OpenAI ou Mistral)

python data/generate_dataset.py                       # régénère data/emails.csv
python -m email_agents.evaluate                       # baseline seule (sans clé API)
python -m email_agents.evaluate --agents --limit 60   # baseline + agents LLM
python -m email_agents.cli --subject "Double prélèvement" \
    --body "Nous avons été prélevés deux fois, c'est urgent." --sender compta@client.com
pytest -q                                             # tests (faux LLM, sans réseau)
```

Sorties dans `reports/` : `metrics.json`, matrices de confusion, `agent_errors.csv`
(erreurs des agents avec leur trace de décision).

## Structure

```
email_agents/
  llm.py         client OpenAI / Mistral, sorties JSON validées + auto-correction
  schemas.py     schémas Pydantic (entrées, sorties des agents, résultat)
  agents.py      graphe LangGraph : classify → prioritize → review → finalize
  baseline.py    TF-IDF (mots + caractères) + régression logistique
  evaluate.py    protocole d'évaluation, métriques, matrices de confusion, analyse d'erreurs
  cli.py         classer un email en ligne de commande
data/            générateur et jeu de données
tests/           tests unitaires du graphe avec un faux LLM
```

## Limites et pistes

- Données synthétiques : utiles pour un protocole contrôlé, mais à valider sur de vrais emails anonymisés.
- 3 appels LLM par email (plus en cas de rejet) : un mode « un seul appel » réduirait coût et latence.
- Pistes : calibration de la confiance, apprentissage few-shot à partir des erreurs du reviewer,
  routage hybride (baseline si elle est confiante, LLM sinon) pour réduire les coûts.

## Auteur

Emmanuel Wandji – ENSTA, Institut Polytechnique de Paris ·
[LinkedIn](https://www.linkedin.com/in/emmanuel-wandji) · [GitHub](https://github.com/leg234-png)
