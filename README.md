---
title: Youtube Sentiment API
sdk: docker
app_port: 7860
---

# YouTube Sentiment Lens

Projet MLOps de classification de sentiment des commentaires Reddit/YouTube avec TF-IDF, Logistic Regression optimisée, API FastAPI et extension Chrome Manifest V3.

## Architecture

`data/raw` conserve la source; `data/processed` contient les splits et l'EDA; `src/data` prépare les données; `src/models` entraîne et évalue; `src/api` sert le pipeline; `chrome-extension` fournit l'interface; `Dockerfile` cible Hugging Face Spaces sur le port 7860.

## Installation et exécution

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.models.train
uvicorn app_api:app --reload --port 8000
```

Le téléchargement est automatique et écrit `data/raw/reddit.csv`. Le modèle et les métriques sont écrits dans `models/`. Pour l'extension : ouvrir `chrome://extensions`, activer le mode développeur, puis charger le dossier `chrome-extension`.

## Validation

```powershell
pytest -q
```

Les métriques complètes sont dans `models/metrics.json`; le seuil demandé est accuracy >= 0.80, F1 par classe > 0.75 et inférence d'un batch de 50 < 100 ms.

## Déploiement Hugging Face

Créer un Space Docker, pousser le contenu du dépôt avec `Dockerfile`, `requirements-prod.txt`, `app_api.py`, `src/` et `models/`, puis remplacer l'URL locale dans `chrome-extension/popup.html` par l'URL `https://<espace>.hf.space`. La création et le push nécessitent les identifiants Hugging Face de l'étudiant.

## Questions de réflexion

- Séparer raw/processed protège la source originale et rend les transformations reproductibles.
- Les secrets, environnements virtuels, caches, logs, données générées et modèles lourds ne doivent pas être commités.
- TF-IDF réduit le poids des mots fréquents et valorise les termes discriminants; la matrice de confusion montre les confusions entre classes.
- Les emojis sont conservés autant que possible par le nettoyage Unicode; les URLs et mentions sont remplacées par des tokens pour garder leur signal sans leur bruit.
- Un split stratifié 80/20 conserve la distribution des classes.
- FastAPI apporte validation Pydantic, documentation OpenAPI et bonnes performances; CORS autorise l'extension à appeler l'API.
- Les commentaires YouTube sont dynamiques : le content script lit ceux présents, et l'utilisateur peut relancer l'analyse après chargement de nouveaux commentaires.
- Les requirements dev incluent tests/EDA; la production ne conserve que les dépendances d'exécution, ce qui réduit l'image Docker.
