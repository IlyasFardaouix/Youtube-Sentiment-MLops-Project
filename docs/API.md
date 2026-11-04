# API

Lancer localement : `uvicorn app_api:app --reload --port 8000`

`GET /health` vérifie le service et le chargement du modèle.

`POST /predict_batch`

```json
{"comments": ["I love this video", "This is disappointing"]}
```

La réponse contient une prédiction par commentaire (`label`, `sentiment`, `confidence`) et les statistiques du batch.
