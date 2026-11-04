# API

Lancer localement : `uvicorn app_api:app --reload --port 8000`

Dans GitHub Codespaces, lancer l'image avec `docker run -d -p 7860:7860 -e PORT=7860 youtube-sentiment-api`, puis rendre le port `7860` public. L'URL publique prend la forme `https://<nom-du-codespace>-7860.app.github.dev`.

`GET /health` vérifie le service et le chargement du modèle.

`POST /predict_batch`

```json
{"comments": ["I love this video", "This is disappointing"]}
```

La réponse contient une prédiction par commentaire (`label`, `sentiment`, `confidence`) et les statistiques du batch.
