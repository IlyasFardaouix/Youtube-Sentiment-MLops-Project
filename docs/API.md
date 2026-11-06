# API

## Lancement

Localement : `uvicorn app_api:app --reload --port 8000`

Dans GitHub Codespaces, lancer l'image avec `docker run -d -p 7860:7860 -e PORT=7860 youtube-sentiment-api`, puis rendre le port `7860` public. L'URL publique prend la forme `https://<nom-du-codespace>-7860.app.github.dev`.

## `GET /health`

Vérifie le service et le chargement du modèle.

```json
{"status": "ok", "model_loaded": true}
```

## `POST /predict_batch`

Analyse un batch de 1 à 500 commentaires.

Requête :

```json
{"comments": ["I love this video", "This is disappointing"]}
```

Réponse :

```json
{
  "predictions": [
    {"text": "I love this video", "label": 1, "sentiment": "positive", "confidence": 0.87},
    {"text": "This is disappointing", "label": -1, "sentiment": "negative", "confidence": 0.79}
  ],
  "statistics": {
    "total": 2,
    "counts": {"-1": 1, "0": 0, "1": 1},
    "latency_ms": 4.1
  }
}
```

## Erreurs

| Cas | Code |
|---|---|
| Batch vide ou > 500, type invalide, commentaire vide | `422` |
| Modèle non chargé | `503` |
