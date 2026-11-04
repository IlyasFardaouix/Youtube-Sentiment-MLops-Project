"""FastAPI service for batch sentiment prediction."""
from pathlib import Path
from time import perf_counter

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

app = FastAPI(title="Reddit Sentiment API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictionRequest(BaseModel):
    comments: list[str] = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict_batch")
def predict_batch(request: PredictionRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained")
    if any(not comment.strip() for comment in request.comments):
        raise HTTPException(status_code=422, detail="Comments must not be empty")
    started = perf_counter()
    predictions = model.predict(request.comments).tolist()
    probabilities = model.predict_proba(request.comments) if hasattr(model, "predict_proba") else None
    results = []
    for index, (comment, label) in enumerate(zip(request.comments, predictions)):
        confidence = float(max(probabilities[index])) if probabilities is not None else None
        results.append({"text": comment, "label": int(label), "sentiment": {"-1": "negative", "0": "neutral", "1": "positive"}[str(label)], "confidence": confidence})
    counts = {str(label): predictions.count(label) for label in [-1, 0, 1]}
    return {"predictions": results, "statistics": {"total": len(results), "counts": counts, "latency_ms": (perf_counter() - started) * 1000}}
