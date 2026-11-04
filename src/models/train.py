"""Train TF-IDF classifiers, tune the best candidate, and persist the artifacts."""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.prepare_data import prepare

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def train() -> dict:
    train_frame, test_frame = prepare()
    candidates = {
        "logistic_regression": Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "linear_svm": Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)), ("classifier", LinearSVC(class_weight="balanced"))]),
    }
    scores = {}
    for name, candidate in candidates.items():
        candidate.fit(train_frame["text"], train_frame["label"])
        scores[name] = float(candidate.score(test_frame["text"], test_frame["label"]))
    search = GridSearchCV(
        candidates["logistic_regression"],
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2], "classifier__C": [0.5, 1.0, 2.0]},
        scoring="f1_macro", cv=3, n_jobs=1,
    )
    search.fit(train_frame["text"], train_frame["label"])
    model = search.best_estimator_
    predictions = model.predict(test_frame["text"])
    labels = [-1, 0, 1]
    report = classification_report(test_frame["label"], predictions, labels=labels, output_dict=True, zero_division=0)
    batch = test_frame["text"].head(50).tolist()
    started = time.perf_counter()
    model.predict(batch)
    inference_ms = (time.perf_counter() - started) * 1000
    metrics = {"accuracy": accuracy_score(test_frame["label"], predictions), "macro_f1": f1_score(test_frame["label"], predictions, average="macro"), "per_class": report, "confusion_matrix": confusion_matrix(test_frame["label"], predictions, labels=labels).tolist(), "inference_50_ms": inference_ms, "candidates": scores, "best_params": search.best_params_}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "sentiment_pipeline.joblib")
    joblib.dump(model.named_steps["tfidf"], MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(model.named_steps["classifier"], MODELS / "sentiment_classifier.joblib")
    with (MODELS / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "inference_50_ms": inference_ms}, indent=2))
    return metrics


if __name__ == "__main__":
    train()
