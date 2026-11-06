import json
import time
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "models" / "metrics.json"
PIPELINE = ROOT / "models" / "sentiment_pipeline.joblib"

requires_metrics = pytest.mark.skipif(not METRICS.exists(), reason="metrics.json not available")
requires_pipeline = pytest.mark.skipif(not PIPELINE.exists(), reason="pipeline not available")


@requires_metrics
def test_accuracy_meets_threshold():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert metrics["accuracy"] >= 0.80


@requires_metrics
def test_per_class_f1_meets_threshold():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    for label in ("-1", "0", "1"):
        assert metrics["per_class"][label]["f1-score"] > 0.75


@requires_metrics
def test_reported_inference_latency():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert metrics["inference_50_ms"] < 100


@requires_pipeline
def test_pipeline_predicts_expected_shape():
    model = joblib.load(PIPELINE)
    comments = ["awesome", "awful", "ok"]
    predictions = model.predict(comments)
    assert len(predictions) == len(comments)
    assert {int(p) for p in predictions}.issubset({-1, 0, 1})


@requires_pipeline
def test_pipeline_handles_edge_cases():
    model = joblib.load(PIPELINE)
    edge = ["a", "😀😀😀", "x" * 5000, "mélange english français"]
    predictions = model.predict(edge)
    assert len(predictions) == len(edge)


@requires_pipeline
def test_pipeline_live_latency_under_100ms():
    model = joblib.load(PIPELINE)
    model.predict(["warmup"])
    batch = ["this is a sample comment"] * 50
    start = time.perf_counter()
    model.predict(batch)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100
