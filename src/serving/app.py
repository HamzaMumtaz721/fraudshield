import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.features.preprocessing import add_features

BASE = Path(__file__).resolve().parents[2]   # project root
META = json.load(open(BASE / "artifacts" / "meta.json"))
FEATURES = META["features"]
THRESHOLD = META["threshold"]
IN_SCOPE = {"TRANSFER", "CASH_OUT"}

# The champion model is named in registry.json once a challenger has been promoted
REG = BASE / "artifacts" / "registry.json"
MODEL_FILE = json.load(open(REG))["champion"] if REG.exists() else "xgb_full.json"
MODEL_PATH = BASE / "artifacts" / MODEL_FILE
MODEL = xgb.XGBClassifier()
MODEL.load_model(MODEL_PATH)
MODEL_VERSION = MODEL_FILE.removesuffix(".json")
MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

app = FastAPI(title="FraudShield scoring service")

# ---- Prometheus metrics -------------------------------------------------
REQUESTS = Counter("fraudshield_requests_total", "HTTP requests",
                   ["endpoint", "method", "status"])
LATENCY = Histogram("fraudshield_request_latency_seconds", "HTTP request latency",
                    ["endpoint"], buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5])
PREDICTIONS = Counter("fraudshield_predictions_total", "Predictions by model version",
                      ["model_version", "outcome"])
RISK = Histogram("fraudshield_risk_score", "Risk scores of scored transactions",
                 buckets=[.1, .3, .5, .7, .9, .95, 1.0])
MODEL_INFO = Gauge("fraudshield_model_info", "Currently loaded model",
                   ["model_version", "threshold"])
MODEL_INFO.labels(MODEL_VERSION, str(THRESHOLD)).set(1)

# Only these paths get their own label; anything else is "other" so scanners
# cannot create unlimited label values.
KNOWN_PATHS = {"/health", "/model-info", "/predict", "/score", "/stats", "/docs", "/openapi.json"}


@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path if request.url.path in KNOWN_PATHS else "other"
        REQUESTS.labels(path, request.method, str(status)).inc()
        LATENCY.labels(path).observe(time.perf_counter() - start)


# ---- prediction log (no account ids are ever received or stored) ---------
LOG_PATH = BASE / "logs" / "predictions.csv"
LOG_PATH.parent.mkdir(exist_ok=True)
LOG_FIELDS = ["time_utc", "model_version", "step", "type", "amount", "risk_score", "alert"]
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(LOG_FIELDS)

STATS = {"scored": 0, "alerts": 0, "skipped": 0, "latency_ms_total": 0.0}


class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


def build_features(t: Transaction) -> pd.DataFrame:
    return add_features(pd.DataFrame([t.model_dump()]))[FEATURES]


def risk_tier(score: float) -> str:
    # Only THRESHOLD was tuned; the 0.5 cut-off is my choice.
    if score >= THRESHOLD:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {"model_version": MODEL_VERSION, "model_file": MODEL_FILE,
            "sha256": MODEL_SHA256, "xgboost_version": xgb.__version__,
            "threshold": THRESHOLD, "features": FEATURES,
            "types_in_scope": sorted(IN_SCOPE),
            "train_steps": META.get("train_steps"),
            "val_steps": META.get("val_steps"),
            "review_cost_assumed": META.get("review_cost_assumed")}


@app.post("/predict")
@app.post("/score")   # kept as an alias so existing clients keep working
def predict(t: Transaction):
    if t.type not in IN_SCOPE:
        STATS["skipped"] += 1
        PREDICTIONS.labels(MODEL_VERSION, "not_scored").inc()
        return {"scored": False, "alert": False, "risk_tier": "not_scored",
                "model_version": MODEL_VERSION,
                "reason": "model was trained only on TRANSFER and CASH_OUT"}
    t0 = time.perf_counter()
    s = float(MODEL.predict_proba(build_features(t))[:, 1][0])
    alert = s >= THRESHOLD
    STATS["scored"] += 1
    STATS["alerts"] += int(alert)
    STATS["latency_ms_total"] += (time.perf_counter() - t0) * 1000
    PREDICTIONS.labels(MODEL_VERSION, "alert" if alert else "no_alert").inc()
    RISK.observe(s)
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), MODEL_VERSION,
                                t.step, t.type, t.amount, round(s, 4), int(alert)])
    return {"scored": True, "risk_score": round(s, 4), "alert": alert,
            "risk_tier": risk_tier(s), "threshold": THRESHOLD,
            "model_version": MODEL_VERSION}


@app.get("/stats")
def stats():
    n = STATS["scored"]
    return {**STATS,
            "alert_rate": round(STATS["alerts"] / n, 4) if n else None,
            "avg_latency_ms": round(STATS["latency_ms_total"] / n, 2) if n else None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)