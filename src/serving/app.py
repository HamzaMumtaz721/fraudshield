import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
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
MODEL = xgb.XGBClassifier()
MODEL.load_model(BASE / "artifacts" / MODEL_FILE)

app = FastAPI(title="FraudShield scoring service")

LOG_PATH = BASE / "logs" / "scores.csv"
LOG_PATH.parent.mkdir(exist_ok=True)
LOG_FIELDS = ["time_utc", "step", "type", "amount", "risk_score", "alert"]
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
    return {"status": "ok", "model": MODEL_FILE, "threshold": THRESHOLD,
            "features": FEATURES}


@app.post("/score")
def score(t: Transaction):
    if t.type not in IN_SCOPE:
        STATS["skipped"] += 1
        return {"scored": False, "alert": False, "risk_tier": "not_scored",
                "reason": "model was trained only on TRANSFER and CASH_OUT"}
    t0 = time.perf_counter()
    s = float(MODEL.predict_proba(build_features(t))[:, 1][0])
    alert = s >= THRESHOLD
    STATS["scored"] += 1
    STATS["alerts"] += int(alert)
    STATS["latency_ms_total"] += (time.perf_counter() - t0) * 1000
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), t.step,
                                t.type, t.amount, round(s, 4), int(alert)])
    return {"scored": True, "risk_score": round(s, 4), "alert": alert,
            "risk_tier": risk_tier(s), "threshold": THRESHOLD}


@app.get("/metrics")
def metrics():
    n = STATS["scored"]
    return {**STATS,
            "alert_rate": round(STATS["alerts"] / n, 4) if n else None,
            "avg_latency_ms": round(STATS["latency_ms_total"] / n, 2) if n else None}