from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

TEXTBOOK = {"step": 100, "type": "TRANSFER", "amount": 500000,
            "oldbalanceOrg": 500000, "newbalanceOrig": 0,
            "oldbalanceDest": 0, "newbalanceDest": 0}
MISSED_A = {"step": 586, "type": "TRANSFER", "amount": 353874.22,
            "oldbalanceOrg": 353874.22, "newbalanceOrig": 353874.22,
            "oldbalanceDest": 0, "newbalanceDest": 0}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_model_info():
    j = client.get("/model-info").json()
    assert j["threshold"] == 0.9
    assert j["model_version"]
    assert len(j["sha256"]) == 64
    assert "errorOrig" in j["features"]


def test_textbook_fraud_alerts():
    j = client.post("/predict", json=TEXTBOOK).json()
    assert j["scored"] is True
    assert abs(j["risk_score"] - 0.9603) < 0.001
    assert j["alert"] is True
    assert j["risk_tier"] == "high"
    assert j["model_version"]


def test_known_missed_fraud_matches_notebook():
    j = client.post("/predict", json=MISSED_A).json()
    assert abs(j["risk_score"] - 0.0574) < 0.001
    assert j["alert"] is False


def test_score_alias_still_works():
    a = client.post("/predict", json=TEXTBOOK).json()
    b = client.post("/score", json=TEXTBOOK).json()
    assert a["risk_score"] == b["risk_score"]


def test_payment_is_not_scored():
    body = {"step": 50, "type": "PAYMENT", "amount": 1000, "oldbalanceOrg": 5000,
            "newbalanceOrig": 4000, "oldbalanceDest": 0, "newbalanceDest": 0}
    j = client.post("/predict", json=body).json()
    assert j["scored"] is False
    assert j["alert"] is False
    assert j["risk_tier"] == "not_scored"


def test_invalid_type_rejected():
    body = dict(TEXTBOOK, type="BITCOIN")
    assert client.post("/predict", json=body).status_code == 422


def test_negative_amount_rejected():
    body = dict(TEXTBOOK, amount=-5)
    assert client.post("/predict", json=body).status_code == 422


def test_missing_field_rejected():
    body = {k: v for k, v in TEXTBOOK.items() if k != "amount"}
    assert client.post("/predict", json=body).status_code == 422


def test_stats_counts_scored_rows():
    before = client.get("/stats").json()["scored"]
    client.post("/predict", json=TEXTBOOK)
    assert client.get("/stats").json()["scored"] == before + 1


def test_prometheus_metrics_exposed():
    client.post("/predict", json=TEXTBOOK)
    client.post("/predict", json=dict(TEXTBOOK, amount=-5))   # one invalid request
    client.get("/no-such-page")
    text = client.get("/metrics").text
    assert "fraudshield_requests_total" in text
    assert "fraudshield_request_latency_seconds_bucket" in text
    assert "fraudshield_predictions_total" in text
    assert "fraudshield_model_info" in text
    assert 'status="422"' in text       # data-quality failures are counted
    assert 'endpoint="other"' in text   # unknown paths do not create new labels