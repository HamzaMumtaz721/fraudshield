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
    assert r.json()["threshold"] == 0.9


def test_textbook_fraud_alerts():
    j = client.post("/score", json=TEXTBOOK).json()
    assert j["scored"] is True
    assert abs(j["risk_score"] - 0.9603) < 0.001
    assert j["alert"] is True
    assert j["risk_tier"] == "high"


def test_known_missed_fraud_matches_notebook():
    j = client.post("/score", json=MISSED_A).json()
    assert abs(j["risk_score"] - 0.0574) < 0.001
    assert j["alert"] is False


def test_payment_is_not_scored():
    body = {"step": 50, "type": "PAYMENT", "amount": 1000, "oldbalanceOrg": 5000,
            "newbalanceOrig": 4000, "oldbalanceDest": 0, "newbalanceDest": 0}
    j = client.post("/score", json=body).json()
    assert j["scored"] is False
    assert j["alert"] is False
    assert j["risk_tier"] == "not_scored"


def test_invalid_type_rejected():
    body = dict(TEXTBOOK, type="BITCOIN")
    assert client.post("/score", json=body).status_code == 422


def test_negative_amount_rejected():
    body = dict(TEXTBOOK, amount=-5)
    assert client.post("/score", json=body).status_code == 422


def test_metrics_counts_scored_rows():
    before = client.get("/metrics").json()["scored"]
    client.post("/score", json=TEXTBOOK)
    assert client.get("/metrics").json()["scored"] == before + 1