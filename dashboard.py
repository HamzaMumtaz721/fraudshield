import csv
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="FraudShield", layout="wide")

API = "http://127.0.0.1:8000"
PRED_LOG = "logs/dashboard_predictions.csv"
FIELDS = ["time_utc", "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
          "oldbalanceDest", "newbalanceDest", "scored", "risk_score", "alert", "risk_tier"]
os.makedirs("logs", exist_ok=True)
if not os.path.exists(PRED_LOG):
    with open(PRED_LOG, "w", newline="") as f:
        csv.writer(f).writerow(FIELDS)

if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page):
    st.session_state.page = page


def log_prediction(p, r):
    with open(PRED_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(), p["step"], p["type"], p["amount"],
            p["oldbalanceOrg"], p["newbalanceOrig"], p["oldbalanceDest"], p["newbalanceDest"],
            int(r["scored"]), r.get("risk_score", ""), int(r["alert"]), r["risk_tier"]])


# ---------------------------------------------------------------- HOME
def home_page():
    st.title("FraudShield")
    st.caption("Predictions made through this dashboard. 'Flagged' means the model raised an "
               "alert. The true label of a typed-in transaction is unknown.")
    df = pd.read_csv(PRED_LOG)
    total = len(df)
    scored = df[df["scored"] == 1]
    flagged = int((scored["alert"] == 1).sum())
    not_flagged = int((scored["alert"] == 0).sum())
    not_scored = int((df["scored"] == 0).sum())

    a, b, c, d = st.columns(4)
    a.metric("Predictions made", total)
    b.metric("Flagged as fraud", flagged)
    c.metric("Not flagged", not_flagged)
    d.metric("Not scored (other types)", not_scored)

    st.write("")
    left, right = st.columns(2)
    left.button("Make a prediction", on_click=go, args=("predict",),
                use_container_width=True, type="primary")
    right.button("View monitoring (clean / drift runs)", on_click=go, args=("monitor",),
                 use_container_width=True)

    if total:
        st.subheader("Flagged vs not flagged")
        st.bar_chart(pd.Series({"flagged as fraud": flagged, "not flagged": not_flagged,
                                "not scored": not_scored}))
        st.subheader("Latest predictions")
        st.dataframe(df.tail(10).iloc[::-1][["time_utc", "type", "amount", "risk_score",
                                             "alert", "risk_tier"]],
                     use_container_width=True)
    else:
        st.info("No predictions yet. Click 'Make a prediction'.")


# ---------------------------------------------------------------- PREDICT
PRESETS = {
    "Custom": dict(step=100, type="TRANSFER", amount=0.0, oldO=0.0, newO=0.0, oldD=0.0, newD=0.0),
    "Textbook fraud pattern": dict(step=100, type="TRANSFER", amount=500000.0,
                                   oldO=500000.0, newO=0.0, oldD=0.0, newD=0.0),
    "Ordinary cash-out": dict(step=300, type="CASH_OUT", amount=150000.0,
                              oldO=0.0, newO=0.0, oldD=200000.0, newD=350000.0),
    "Known missed fraud (from the notebook)": dict(step=586, type="TRANSFER", amount=353874.22,
                                                   oldO=353874.22, newO=353874.22, oldD=0.0, newD=0.0),
    "Payment (not scored)": dict(step=50, type="PAYMENT", amount=1000.0,
                                 oldO=5000.0, newO=4000.0, oldD=0.0, newD=0.0),
}
TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]


def predict_page():
    st.button("< Back to home", on_click=go, args=("home",))
    st.title("Make a prediction")
    st.caption("The form sends the transaction to the scoring API. "
               "Start the API first: uvicorn app:app --log-level warning")

    preset = st.selectbox("Start from an example", list(PRESETS))
    v = PRESETS[preset]
    k = preset  # new keys per preset so the fields reset when the example changes

    with st.form("tx"):
        c1, c2, c3 = st.columns(3)
        step = c1.number_input("Step (hours since start)", min_value=1, value=v["step"],
                               step=1, key=f"step_{k}")
        ttype = c2.selectbox("Type", TYPES, index=TYPES.index(v["type"]), key=f"type_{k}")
        amount = c3.number_input("Amount", min_value=0.0, value=v["amount"],
                                 format="%.2f", key=f"amt_{k}")
        c4, c5, c6, c7 = st.columns(4)
        oldO = c4.number_input("Sender balance before", min_value=0.0, value=v["oldO"],
                               format="%.2f", key=f"oo_{k}")
        newO = c5.number_input("Sender balance after", min_value=0.0, value=v["newO"],
                               format="%.2f", key=f"no_{k}")
        oldD = c6.number_input("Receiver balance before", min_value=0.0, value=v["oldD"],
                               format="%.2f", key=f"od_{k}")
        newD = c7.number_input("Receiver balance after", min_value=0.0, value=v["newD"],
                               format="%.2f", key=f"nd_{k}")
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {"step": int(step), "type": ttype, "amount": float(amount),
                   "oldbalanceOrg": float(oldO), "newbalanceOrig": float(newO),
                   "oldbalanceDest": float(oldD), "newbalanceDest": float(newD)}
        try:
            r = requests.post(API + "/score", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the scoring API at {API}. Is it running? ({type(e).__name__})")
            return
        log_prediction(payload, res)
        if not res["scored"]:
            st.warning("Not scored: the model was trained only on TRANSFER and CASH_OUT.")
        elif res["alert"]:
            st.error(f"FLAGGED as possible fraud. Risk score {res['risk_score']} "
                     f"(threshold {res['threshold']}, tier {res['risk_tier']}).")
        else:
            st.success(f"Not flagged. Risk score {res['risk_score']} "
                       f"(threshold {res['threshold']}, tier {res['risk_tier']}).")
        st.caption("The risk score is a model output, not a calibrated probability of fraud.")


# ---------------------------------------------------------------- MONITOR
MONEY = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest", "errorOrig"]
WINDOW = 24


@st.cache_data
def load(replay_path, results_path):
    rp = pd.read_csv(replay_path)
    rs = pd.read_csv(results_path)
    rp["scored"] = rs["scored"].values
    rp["alert"] = rs["alert"].values
    rp["risk_score"] = rs["risk_score"].values
    return rp


def psi(ref, cur):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    q = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feats(d):
    return pd.DataFrame({
        "amount": d["amount"], "oldbalanceOrg": d["oldbalanceOrg"],
        "oldbalanceDest": d["oldbalanceDest"], "newbalanceDest": d["newbalanceDest"],
        "errorOrig": d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]})


def window_table(d, min_rows):
    d = d[d["scored"] == 1]
    ref = feats(d[(d["step"] >= 401) & (d["step"] <= 499)])
    out = []
    for lo in range(500, 744, WINDOW):
        w = d[(d["step"] >= lo) & (d["step"] < lo + WINDOW)]
        if len(w) < min_rows:
            continue
        cur = feats(w)
        p = {c: psi(ref[c], cur[c]) for c in MONEY}
        n_over = sum(x > 0.1 for x in p.values())
        out.append({"window": f"{lo}-{lo + WINDOW - 1}", "rows": len(w),
                    "frauds": int(w["isFraud"].sum()), "alerts": int(w["alert"].sum()),
                    "missed": int(((w["alert"] == 0) & (w["isFraud"] == 1)).sum()),
                    "false_alarms": int(((w["alert"] == 1) & (w["isFraud"] == 0)).sum()),
                    "cols_over_0.1": n_over, "max_PSI": round(max(p.values()), 3),
                    "drift_alarm": n_over >= 2})
    return pd.DataFrame(out)


def monitor_page():
    st.button("< Back to home", on_click=go, args=("home",))
    st.title("Monitoring: replay of PaySim steps 401-743")
    st.caption("Drift was injected by doubling all money columns from step 600 in the 'drift' "
               "run. Easy model, threshold 0.90.")

    run = st.sidebar.radio("Run", ["drift (money doubled from step 600)", "clean"])
    min_rows = st.sidebar.slider("Minimum rows per window", 0, 3000, 500, 100)
    st.sidebar.write("Alarm rule: 2 or more of 5 money columns with PSI above 0.1. "
                     "Thresholds are my choice, not tuned.")

    if run.startswith("drift"):
        data = load("replay_drift.csv", "results_drift.csv")
    else:
        data = load("replay.csv", "results_clean.csv")

    sc = data[data["scored"] == 1]
    tp = int(((sc["alert"] == 1) & (sc["isFraud"] == 1)).sum())
    fp = int(((sc["alert"] == 1) & (sc["isFraud"] == 0)).sum())
    fn = int(((sc["alert"] == 0) & (sc["isFraud"] == 1)).sum())

    a, b, c, d_ = st.columns(4)
    a.metric("Rows scored", f"{len(sc):,}")
    b.metric("Frauds caught", f"{tp:,} / {tp + fn:,}")
    c.metric("False alarms", fp)
    d_.metric("Rows skipped (other types)", f"{int((data['scored'] == 0).sum()):,}")

    wt = window_table(data, min_rows)

    st.subheader("Drift: max PSI per window")
    st.line_chart(wt.set_index("window")["max_PSI"])
    st.caption("Windows with fewer rows than the slider are hidden. Small windows give noisy PSI.")

    st.subheader("Alerts vs real frauds per window")
    st.bar_chart(wt.set_index("window")[["frauds", "alerts"]])

    st.subheader("Window details")
    st.dataframe(wt, use_container_width=True)

    if wt["drift_alarm"].any():
        st.error(f"Drift alarm first fires in window {wt[wt['drift_alarm']].iloc[0]['window']}.")
    else:
        st.success("No drift alarm in any window.")

    st.subheader("Risk score distribution (scored rows)")
    st.bar_chart(pd.cut(sc["risk_score"].astype(float), [0, .1, .3, .5, .7, .9, 1.0])
                 .value_counts().sort_index().rename(lambda x: str(x)))


# ---------------------------------------------------------------- ROUTER
{"home": home_page, "predict": predict_page, "monitor": monitor_page}[st.session_state.page]()