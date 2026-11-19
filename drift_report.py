import numpy as np
import pandas as pd

def feats(d):
    return pd.DataFrame({
        "hour": d["step"] % 24, "amount": d["amount"],
        "is_transfer": (d["type"] == "TRANSFER").astype(int),
        "oldbalanceOrg": d["oldbalanceOrg"], "newbalanceOrig": d["newbalanceOrig"],
        "oldbalanceDest": d["oldbalanceDest"], "newbalanceDest": d["newbalanceDest"],
        "errorOrig": d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"],
        "errorDest": d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"],
    })

def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")      # almost constant column: PSI not meaningful
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))

rd = pd.read_csv("replay_drift.csv")
rd = rd[rd["type"].isin(["TRANSFER", "CASH_OUT"])]
ref = feats(rd[(rd["step"] >= 401) & (rd["step"] <= 499)])
cur = feats(rd[(rd["step"] >= 600) & (rd["step"] <= 699)])

rows = [{"feature": c, "PSI": round(psi(ref[c], cur[c]), 3)} for c in ref.columns]
print(pd.DataFrame(rows).sort_values("PSI", ascending=False).to_string(index=False))

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    r = Report([DataDriftPreset()])
    snap = r.run(cur, ref)
    snap.save_html("drift_report.html")
    print("\nEvidently report saved: drift_report.html")
except Exception as e:
    print("\nEvidently step failed:", type(e).__name__, e)