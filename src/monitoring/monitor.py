import numpy as np
import pandas as pd

MONEY = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest", "errorOrig"]

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

for name, path in [("clean", "replay.csv"), ("drift", "replay_drift.csv")]:
    d = pd.read_csv(path)
    d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]
    ref = feats(d[(d["step"] >= 401) & (d["step"] <= 499)])
    print(f"\n{name}: windows of 24 steps vs reference 401-499 (alarm if 2+ columns have PSI > 0.1)")
    for lo in range(500, 744, 24):
        cur = feats(d[(d["step"] >= lo) & (d["step"] < lo + 24)])
        if len(cur) < 500:
            continue
        p = {c: psi(ref[c], cur[c]) for c in MONEY}
        n = sum(v > 0.1 for v in p.values())
        print(f"  steps {lo}-{lo+23}: rows={len(cur)}, columns over 0.1: {n}, max PSI={max(p.values()):.3f}, ALARM={n >= 2}")