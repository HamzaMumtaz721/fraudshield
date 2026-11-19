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

def psi_arrays(p, q):
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))

def psi(ref, cur, col):
    if col in ("is_transfer", "hour"):          # categorical: compare category shares
        cats = sorted(set(ref) | set(cur))
        p = np.array([(ref == c).mean() for c in cats])
        q = np.array([(cur == c).mean() for c in cats])
        return psi_arrays(p, q)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_arrays(np.histogram(ref, edges)[0] / len(ref),
                      np.histogram(cur, edges)[0] / len(cur))

out = {}
for name, path in [("clean", "replay.csv"), ("drift", "replay_drift.csv")]:
    d = pd.read_csv(path)
    d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]
    ref = feats(d[(d["step"] >= 401) & (d["step"] <= 499)])
    cur = feats(d[(d["step"] >= 600) & (d["step"] <= 699)])
    out[name] = {c: round(psi(ref[c], cur[c], c), 3) for c in ref.columns}

print(pd.DataFrame(out).sort_values("drift", ascending=False))

# Is the hour shift a fraud-share effect?
d = pd.read_csv("replay.csv")
d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]
for lo, hi in [(401, 499), (600, 699)]:
    w = d[(d["step"] >= lo) & (d["step"] <= hi)]
    print(f"\nsteps {lo}-{hi}: rows={len(w)}, fraud share={w['isFraud'].mean():.3f}")
    print(" share of rows in night hours 2-6, fraud:",
          round(((w['step'] % 24).between(2, 6))[w['isFraud'] == 1].mean(), 3),
          "| normal:", round(((w['step'] % 24).between(2, 6))[w['isFraud'] == 0].mean(), 3))