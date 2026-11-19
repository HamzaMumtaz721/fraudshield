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

d = pd.read_csv("replay.csv")
d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]
ref_all = d[(d["step"] >= 401) & (d["step"] <= 499)]
ref_norm = ref_all[ref_all["isFraud"] == 0]

w = d[(d["step"] >= 596) & (d["step"] <= 619)]
print("rows in 596-619: steps 596-599 =", int((w["step"] <= 599).sum()),
      "| steps 600-619 =", int((w["step"] >= 600).sum()))

for lo, hi in [(692, 715), (716, 739)]:
    cur = d[(d["step"] >= lo) & (d["step"] <= hi)]
    cur_n = cur[cur["isFraud"] == 0]
    a = {c: round(psi(feats(ref_all)[c], feats(cur)[c]), 3) for c in MONEY}
    b = {c: round(psi(feats(ref_norm)[c], feats(cur_n)[c]), 3) for c in MONEY}
    print(f"\nsteps {lo}-{hi}: rows={len(cur)}, fraud share={cur['isFraud'].mean():.3f}, normal rows={len(cur_n)}")
    print("  PSI, all rows    :", a)
    print("  PSI, normal only :", b)