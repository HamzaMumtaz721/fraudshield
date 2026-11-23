import numpy as np
import pandas as pd

clean = pd.read_csv("results_clean.csv")
drift = pd.read_csv("results_drift.csv")
rc = pd.read_csv("replay.csv")
rd = pd.read_csv("replay_drift.csv")
print("rows:", len(clean), len(drift), len(rd), "| i unique:", clean["i"].is_unique, drift["i"].is_unique)

# 1. Which rows got a different decision?
m = clean.merge(drift, on="i", suffixes=("_c", "_d"))
diff = m[m["alert_c"] != m["alert_d"]]
print("\nRows where the alert differs:", len(diff))
for _, r in diff.iterrows():
    row = rd.iloc[int(r["i"])]
    print(f"  i={int(r['i'])} step={int(row['step'])} fraud={int(row['isFraud'])} "
          f"score clean={r['risk_score_c']} drift={r['risk_score_d']} amount(drift)={row['amount']:.0f}")

# 2. Per-window view of the drift run
d = drift.copy()
d["amount"] = rd["amount"].values
d = d[d["scored"] == 1]
d["window"] = pd.cut(d["step"], [400, 499, 599, 699, 743],
                     labels=["401-499", "500-599", "600-699", "700-743"])
print("\nPer window (drift run, scored rows):")
print(d.groupby("window", observed=True).agg(
    rows=("i", "count"), frauds=("fraud", "sum"), alerts=("alert", "sum"),
    median_amount=("amount", "median"), mean_score=("risk_score", "mean")).round(2))

# 3. Population stability index of amount vs the first window
def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))

def window_psi(replay, res, name):
    x = replay.copy()
    x["scored"] = res["scored"].values
    x = x[x["scored"] == 1]
    ref = x[(x["step"] >= 401) & (x["step"] <= 499)]["amount"]
    print(f"\nPSI of amount vs steps 401-499 ({name}):")
    for lo, hi in [(500, 599), (600, 699), (700, 743)]:
        cur = x[(x["step"] >= lo) & (x["step"] <= hi)]["amount"]
        print(f"  {lo}-{hi}: {psi(ref, cur):.3f}")

window_psi(rc, clean, "clean data")
window_psi(rd, drift, "drifted data")