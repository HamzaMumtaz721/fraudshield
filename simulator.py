import argparse
import csv
import os
import time

import pandas as pd
import requests

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--start-step", type=int, default=401)
p.add_argument("--limit", type=int, default=5_000_000)
p.add_argument("--rate", type=float, default=0, help="rows per second (0 = as fast as possible)")
p.add_argument("--replay", default="replay.csv", help="input file to replay")
p.add_argument("--out", default="results.csv", help="where to save per-row results")
a = p.parse_args()

df = pd.read_csv(a.replay)
df = df[df["step"] >= a.start_step].head(a.limit).reset_index(drop=True)

OUT = a.out
done = 0
if os.path.exists(OUT):
    done = max(sum(1 for _ in open(OUT)) - 1, 0)   # rows already saved
print(f"total rows: {len(df)} | already done: {done}")

fields = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
          "oldbalanceDest", "newbalanceDest"]
sess = requests.Session()


def post(payload):
    global sess
    for attempt in range(5):
        try:
            r = sess.post(a.url + "/score", json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"  retry {attempt + 1}/5 after {type(e).__name__}")
            sess.close()
            sess = requests.Session()
            time.sleep(2)
    raise SystemExit("Server not reachable. Restart it, then re-run this same command to resume.")


new_file = done == 0
f = open(OUT, "a", newline="")
w = csv.writer(f)
if new_file:
    w.writerow(["i", "step", "type", "fraud", "scored", "risk_score", "alert", "latency_ms"])

for i in range(done, len(df)):
    row = df.iloc[i]
    payload = {k: row[k] for k in fields}
    payload["step"] = int(payload["step"])
    payload["amount"] = float(payload["amount"])
    for k in fields[3:]:
        payload[k] = float(payload[k])
    t0 = time.perf_counter()
    j = post(payload)
    lat = (time.perf_counter() - t0) * 1000
    w.writerow([i, payload["step"], payload["type"], int(row["isFraud"]),
                int(j["scored"]), j.get("risk_score", ""), int(j["alert"]), round(lat, 2)])
    if a.rate:
        time.sleep(1 / a.rate)
    if (i + 1) % 1000 == 0:
        f.flush()
        print("sent", i + 1)
f.close()

# Summary from the saved file, so it also works after a resume
out = pd.read_csv(OUT)
sc = out[out["scored"] == 1]
tp = int(((sc["alert"] == 1) & (sc["fraud"] == 1)).sum())
fp = int(((sc["alert"] == 1) & (sc["fraud"] == 0)).sum())
fn = int(((sc["alert"] == 0) & (sc["fraud"] == 1)).sum())
print("\nrows:", len(out), "| scored:", len(sc), "| skipped (other types):", len(out) - len(sc))
print("frauds in skipped rows:", int(out[out["scored"] == 0]["fraud"].sum()))
print(f"alerts: {tp + fp} | frauds caught: {tp}/{tp + fn} | false alarms: {fp}")
print(f"avg round-trip latency: {out['latency_ms'].mean():.1f} ms")