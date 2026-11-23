import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def feats(d):
    return pd.DataFrame({
        "hour": d["step"] % 24, "amount": d["amount"],
        "is_transfer": (d["type"] == "TRANSFER").astype(int),
        "oldbalanceOrg": d["oldbalanceOrg"], "newbalanceOrig": d["newbalanceOrig"],
        "oldbalanceDest": d["oldbalanceDest"], "newbalanceDest": d["newbalanceDest"],
        "errorOrig": d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"],
        "errorDest": d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"],
    })

d = pd.read_csv("replay.csv")
d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]
ref = feats(d[(d["step"] >= 401) & (d["step"] <= 499)])
cur = feats(d[(d["step"] >= 600) & (d["step"] <= 699)])
Report([DataDriftPreset()]).run(cur, ref).save_html("drift_report_clean.html")
print("saved drift_report_clean.html")