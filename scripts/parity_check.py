import json
import numpy as np
import pandas as pd
import xgboost as xgb

print("xgboost version:", xgb.__version__)

meta = json.load(open("artifacts/meta.json"))
features = meta["features"]

model = xgb.XGBClassifier()
model.load_model("artifacts/xgb_full.json")

def make_row(step, ttype, amount, old_o, new_o, old_d, new_d):
    return {
        "hour": step % 24,
        "amount": amount,
        "is_transfer": int(ttype == "TRANSFER"),
        "oldbalanceOrg": old_o,
        "newbalanceOrig": new_o,
        "oldbalanceDest": old_d,
        "newbalanceDest": new_d,
        "errorOrig": new_o + amount - old_o,
        "errorDest": old_d + amount - new_d,
    }

rows = [
    # Step 22, missed fraud A (Kaggle score 0.0574)
    make_row(586, "TRANSFER", 353874.22, 353874.22, 353874.22, 0.0, 0.0),
    # Step 22, missed fraud B (Kaggle score 0.1088)
    make_row(646, "TRANSFER", 399045.08, 10399045.08, 10399045.08, 0.0, 0.0),
    # Made-up "textbook fraud": balanced books, account drained, new receiver
    make_row(100, "TRANSFER", 500000.0, 500000.0, 0.0, 0.0, 0.0),
    # Made-up ordinary cash-out from an empty account
    make_row(300, "CASH_OUT", 150000.0, 0.0, 0.0, 200000.0, 350000.0),
]

X = pd.DataFrame(rows)[features]
scores = model.predict_proba(X)[:, 1]
for name, s in zip(["missed fraud A", "missed fraud B", "textbook fraud (made up)", "empty-sender cash-out (made up)"], scores):
    print(f"{name}: {s:.4f}")