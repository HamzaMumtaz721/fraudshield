import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.features.preprocessing import add_features

FEATURES = json.load(open("artifacts/meta.json"))["features"]
THRESHOLD = 0.90
REVIEW_COST = 50   # assumed, not measured


def main():
    d = pd.read_csv("replay_drift.csv")
    d = add_features(d[d["type"].isin(["TRANSFER", "CASH_OUT"])])

    train = d[(d["step"] >= 401) & (d["step"] <= 650)]   # includes drifted steps 600-650
    hold = d[d["step"] > 650]                            # never trained on
    print("train rows:", len(train), "frauds:", int(train["isFraud"].sum()),
          "| holdout rows:", len(hold), "frauds:", int(hold["isFraud"].sum()))

    spw = (train["isFraud"] == 0).sum() / (train["isFraud"] == 1).sum()
    challenger = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                                   scale_pos_weight=spw, tree_method="hist",
                                   n_jobs=-1, random_state=1)
    challenger.fit(train[FEATURES], train["isFraud"])

    champion = xgb.XGBClassifier()
    champion.load_model("artifacts/xgb_full.json")

    def evaluate(model):
        p = model.predict_proba(hold[FEATURES])[:, 1]
        y = hold["isFraud"].values
        flag = p >= THRESHOLD
        missed = int(((~flag) & (y == 1)).sum())
        fa = int((flag & (y == 0)).sum())
        cost = hold["amount"].values[(~flag) & (y == 1)].sum() + REVIEW_COST * fa
        return {"pr_auc": round(float(average_precision_score(y, p)), 4),
                "missed": missed, "false_alarms": fa, "cost": round(float(cost))}

    c, ch = evaluate(champion), evaluate(challenger)
    print("champion  :", c)
    print("challenger:", ch)

    promote = ch["pr_auc"] > c["pr_auc"] and ch["cost"] <= c["cost"]
    print("PROMOTE challenger" if promote else "KEEP champion")

    if promote:
        challenger.save_model("artifacts/xgb_full_v2.json")
        json.dump({"champion": "xgb_full_v2.json", "previous": "xgb_full.json"},
                  open("artifacts/registry.json", "w"), indent=2)
        print("saved artifacts/xgb_full_v2.json and registry.json")


if __name__ == "__main__":
    main()