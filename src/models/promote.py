import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient

from src.data.ingestion import load_raw
from src.data.validation import validate
from src.features.preprocessing import add_features
from src.models.evaluate import evaluate, passes_gate

ART = Path("artifacts")
REG = ART / "registry.json"
BASELINE = "xgb_full.json"      # the model currently deployed (trained on Kaggle)
MAX_PR_AUC_DROP = 0.001         # promotion rule, fixed in advance (my choice, not tuned)


def setup(config_path):
    cfg = yaml.safe_load(open(config_path))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    return cfg, MlflowClient(), cfg["mlflow"]["registered_model"]


def read_state():
    if REG.exists():
        return json.load(open(REG))
    return {"champion": BASELINE, "previous": None, "version": None, "previous_version": None}


def holdouts(cfg):
    df = load_raw(cfg["data"]["raw_path"])
    validate(df)
    df = add_features(df[df["type"].isin(cfg["data"]["types_in_scope"])])

    def pick(key):
        lo, hi = cfg["data"][key]
        return df[(df["step"] >= lo) & (df["step"] <= hi)]

    return pick("val_steps"), pick("test_steps")


def score(model, d, cfg):
    p = model.predict_proba(d[cfg["features"]])[:, 1]
    return evaluate(d["isFraud"], p, d["amount"], cfg["evaluation"]["threshold"],
                    cfg["evaluation"]["review_cost"])


def promote(args):
    cfg, client, name = setup(args.config)
    try:
        cand = client.get_model_version_by_alias(name, "candidate")
    except Exception:
        sys.exit("No model version has the alias 'candidate'. Run the training first.")

    state = read_state()
    champ_model = xgb.XGBClassifier()
    champ_model.load_model(ART / state["champion"])
    cand_model = mlflow.xgboost.load_model(f"models:/{name}@candidate")

    val, test = holdouts(cfg)
    r = {who: {"val": score(m, val, cfg), "test": score(m, test, cfg)}
         for who, m in (("champion", champ_model), ("candidate", cand_model))}

    gate_ok, gate_reasons = passes_gate(r["candidate"]["val"], cfg["gate"])
    c, k = r["champion"]["test"], r["candidate"]["test"]
    cost_ok = k["cost"] <= c["cost"]
    auc_ok = k["pr_auc"] >= c["pr_auc"] - MAX_PR_AUC_DROP

    print(f"\nChampion : {state['champion']}")
    print(f"Candidate: {name} version {cand.version}")
    for split in ("val", "test"):
        for who in ("champion", "candidate"):
            m = r[who][split]
            print(f"  {split:4s} {who:9s} PR-AUC {m['pr_auc']:.4f} recall {m['recall']:.4f} "
                  f"missed {m['missed']:>4d} false alarms {m['false_alarms']:>4d} cost {m['cost']:,.0f}")

    if not gate_ok:
        decision, reason = "REJECT", "failed quality gate: " + "; ".join(gate_reasons)
    elif cost_ok and auc_ok:
        decision, reason = "PROMOTE", "passed the gate; test cost not higher and PR-AUC within tolerance"
    elif args.manual_approve:
        decision, reason = "PROMOTE", f"MANUAL APPROVAL (comparison not met): {args.manual_approve}"
    else:
        why = []
        if not cost_ok:
            why.append(f"candidate test cost {k['cost']:,.0f} > champion {c['cost']:,.0f}")
        if not auc_ok:
            why.append(f"candidate PR-AUC {k['pr_auc']:.4f} < champion {c['pr_auc']:.4f} - {MAX_PR_AUC_DROP}")
        decision, reason = "KEEP", "; ".join(why)
    print(f"\nDecision: {decision}. {reason}")

    if args.dry_run:
        print("(dry run: nothing changed)")
        return 0

    with mlflow.start_run(run_name=f"promotion-v{cand.version}"):
        mlflow.set_tags({"decision": decision, "reason": reason[:500],
                         "candidate_version": str(cand.version),
                         "champion_file": state["champion"]})
        for who in r:
            for split in r[who]:
                for key in ("pr_auc", "recall", "missed", "false_alarms", "cost"):
                    mlflow.log_metric(f"{who}_{split}_{key}", float(r[who][split][key]))

    if decision != "PROMOTE":
        return 0 if decision == "KEEP" else 1

    new_file = f"{name}_v{cand.version}.json"
    cand_model.save_model(ART / new_file)
    json.dump({"champion": new_file, "previous": state["champion"],
               "version": int(cand.version), "previous_version": state["version"]},
              open(REG, "w"), indent=2)
    client.set_registered_model_alias(name, "champion", str(cand.version))
    if state["version"]:
        client.set_registered_model_alias(name, "previous", str(state["version"]))
    client.delete_registered_model_alias(name, "candidate")
    client.set_model_version_tag(name, cand.version, "promotion_reason", reason[:250])
    print(f"Promoted. Exported {ART / new_file} and updated {REG}.")
    print("Rebuild the API to serve it:  docker compose up -d --build api")
    return 0


def rollback(args):
    cfg, client, name = setup(args.config)
    if not REG.exists():
        sys.exit("No promotion has happened yet, so there is nothing to roll back.")
    s = json.load(open(REG))
    if not s.get("previous"):
        sys.exit("registry.json has no previous model.")
    new = {"champion": s["previous"], "previous": s["champion"],
           "version": s.get("previous_version"), "previous_version": s.get("version")}
    json.dump(new, open(REG, "w"), indent=2)
    for alias, ver in (("champion", new["version"]), ("previous", new["previous_version"])):
        try:
            if ver:
                client.set_registered_model_alias(name, alias, str(ver))
            else:
                client.delete_registered_model_alias(name, alias)
        except Exception:
            pass
    with mlflow.start_run(run_name="rollback"):
        mlflow.set_tags({"decision": "ROLLBACK", "now_serving": new["champion"],
                         "was_serving": new["previous"]})
    print(f"Rolled back. Serving {new['champion']} (was {new['previous']}).")
    print("Rebuild the API to apply:  docker compose up -d --build api")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--config", default="configs/model_params.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manual-approve", default="",
                   help="reason text; overrides the champion comparison only, never the quality gate")
    r = sub.add_parser("rollback")
    r.add_argument("--config", default="configs/model_params.yaml")
    args = ap.parse_args()
    return promote(args) if args.cmd == "promote" else rollback(args)


if __name__ == "__main__":
    sys.exit(main())