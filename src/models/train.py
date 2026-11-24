import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_recall_curve

from src.data.ingestion import file_sha256, load_raw
from src.data.validation import validate
from src.features.preprocessing import add_features
from src.models.evaluate import evaluate, passes_gate


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return commit, dirty
    except Exception:
        return "unknown", False


def steps(df, lo_hi):
    return df[(df["step"] >= lo_hi[0]) & (df["step"] <= lo_hi[1])]


def pr_figure(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.set(xlabel="recall", ylabel="precision", ylim=(0, 1.02),
           title="Validation precision-recall curve")
    return fig


def importance_figure(model, features):
    imp = np.asarray(model.feature_importances_)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(features)[order], imp[order])
    ax.set_title("Feature importance (XGBoost default type)")
    fig.tight_layout()
    return fig


def log_model(model):
    try:
        return mlflow.xgboost.log_model(model, name="model")
    except TypeError:                       # older MLflow versions
        return mlflow.xgboost.log_model(model, artifact_path="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_params.yaml")
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="subsample the training rows for a quick dry run")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    client = MlflowClient()

    # ---- data: ingest, validate, features, split ----
    raw_path = cfg["data"]["raw_path"]
    print("loading", raw_path)
    df = load_raw(raw_path)
    validate(df)                                       # raises if the data is bad
    data_sha = file_sha256(raw_path)
    df = add_features(df[df["type"].isin(cfg["data"]["types_in_scope"])])
    train, val, test = (steps(df, cfg["data"][k]) for k in ("train_steps", "val_steps", "test_steps"))
    if args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=1)
    feats = cfg["features"]
    thr, fee = cfg["evaluation"]["threshold"], cfg["evaluation"]["review_cost"]
    n_pos = int((train["isFraud"] == 1).sum())
    base_w = (len(train) - n_pos) / n_pos
    print(f"train {len(train)} rows ({n_pos} frauds) | val {len(val)} | test {len(test)}")

    commit, dirty = git_info()
    tags = {"git_commit": commit, "git_dirty": str(dirty), "data_sha256": data_sha}
    s = cfg["search"]
    combos = list(itertools.product(s["grid"]["max_depth"], s["grid"]["learning_rate"],
                                    s["grid"]["pos_weight_mode"]))
    results = []

    with mlflow.start_run(run_name="tuning-" + datetime.now().strftime("%Y%m%d-%H%M%S")) as parent:
        mlflow.set_tags(tags)
        mlflow.log_params({"n_configs": len(combos), "threshold": thr,
                           "review_cost_assumed": fee, "train_rows": len(train),
                           "val_rows": len(val), "test_rows": len(test), "train_frauds": n_pos,
                           "gate_min_val_pr_auc": cfg["gate"]["min_val_pr_auc"],
                           "gate_min_val_recall": cfg["gate"]["min_val_recall"]})

        for i, (depth, lr, mode) in enumerate(combos):
            weight = base_w if mode == "full" else float(np.sqrt(base_w))
            name = f"xgb-d{depth}-lr{lr}-{mode}"
            with mlflow.start_run(run_name=name, nested=True) as run:
                mlflow.set_tags(tags)
                mlflow.log_params({"model_type": "XGBClassifier", "max_depth": depth,
                                   "learning_rate": lr, "pos_weight_mode": mode,
                                   "scale_pos_weight": round(weight, 2),
                                   "n_estimators_max": s["base"]["n_estimators"],
                                   "early_stopping_rounds": s["base"]["early_stopping_rounds"],
                                   "random_state": s["base"]["random_state"],
                                   "features": ",".join(feats)})
                model = xgb.XGBClassifier(
                    n_estimators=s["base"]["n_estimators"], max_depth=depth, learning_rate=lr,
                    scale_pos_weight=weight, tree_method="hist", eval_metric="aucpr",
                    early_stopping_rounds=s["base"]["early_stopping_rounds"],
                    n_jobs=-1, random_state=s["base"]["random_state"])
                t0 = time.perf_counter()
                model.fit(train[feats], train["isFraud"],
                          eval_set=[(val[feats], val["isFraud"])], verbose=False)
                seconds = time.perf_counter() - t0

                pv = model.predict_proba(val[feats])[:, 1]
                pt = model.predict_proba(test[feats])[:, 1]
                mv = evaluate(val["isFraud"], pv, val["amount"], thr, fee)
                mt = evaluate(test["isFraud"], pt, test["amount"], thr, fee)
                ok, reasons = passes_gate(mv, cfg["gate"])

                mlflow.log_metrics({**{f"val_{k}": v for k, v in mv.items()},
                                    **{f"test_{k}": v for k, v in mt.items()},
                                    "train_seconds": seconds,
                                    "best_iteration": float(model.best_iteration)})
                mlflow.set_tag("gate_passed", str(ok))
                if reasons:
                    mlflow.set_tag("gate_failures", "; ".join(reasons))
                mlflow.log_figure(pr_figure(val["isFraud"], pv), "val_pr_curve.png")
                mlflow.log_figure(importance_figure(model, feats), "feature_importance.png")
                info = log_model(model)
                plt.close("all")

                results.append({"run_name": name, "run_id": run.info.run_id,
                                "model_uri": info.model_uri, "max_depth": depth,
                                "learning_rate": lr, "pos_weight_mode": mode,
                                "val_pr_auc": mv["pr_auc"], "val_recall": mv["recall"],
                                "val_false_alarms": mv["false_alarms"], "val_cost": mv["cost"],
                                "test_pr_auc": mt["pr_auc"], "gate_passed": ok})
                print(f"[{i + 1}/{len(combos)}] {name}: val PR-AUC {mv['pr_auc']:.4f}, "
                      f"val recall {mv['recall']:.4f}, gate {'PASS' if ok else 'FAIL'}")

        res = pd.DataFrame(results)
        mlflow.log_text(res.drop(columns=["model_uri"]).to_csv(index=False), "comparison.csv")
        show = ["run_name", "val_pr_auc", "val_recall", "val_false_alarms", "test_pr_auc", "gate_passed"]
        print("\n" + res[show].round(4).to_string(index=False))

        passing = res[res["gate_passed"]].copy()
        if passing.empty:
            mlflow.set_tag("outcome", "no model passed the quality gate")
            print("\nNo model passed the quality gate. Nothing was registered.")
            return 1

        passing["pr_auc_r"] = passing["val_pr_auc"].round(4)
        best = passing.sort_values(["pr_auc_r", "val_cost"], ascending=[False, True]).iloc[0]
        reg_name = cfg["mlflow"]["registered_model"]
        mv_new = mlflow.register_model(best["model_uri"], reg_name)
        client.set_registered_model_alias(reg_name, "candidate", mv_new.version)
        for k, v in {"source_run_id": best["run_id"], "val_pr_auc": f"{best['val_pr_auc']:.4f}",
                     "git_commit": commit, "data_sha256": data_sha}.items():
            client.set_model_version_tag(reg_name, mv_new.version, k, v)
        mlflow.set_tags({"outcome": "registered", "selected_run": best["run_name"],
                         "registered_version": str(mv_new.version)})
        print(f"\nSelected {best['run_name']} (highest validation PR-AUC among runs passing the gate; "
              f"ties broken by lower validation cost).")
        print(f"Registered {reg_name} version {mv_new.version} with alias 'candidate'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())