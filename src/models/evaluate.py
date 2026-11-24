import numpy as np
from sklearn.metrics import average_precision_score


def evaluate(y, p, amount, threshold: float, review_cost: float) -> dict:
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount)
    flag = p >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int(((~flag) & (y == 1)).sum())
    cost = float(amount[(~flag) & (y == 1)].sum() + review_cost * fp)
    return {"pr_auc": float(average_precision_score(y, p)),
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "missed": fn, "false_alarms": fp, "alerts": tp + fp, "cost": cost}


def passes_gate(val_metrics: dict, gate: dict):
    reasons = []
    if val_metrics["pr_auc"] < gate["min_val_pr_auc"]:
        reasons.append(f"val PR-AUC {val_metrics['pr_auc']:.4f} < {gate['min_val_pr_auc']}")
    if val_metrics["recall"] < gate["min_val_recall"]:
        reasons.append(f"val recall {val_metrics['recall']:.4f} < {gate['min_val_recall']}")
    return (not reasons), reasons