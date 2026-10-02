"""
Computes classification accuracy, precision, false positive rate and false negative rate
from flood_model.py's output, at each forecast lead time (1, 2, 3, 6 months). flood_model.py
itself only reports MAE, ROC-AUC and Brier skill.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_auc_score

FORECASTS_PATH = "processing_data/floods/forecasts_rf.csv" # Change to the XGBoost one or RF one
DEFAULT_THRESHOLD = 0.5
TARGET_FNR = 0.10  # find the threshold that gets false negative rate down to about this level

THRESHOLD_GRID = np.unique(np.concatenate([
    np.round(np.arange(0.005, 0.05, 0.005), 3),
    np.round(np.arange(0.05, 1.00, 0.05), 3),
]))


def metrics_for(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": len(y_true),
        "positive_rate": y_true.mean(),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "fnr": fn / (fn + tp) if (fn + tp) else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def scan_thresholds(y_true, y_prob):
    return pd.DataFrame([metrics_for(y_true, y_prob, t) for t in THRESHOLD_GRID])


def best_threshold_for_target_fnr(scan, target_fnr=TARGET_FNR):
    """Pick the highest threshold whose FNR is still <= target_fnr (highest = fewest false
    alarms while still hitting the target); if none qualifies, pick the lowest FNR available."""
    ok = scan[scan["fnr"] <= target_fnr]
    if len(ok):
        row = ok.sort_values("threshold", ascending=False).iloc[0]
        row["hit_target"] = True
        return row
    row = scan.sort_values("fnr").iloc[0]
    row["hit_target"] = False
    return row


def main():
    preds = pd.read_csv(FORECASTS_PATH).rename(columns={"p_rf": "p_flood"})

    print(f"Default threshold ({DEFAULT_THRESHOLD}) vs climatology baseline:\n")
    rows = []
    for h, g in preds.groupby("lead"):
        m = metrics_for(g["flood"], g["p_flood"], DEFAULT_THRESHOLD)
        m["lead"], m["baseline"] = h, "model"
        rows.append(m)
        mc = metrics_for(g["flood"], g["p_clim"], DEFAULT_THRESHOLD)
        mc["lead"], mc["baseline"] = h, "climatology"
        rows.append(mc)
    base = pd.DataFrame(rows).sort_values(["lead", "baseline"])
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(base[["lead", "baseline", "n", "positive_rate", "accuracy", "precision", "fpr", "fnr", "roc_auc"]]
          .to_string(index=False))

    print(f"\n--- Threshold scan per lead time (target: false negative rate <= {TARGET_FNR}) ---\n")
    best_rows = []
    for h, g in preds.groupby("lead"):
        scan = scan_thresholds(g["flood"], g["p_flood"])
        best = best_threshold_for_target_fnr(scan)
        best["lead"] = h
        best_rows.append(best)

        print(f"Lead {h} month(s) - full scan:")
        print(scan[["threshold", "accuracy", "precision", "fpr", "fnr"]].to_string(index=False))
        status = "hit target" if best["hit_target"] else "TARGET NOT REACHABLE - closest available shown"
        print(f"-> chosen threshold: {best['threshold']}  ({status})  "
              f"accuracy={best['accuracy']:.3f}, precision={best['precision']:.3f}, "
              f"fpr={best['fpr']:.3f}, fnr={best['fnr']:.3f}\n")

    summary = pd.DataFrame(best_rows)[
        ["lead", "threshold", "hit_target", "n", "positive_rate", "accuracy", "precision", "fpr", "fnr", "roc_auc"]
    ]
    print("--- Summary: best threshold per lead time ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()