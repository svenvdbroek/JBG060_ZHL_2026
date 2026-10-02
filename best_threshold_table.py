import numpy as np
import pandas as pd

BETA = 3  # recall counts BETA times as much as precision; raise it to punish missed floods harder
keys = ["adm2_pcode", "target_year", "target_month", "lead"]

rf = pd.read_csv("processing_data/floods/forecasts_rf.csv")[keys + ["flood", "p_rf"]]
xg = pd.read_csv("processing_data/floods/forecasts.csv")[keys + ["p_flood"]].rename(columns={"p_flood": "p_xgb"})
d = rf.merge(xg, on=keys)  # identical rows for both models, all test years pooled

grid = np.round(np.arange(0.001, 1.0, 0.001), 3)


def best_point(y, p):
    pred = p[:, None] >= grid[None, :]
    pos = (y == 1)[:, None]
    tp = (pred & pos).sum(0)
    fp = (pred & ~pos).sum(0)
    fn = (~pred & pos).sum(0)
    tn = (~pred & ~pos).sum(0)
    b2 = BETA ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)
        precision = tp / (tp + fp)
    i = int(np.nanargmax(fbeta))
    return {"threshold": grid[i],
            "accuracy": (tp[i] + tn[i]) / len(y),
            "precision": precision[i],
            "FPR": fp[i] / (fp[i] + tn[i]),
            "FNR": fn[i] / (fn[i] + tp[i]),
            f"F{BETA}": fbeta[i]}


for name, col in [("XGBoost", "p_xgb"), ("Random Forest", "p_rf")]:
    rows = []
    for h, s in d.groupby("lead"):
        rows.append({"lead": f"{h} month(s)", **best_point(s["flood"].values, s[col].values)})
    t = pd.DataFrame(rows)
    for c in ["accuracy", "precision", "FPR", "FNR", f"F{BETA}"]:
        t[c] = (t[c] * 100).round(1).astype(str) + "%"
    print(f"\n{name}: threshold maximising F{BETA} per lead")
    print(t.to_string(index=False))