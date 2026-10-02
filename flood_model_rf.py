import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

# Same data, features, leads, thresholds and expanding-window split as the XGBoost script,
# so results can be compared directly.

data = pd.read_csv("processing_data/floods/model_table.csv")
data = data[data["coverage"] > 0.5]
data = data.sort_values(["adm2_pcode", "year", "month"]).reset_index(drop=True)
data["t"] = data["year"] * 12 + data["month"] - 1
print(data["adm2_pcode"].nunique(), "counties,", len(data), "rows")

g = data.groupby("adm2_pcode")["frac_flooded"]
data["frac_max12"] = g.transform(lambda s: s.rolling(12, min_periods=1).max())
data["frac_mean12"] = g.transform(lambda s: s.rolling(12, min_periods=1).mean())

threshold = 0.02
horizons = [1, 2, 3, 6]
test_years = range(2012, 2026)

features = ["frac_flooded", "frac_unusual", "flood_days",
            "frac_lag1", "frac_lag2", "frac_lag12",
            "frac_max12", "frac_mean12",
            "frac_target_lastyear",
            "lat", "lon", "target_month"]
print(len(features), "features")


reg_params = dict(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                  n_jobs=-1, random_state=42)
clf_params = dict(n_estimators=300, min_samples_leaf=10, max_features=0.5,
                  n_jobs=-1, random_state=42)  # no class_weight: it would distort the probabilities / Brier score


def X(df):
    return df[features].fillna(-1)


def proba(model, df):
    p = model.predict_proba(X(df))
    if p.shape[1] == 1:  # training set contained only one class
        return np.full(len(df), float(model.classes_[0]))
    return p[:, 1]


preds = []
last_models = {}
for h in horizons:
    future = data[["adm2_pcode", "t", "frac_flooded"]].copy()
    future["t"] -= h
    df = data.merge(future.rename(columns={"frac_flooded": "target"}), on=["adm2_pcode", "t"])
    last_year = data[["adm2_pcode", "t", "frac_flooded"]].copy()
    last_year["t"] += 12 - h
    df = df.merge(last_year.rename(columns={"frac_flooded": "frac_target_lastyear"}), on=["adm2_pcode", "t"], how="left")
    df["target_year"] = (df["t"] + h) // 12
    df["target_month"] = (df["t"] + h) % 12 + 1
    df["change"] = df["target"] - df["frac_flooded"]
    df["flood"] = (df["target"] > threshold).astype(int)

    for test_year in test_years:
        train = df[df["target_year"] < test_year - 1]
        valid = df[df["target_year"] == test_year - 1]
        test = df[df["target_year"] == test_year].copy()

        m_reg = RandomForestRegressor(**reg_params).fit(X(train), train["change"])
        m_clf = RandomForestClassifier(**clf_params).fit(X(train), train["flood"])

        test["pred_rf"] = (test["frac_flooded"] + m_reg.predict(X(test))).clip(0, 1)
        test["p_rf"] = proba(m_clf, test)

        # baselines: identical to the XGBoost script (climatology uses train + validation years)
        seen = pd.concat([train, valid])
        clim = seen.groupby(["adm2_pcode", "target_month"])[["target", "flood"]].mean()
        key = test.set_index(["adm2_pcode", "target_month"]).index
        test["persistence"] = test["frac_flooded"]
        test["climatology"] = key.map(clim["target"]).fillna(0)
        test["p_clim"] = key.map(clim["flood"]).fillna(seen["flood"].mean())
        test["lead"] = h
        preds.append(test[["adm2_pcode", "adm2_name", "adm1_name", "target_year", "target_month", "lead",
                           "target", "flood", "pred_rf", "p_rf",
                           "persistence", "climatology", "p_clim"]])
        last_models[h] = (m_reg, m_clf)
    print("lead", h, "done")

preds = pd.concat(preds, ignore_index=True)
os.makedirs("processing_data/floods", exist_ok=True)
preds.to_csv("processing_data/floods/forecasts_rf.csv", index=False)
preds["period"] = np.where(preds["target_year"] < 2020, "2012-2019", "2020-2025")


def mae_table(d, cols):
    rows = []
    for (h, year), gr in d.groupby(["lead", "target_year"]):
        r = {"lead": h, "year": year, "period": "2012-2019" if year < 2020 else "2020-2025"}
        r.update({name: mean_absolute_error(gr["target"], gr[c]) for name, c in cols.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def prob_table(d, cols):
    rows = []
    for (period, h), gr in pd.concat([d, d.assign(period="all years")]).groupby(["period", "lead"]):
        base = brier_score_loss(gr["flood"], gr["p_clim"])
        r = {"period": period, "lead": h, "floods": int(gr["flood"].sum())}
        for name, c in cols.items():
            r[f"auc_{name}"] = round(roc_auc_score(gr["flood"], gr[c]), 3)
            r[f"skill_{name}"] = round(1 - brier_score_loss(gr["flood"], gr[c]) / base, 3)
        rows.append(r)
    return pd.DataFrame(rows)


# RF on its own
mae_rf = mae_table(preds, {"rf": "pred_rf", "persistence": "persistence", "climatology": "climatology"})
mae_rf.drop(columns="period").to_csv("processing_data/floods/forecast_results_rf.csv", index=False)
print("\nRF: MAE of fraction flooded, x1000 (lower is better):")
print((mae_rf.groupby(["period", "lead"])[["rf", "persistence", "climatology"]].mean() * 1000).round(2).to_string())
print(f"\nRF: probability of more than {threshold:.0%} flooded")
print(prob_table(preds, {"rf": "p_rf"}).to_string(index=False))

# RF vs XGBoost, on exactly the same rows
xgb_path = "processing_data/floods/forecasts.csv"
if os.path.exists(xgb_path):
    keys = ["adm2_pcode", "target_year", "target_month", "lead"]
    x = pd.read_csv(xgb_path)[keys + ["pred", "p_flood"]].rename(columns={"pred": "pred_xgb", "p_flood": "p_xgb"})
    both = preds.merge(x, on=keys, how="inner")
    print(f"\nRows in comparison: {len(both)} (RF: {len(preds)}, XGBoost: {len(x)})")

    cols = {"rf": "pred_rf", "xgboost": "pred_xgb", "persistence": "persistence", "climatology": "climatology"}
    mae_both = mae_table(both, cols)
    print("\nMAE of fraction flooded, x1000 (lower is better):")
    print((mae_both.groupby(["period", "lead"])[list(cols)].mean() * 1000).round(2).to_string())

    print(f"\nProbability of more than {threshold:.0%} flooded (auc higher is better, skill vs climatology):")
    print(prob_table(both, {"rf": "p_rf", "xgboost": "p_xgb"}).to_string(index=False))

    os.makedirs("images", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, period in zip(axes, ["2012-2019", "2020-2025"]):
        p = mae_both[mae_both["period"] == period].groupby("lead")[list(cols)].mean()
        for c in p.columns:
            ax.plot(p.index, p[c] * 100, marker="o", label=c)
        ax.set_xticks(horizons)
        ax.set_xlabel("months ahead")
        ax.set_ylabel("average error (% of county)")
        ax.set_title(period)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Random Forest vs XGBoost: error by forecast horizon")
    plt.tight_layout()
    plt.savefig("images/rf_vs_xgb_lead.png", dpi=150, bbox_inches="tight")
    print("\nSaved images/rf_vs_xgb_lead.png")
else:
    print(f"\n{xgb_path} not found - run the XGBoost script first to get the side-by-side comparison.")

# RF feature importance (lead 1, impurity-based; not directly comparable to SHAP)
m_reg, m_clf = last_models[1]
for name, m in [("amount model", m_reg), ("probability model", m_clf)]:
    imp = pd.Series(m.feature_importances_, index=features)
    print(f"\ntop features {name} (RF, lead 1, last test year):\n",
          (imp / imp.sum() * 100).sort_values(ascending=False).head(10).round(1).to_string())