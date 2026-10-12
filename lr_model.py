import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import shap

# Same data, features, leads, threshold and expanding-window split as the XGBoost and RF scripts,
# so results can be compared directly. Logistic regression only does the probability part.

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


frac_cols = ["frac_flooded", "frac_unusual", "frac_lag1", "frac_lag2", "frac_lag12",
             "frac_max12", "frac_mean12", "frac_target_lastyear"]
features = ["frac_flooded", "frac_unusual", "flood_days",
            "frac_lag1", "frac_lag2", "frac_lag12",
            "frac_max12", "frac_mean12",
            "frac_target_lastyear", "lastyear_missing",
            "lat", "lon", "month_sin", "month_cos"]
print(len(features), "features")

Cs = [0.001, 0.01, 0.1, 1, 10]  # inverse regularisation strength, picked on the validation year


def X(df):
    out = df.copy()
    out["lastyear_missing"] = out["frac_target_lastyear"].isna().astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["target_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["target_month"] / 12)
    out = out[features].fillna(0)
    out[frac_cols] = np.sqrt(out[frac_cols])
    return out


def fit(train, valid):
    if train["flood"].nunique() < 2:  # training set contained only one class
        return float(train["flood"].iloc[0])
    best, best_ll = None, np.inf
    for C in Cs:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        m.fit(X(train), train["flood"])
        ll = log_loss(valid["flood"], m.predict_proba(X(valid))[:, 1], labels=[0, 1])
        if ll < best_ll:
            best, best_ll = m, ll
    return best


def proba(model, df):
    if isinstance(model, float):
        return np.full(len(df), model)
    return model.predict_proba(X(df))[:, 1]


preds = []
last_models = {}
chosen_C = []
for h in horizons:
    future = data[["adm2_pcode", "t", "frac_flooded"]].copy()
    future["t"] -= h
    df = data.merge(future.rename(columns={"frac_flooded": "target"}), on=["adm2_pcode", "t"])
    last_year = data[["adm2_pcode", "t", "frac_flooded"]].copy()
    last_year["t"] += 12 - h
    df = df.merge(last_year.rename(columns={"frac_flooded": "frac_target_lastyear"}), on=["adm2_pcode", "t"], how="left")
    df["target_year"] = (df["t"] + h) // 12
    df["target_month"] = (df["t"] + h) % 12 + 1
    df["flood"] = (df["target"] > threshold).astype(int)

    for test_year in test_years:
        train = df[df["target_year"] < test_year - 1]
        valid = df[df["target_year"] == test_year - 1]
        test = df[df["target_year"] == test_year].copy()

        m_clf = fit(train, valid)
        test["p_lr"] = proba(m_clf, test)
        if not isinstance(m_clf, float):
            chosen_C.append({"lead": h, "year": test_year, "C": m_clf[-1].C})

        # climatology baseline: identical to XGBoost and Random Forest (train + validation years)
        seen = pd.concat([train, valid])
        clim = seen.groupby(["adm2_pcode", "target_month"])[["target", "flood"]].mean()
        key = test.set_index(["adm2_pcode", "target_month"]).index
        test["p_clim"] = key.map(clim["flood"]).fillna(seen["flood"].mean())
        test["lead"] = h
        preds.append(test[["adm2_pcode", "adm2_name", "adm1_name", "target_year", "target_month", "lead",
                           "target", "flood", "p_lr", "p_clim"]])
        last_models[h] = (m_clf, train, test)
    print("lead", h, "done")

preds = pd.concat(preds, ignore_index=True)
os.makedirs("processing_data/floods", exist_ok=True)
preds.to_csv("processing_data/floods/forecasts_lr.csv", index=False)
preds["period"] = np.where(preds["target_year"] < 2020, "2012-2019", "2020-2025")


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


print(f"\nLR: probability of more than {threshold:.0%} flooded (auc higher is better, skill vs climatology)")
res_lr = prob_table(preds, {"lr": "p_lr"})
print(res_lr.to_string(index=False))
res_lr.to_csv("processing_data/floods/forecast_results_lr.csv", index=False)

print("\nChosen C (count of test years per lead):")
print(pd.DataFrame(chosen_C).groupby(["lead", "C"]).size().unstack(fill_value=0).to_string())

# LR vs RF vs XGBoost, on exactly the same rows
keys = ["adm2_pcode", "target_year", "target_month", "lead"]
rf_path = "processing_data/floods/forecasts_rf.csv"
xgb_path = "processing_data/floods/forecasts.csv"
if os.path.exists(rf_path) and os.path.exists(xgb_path):
    rf = pd.read_csv(rf_path)[keys + ["p_rf"]]
    xg = pd.read_csv(xgb_path)[keys + ["p_flood"]].rename(columns={"p_flood": "p_xgb"})
    both = preds.merge(rf, on=keys, how="inner").merge(xg, on=keys, how="inner")
    print(f"\nRows in comparison: {len(both)} (LR: {len(preds)}, RF: {len(rf)}, XGBoost: {len(xg)})")

    cols = {"lr": "p_lr", "rf": "p_rf", "xgboost": "p_xgb"}
    print("\nLR vs RF vs XGBoost, probability of flooding:")
    print(prob_table(both, cols).to_string(index=False))

    os.makedirs("images", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, period in zip(axes, ["2012-2019", "2020-2025"]):
        d = both[both["period"] == period]
        for name, c in cols.items():
            auc = [roc_auc_score(d[d["lead"] == h]["flood"], d[d["lead"] == h][c]) for h in horizons]
            ax.plot(horizons, auc, marker="o", label=name)
        ax.set_xticks(horizons)
        ax.set_xlabel("months ahead")
        ax.set_ylabel("AUC")
        ax.set_title(period)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Flood probability: logistic regression vs RF vs XGBoost")
    plt.tight_layout()
    plt.savefig("images/lr_vs_rf_xgb_auc.png", dpi=150, bbox_inches="tight")
    print("\nSaved images/lr_vs_rf_xgb_auc.png")
else:
    print("\nforecasts_rf.csv and/or forecasts.csv not found - run the RF and XGBoost scripts first for the comparison.")

# SHAP values for LR, computed on the standardised features
def lr_shap(h):
    m, train, test = last_models[h]
    if isinstance(m, float):
        return None, None
    scaler, clf = m[0], m[-1]
    Xs_train = pd.DataFrame(scaler.transform(X(train)), columns=features)
    Xs_test = pd.DataFrame(scaler.transform(X(test)), columns=features)
    sv = shap.LinearExplainer(clf, Xs_train).shap_values(Xs_test)  # log-odds contributions
    return pd.DataFrame(sv, columns=features, index=test.index), test


# standardised coefficients (lead 1, last test year); mainly for reference!
m = last_models[1][0]
if not isinstance(m, float):
    coef = pd.Series(m[-1].coef_[0], index=features)
    print("\nstandardised coefficients, lead 1, last test year (positive = higher flood probability):")
    print(coef.reindex(coef.abs().sort_values(ascending=False).index).round(3).to_string())

groups = {
    "flood history": ["frac_flooded", "frac_unusual", "flood_days", "frac_lag1", "frac_lag2",
                      "frac_lag12", "frac_max12", "frac_mean12", "frac_target_lastyear", "lastyear_missing"],
    "time of year": ["month_sin", "month_cos", "target_month"],
    "location": ["lat", "lon"],
}


def grouped_share(sv):
    """sv: DataFrame of signed SHAP values (rows x features). Returns % share per group."""
    tot = pd.DataFrame({name: sv[[c for c in cols if c in sv.columns]].sum(axis=1)
                        for name, cols in groups.items()})
    imp = tot.abs().mean()
    return (imp / imp.sum() * 100).round(1)


for h in [1, 6]:
    sv, test = lr_shap(h)
    if sv is None:
        continue
    print(f"\nLR SHAP share per group, lead {h}, 2025 (%):")
    print(grouped_share(sv).to_string())