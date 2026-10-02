import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
import os

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

reg_params = dict(objective="reg:absoluteerror", eta=0.05, max_depth=4, min_child_weight=50,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
clf_params = dict(objective="binary:logistic", eval_metric="logloss", eta=0.05, max_depth=4,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)


def fit(params, train, valid, y):
    return xgb.train(params, xgb.DMatrix(train[features], train[y]), num_boost_round=2000,
                     evals=[(xgb.DMatrix(valid[features], valid[y]), "valid")],
                     early_stopping_rounds=100, verbose_eval=False)


def predict(model, df, **kwargs):
    return model.predict(xgb.DMatrix(df[features]), iteration_range=(0, model.best_iteration + 1), **kwargs)


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

        m_reg = fit(reg_params, train, valid, "change")
        m_clf = fit(clf_params, train, valid, "flood")

        test["pred"] = (test["frac_flooded"] + predict(m_reg, test)).clip(0, 1)
        test["p_flood"] = predict(m_clf, test)

        seen = pd.concat([train, valid])
        clim = seen.groupby(["adm2_pcode", "target_month"])[["target", "flood"]].mean()
        key = test.set_index(["adm2_pcode", "target_month"]).index
        test["persistence"] = test["frac_flooded"]
        test["climatology"] = key.map(clim["target"]).fillna(0)
        test["p_clim"] = key.map(clim["flood"]).fillna(seen["flood"].mean())
        test["lead"] = h
        preds.append(test[["adm2_pcode", "adm2_name", "adm1_name", "target_year", "target_month", "lead",
                           "target", "flood", "pred", "p_flood", "persistence", "climatology", "p_clim"]])
        last_models[h] = (m_reg, m_clf, test)
    print("lead", h, "done")

preds = pd.concat(preds, ignore_index=True)
os.makedirs("processing_data/floods", exist_ok=True)
preds.to_csv("processing_data/floods/forecasts.csv", index=False)

rows = []
for (h, year), g in preds.groupby(["lead", "target_year"]):
    rows.append({"lead": h, "year": year,
                 "xgboost": mean_absolute_error(g["target"], g["pred"]),
                 "persistence": mean_absolute_error(g["target"], g["persistence"]),
                 "climatology": mean_absolute_error(g["target"], g["climatology"])})
mae = pd.DataFrame(rows)
mae.to_csv("processing_data/floods/forecast_results.csv", index=False)

mae["period"] = np.where(mae["year"] < 2020, "2012-2019", "2020-2025")
print("\nMAE of fraction flooded, x1000 (lower is better):")
print((mae.groupby(["period", "lead"])[["xgboost", "persistence", "climatology"]].mean() * 1000).round(2).to_string())

preds["period"] = np.where(preds["target_year"] < 2020, "2012-2019", "2020-2025")
print(f"\nprobability of more than {threshold:.0%} flooded")
for (period, h), g in pd.concat([preds, preds.assign(period="all years")]).groupby(["period", "lead"]):
    skill = 1 - brier_score_loss(g["flood"], g["p_flood"]) / brier_score_loss(g["flood"], g["p_clim"])
    print(period, "lead", h, "| floods:", g["flood"].sum(), "| auc:", round(roc_auc_score(g["flood"], g["p_flood"]), 3),
          "| skill:", round(skill, 3))

m_reg, m_clf, test = last_models[1]
shap_reg = pd.DataFrame(predict(m_reg, test, pred_contribs=True)[:, :-1], columns=features).abs().mean()
shap_clf = pd.DataFrame(predict(m_clf, test, pred_contribs=True)[:, :-1], columns=features).abs().mean()
print("\ntop features amount model:\n", (shap_reg / shap_reg.sum() * 100).sort_values(ascending=False).head(10).round(1).to_string())
print("\ntop features probability model:\n", (shap_clf / shap_clf.sum() * 100).sort_values(ascending=False).head(10).round(1).to_string())

names = {"xgboost": "xgboost", "persistence": "same as this month", "climatology": "average for that month"}
os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
one = mae[mae["lead"] == 1].set_index("year")
for c in ["xgboost", "persistence", "climatology"]:
    ax.plot(one.index, one[c] * 100, marker="o", label=names[c])
ax.set_xlabel("year we predict")
ax.set_ylabel("average error (% of county)")
ax.set_title("Forecast 1 month ahead, error per year")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()
plt.savefig("images/forecast_mae.png", dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, period in zip(axes, ["2012-2019", "2020-2025"]):
    p = mae[mae["period"] == period].groupby("lead")[["xgboost", "persistence", "climatology"]].mean()
    for c in p.columns:
        ax.plot(p.index, p[c] * 100, marker="o", label=names[c])
    ax.set_xticks(horizons)
    ax.set_xlabel("months ahead")
    ax.set_ylabel("average error (% of county)")
    ax.set_title(period)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
axes[0].legend()
fig.suptitle("How the error grows when we forecast further ahead")
plt.tight_layout()
plt.savefig("images/forecast_lead.png", dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, s, title in [(axes[0], shap_reg, "amount model"), (axes[1], shap_clf, "probability model")]:
    top = (s / s.sum() * 100).sort_values().tail(12)
    ax.barh(top.index, top.values)
    ax.set_xlabel("share of the explanation (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
fig.suptitle("What drives the 1 month forecast (shap values, 2025)")
plt.tight_layout()
plt.savefig("images/shap_drivers.png", dpi=150, bbox_inches="tight")
