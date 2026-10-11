import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from processing_data.loading_impact_data import load_admin_boundaries
import os


threshold = 0.02
models = {1: "lr", 3: "rf", 6: "rf"}

data = pd.read_csv("processing_data/floods/model_table.csv")
data = data[data["coverage"] > 0.5]
data = data.sort_values(["adm2_pcode", "year", "month"]).reset_index(drop=True)
data["t"] = data["year"] * 12 + data["month"] - 1

g = data.groupby("adm2_pcode")["frac_flooded"]
data["frac_max12"] = g.transform(lambda s: s.rolling(12, min_periods=1).max())
data["frac_mean12"] = g.transform(lambda s: s.rolling(12, min_periods=1).mean())

features_rf = ["frac_flooded", "frac_unusual", "flood_days",
               "frac_lag1", "frac_lag2", "frac_lag12",
               "frac_max12", "frac_mean12",
               "frac_target_lastyear",
               "lat", "lon", "target_month"]
features_lr = ["frac_flooded", "frac_unusual", "flood_days",
               "frac_lag1", "frac_lag2", "frac_lag12",
               "frac_max12", "frac_mean12",
               "frac_target_lastyear", "lastyear_missing",
               "lat", "lon", "month_sin", "month_cos"]
frac_cols = ["frac_flooded", "frac_unusual", "frac_lag1", "frac_lag2", "frac_lag12",
             "frac_max12", "frac_mean12", "frac_target_lastyear"]


def X_lr(df):
    out = df.copy()
    out["lastyear_missing"] = out["frac_target_lastyear"].isna().astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["target_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["target_month"] / 12)
    out = out[features_lr].fillna(0)
    out[frac_cols] = np.sqrt(out[frac_cols])
    return out


def fit_lr(known):
    last = known["target_year"].max()
    train, valid = known[known["target_year"] < last], known[known["target_year"] == last]
    best, best_ll = None, np.inf
    for C in [0.001, 0.01, 0.1, 1, 10]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X_lr(train), train["flood"])
        ll = log_loss(valid["flood"], m.predict_proba(X_lr(valid))[:, 1], labels=[0, 1])
        if ll < best_ll:
            best, best_ll = C, ll
    return make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=2000)).fit(X_lr(known), known["flood"])


t_now = data["t"].max()
print("latest flood data:", t_now // 12, "-", t_now % 12 + 1)
frac = data[["adm2_pcode", "t", "frac_flooded"]]

out = []
for h, model in models.items():
    future = frac.assign(t=frac["t"] - h).rename(columns={"frac_flooded": "target"})
    last_year = frac.assign(t=frac["t"] + 12 - h).rename(columns={"frac_flooded": "frac_target_lastyear"})
    df = data.merge(last_year, on=["adm2_pcode", "t"], how="left")
    df["target_year"] = (df["t"] + h) // 12
    df["target_month"] = (df["t"] + h) % 12 + 1

    known = df.merge(future, on=["adm2_pcode", "t"])
    known["flood"] = (known["target"] > threshold).astype(int)
    now = df[df["t"] == t_now].copy()
    if model == "lr":
        now["p_flood"] = fit_lr(known).predict_proba(X_lr(now))[:, 1]
    else:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=42)
        m.fit(known[features_rf].fillna(-1), known["flood"])
        now["p_flood"] = m.predict_proba(now[features_rf].fillna(-1))[:, 1]
    now["lead"] = h
    now["model"] = model
    out.append(now[["adm2_pcode", "target_year", "target_month", "lead", "model", "frac_flooded", "p_flood"]])
    print("lead", h, model, "done")
out = pd.concat(out, ignore_index=True)

size = data[data["frac_flooded"] > threshold].groupby("adm2_pcode")["frac_flooded"].mean().rename("flood_size")
out = out.merge(size, on="adm2_pcode", how="left")
out["flood_size"] = out["flood_size"].fillna(0)
out["pred"] = out["p_flood"] * out["flood_size"]

past = pd.read_csv("processing_data/floods/displacement.csv").drop_duplicates(["adm2_pcode", "year"])
rate = past["displaced_iom"].sum() / (past["people"] * past["flood_actual"]).sum()
past = past[past["year"] == 2025]
e = pd.read_csv("processing_data/floods/exposure.csv")
e = e[(e["year"] == 2025) & (e["month"] == 12)][["adm2_pcode", "adm2_name", "adm1_name", "coverage",
                                                  "people_total", "cattle_total"]]
out = out.merge(e, on="adm2_pcode").merge(past[["adm2_pcode", "ipc3_share"]], on="adm2_pcode", how="left")
out = out[out["coverage"] >= 0.9]

out["displaced_forecast"] = rate * out["people_total"] * out["pred"]
out["cattle_forecast"] = out["cattle_total"] * out["pred"]
out["food_insecure_displaced"] = out["displaced_forecast"] * out["ipc3_share"]
out.to_csv("processing_data/floods/forecast_next.csv", index=False)

print("whole country per month:")
print(out.groupby(["target_year", "target_month"])[["displaced_forecast", "cattle_forecast",
                                                    "food_insecure_displaced"]].sum().round(0).to_string())

peak = out.sort_values("pred").groupby("adm2_pcode").tail(1)
peak = peak.sort_values("food_insecure_displaced", ascending=False)
print("top 15 counties, worst of the forecast months:")
print(peak[["adm2_name", "adm1_name", "target_month", "model", "p_flood", "pred", "displaced_forecast",
            "cattle_forecast", "ipc3_share", "food_insecure_displaced"]].head(15).round(3).to_string(index=False))

admin1, admin2 = load_admin_boundaries()
os.makedirs("images", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmax = out["displaced_forecast"].max() / 1000
for ax, (h, model) in zip(axes, models.items()):
    one = out[out["lead"] == h]
    m = admin2[["adm2_pcode", "geometry"]].merge(one, on="adm2_pcode", how="left")
    m.assign(k=m["displaced_forecast"] / 1000).plot(column="k", cmap="Reds", vmin=0, vmax=vmax, legend=True, ax=ax,
                                                    edgecolor="gray", linewidth=0.3,
                                                    missing_kwds={"color": "lightgray"},
                                                    legend_kwds={"label": "thousands of people", "shrink": 0.6})
    ax.set_title(f"{one['target_month'].iloc[0]}/{one['target_year'].iloc[0]} ({model}, {h} month ahead)")
    ax.set_axis_off()
fig.suptitle("forecast people displaced by floods (gray = no flood data)")
plt.tight_layout()
plt.savefig("images/forecast_next.png", dpi=150, bbox_inches="tight")
