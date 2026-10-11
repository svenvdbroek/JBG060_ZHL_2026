import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from processing_data.loading_impact_data import load_admin_boundaries, load_ipc_data
import os


season = [7, 8, 9, 10, 11, 12]
map_year = 2025

files = {2020: ("dtm-south-sudan-event-tracking-jan-dec-2020.xlsx", "ET Jan-Dec-2020"),
         2021: ("iom_dtm_ssd_eventtracking_dataset_jan_dec_2021_hdx.xlsx", "ET_Dataset_Private"),
         2022: ("iom_dtm_ssd_eventtracking_dataset_jan_dec-2022_0_hdx.xlsx", "DTMEvent Tracking_Jan-Dec 2022"),
         2023: ("iom_dtm_ssd_eventtracking_dataset_january_december-2023_hdx.xlsx", "DTMEvent Tracking_Jan-Dec 2023"),
         2024: ("iom-ssd-dtm_event-tracking_jan-dec_2024_public_hdx.xlsx", "DTM_Event Tracking_JAN-DEC_2024"),
         2025: ("IOM_DTM_SSD_EventTracking_Dataset_January- December 2025_20260121.xlsx", "DTM_Event Tracking_Dataset")}
years = list(files)

admin1, admin2 = load_admin_boundaries()

iom = []
for year, (f, sheet) in files.items():
    d = pd.read_excel("processing_data/" + f, sheet_name=sheet)
    if year == 2020:
        d["Period start"] = d["Period start"].fillna(d["Date of assessment "])
        d = d.rename(columns={"IDPs_Trigger": "trigger", "Period start": "start",
                              "ADM2_PCODE (Destination)": "adm2_pcode", "Affected population: IDP (ind) ": "n"})
        d = d[d["trigger"].astype(str).str.lower().str.contains("natural")].copy()
    else:
        d = d.rename(columns={"Movement Trigger": "trigger", "Event Started On (From date)": "start",
                              "Event Location:  County PCODE": "adm2_pcode", "No. of Individuals": "n"})
        d = d[d["trigger"].astype(str).str.contains("Flood")].copy()
    d["n"] = pd.to_numeric(d["n"], errors="coerce")
    d["start"] = pd.to_datetime(d["start"], errors="coerce")
    d = d[(d["start"].dt.year == year) & d["start"].dt.month.isin(season)]
    iom.append(d.groupby("adm2_pcode")["n"].sum().reset_index().assign(year=year))
iom = pd.concat(iom).rename(columns={"n": "displaced_iom"})
print("people displaced by floods per year (iom, jul-dec):")
print(iom.groupby("year")["displaced_iom"].sum().astype(int).to_string())

e = pd.read_csv("processing_data/floods/exposure.csv")
e = e[e["year"].isin(years) & e["month"].isin(season)]
df = e.groupby(["adm2_pcode", "adm2_name", "adm1_name", "year"]).agg(
    coverage=("coverage", "first"), people=("people_total", "first"), cattle=("cattle_total", "first"),
    flood_actual=("frac_flooded", "max")).reset_index()

fc = pd.read_csv("processing_data/floods/impact_forecasts.csv")
fc = fc[fc["target_year"].isin(years) & fc["target_month"].isin(season)]
fc = fc.groupby(["adm2_pcode", "target_year", "lead", "model"]).agg(
    flood_pred=("pred", "max"), p_flood=("p_flood", "max"))
fc = fc.reset_index().rename(columns={"target_year": "year"})

ipc = load_ipc_data()
ipc = ipc[ipc["Area_level"] == "County"].copy()
ipc["Area"] = ipc["Area"].replace({"Wau (Rural Only)": "Wau", "Yei county": "Yei"}).str.lower()
ipc["ipc_year"] = ipc["Start Date"].dt.year
ipc = ipc.sort_values("Start Date").groupby(["Area", "ipc_year"]).tail(1)
ipc = ipc.rename(columns={"Phase 3+ Share": "ipc3_share"})[["Area", "ipc_year", "ipc3_share"]]

df = df.merge(iom, on=["adm2_pcode", "year"], how="left")
df["displaced_iom"] = df["displaced_iom"].fillna(0)
df["Area"] = df["adm2_name"].str.lower()
df["ipc_year"] = df["year"].clip(lower=2022)
df = df.merge(ipc, on=["Area", "ipc_year"], how="left").drop(columns=["Area", "ipc_year"])
df = df[df["coverage"] >= 0.9]

rates = {}
for y in years:
    other = df[df["year"] != y]
    rates[y] = other["displaced_iom"].sum() / (other["people"] * other["flood_actual"]).sum()
print("rate fitted on all years:", round(df["displaced_iom"].sum() / (df["people"] * df["flood_actual"]).sum(), 2))

df = df.merge(fc, on=["adm2_pcode", "year"])
df["displaced_forecast"] = df["year"].map(rates) * df["people"] * df["flood_pred"]

results = []
for (h, y), g in df.groupby(["lead", "year"]):
    own_rate = g["displaced_iom"].sum() / (g["people"] * g["flood_actual"]).sum()
    top = g.nlargest(10, "displaced_forecast")
    results.append({"lead": h, "model": g["model"].iloc[0], "year": y, "iom": int(g["displaced_iom"].sum()),
                    "forecast": int(g["displaced_forecast"].sum()),
                    "rate_used": round(rates[y], 2), "rate_that_year": round(own_rate, 2),
                    "rank_corr": round(g["displaced_forecast"].corr(g["displaced_iom"], method="spearman"), 3),
                    "auc": round(roc_auc_score(g["displaced_iom"] > 0, g["displaced_forecast"]), 3),
                    "top10_hits": int((top["displaced_iom"] > 0).sum())})
results = pd.DataFrame(results)
print("each year forecast with a rate fitted on the other years only:")
print(results.to_string(index=False))

df["cattle_forecast"] = df["cattle"] * df["flood_pred"]
df["food_insecure_displaced"] = df["displaced_forecast"] * df["ipc3_share"]

cols = ["adm2_name", "adm1_name", "lead", "model", "p_flood", "flood_pred", "flood_actual", "people",
        "displaced_forecast", "displaced_iom", "cattle_forecast", "ipc3_share", "food_insecure_displaced"]
df = df.sort_values(["lead", "year", "food_insecure_displaced"], ascending=[True, True, False])
for h, g in df[df["year"] == map_year].groupby("lead"):
    print(f"top 15 counties in {map_year} by food insecure people displaced, {g['model'].iloc[0]} {h} month ahead:")
    print(g[cols].head(15).round(3).to_string(index=False))
df[["adm2_pcode", "year"] + cols].to_csv("processing_data/floods/displacement.csv", index=False)

os.makedirs("images", exist_ok=True)
for h, one in df[df["year"] == map_year].groupby("lead"):
    model = one["model"].iloc[0]
    m = admin2[["adm2_pcode", "geometry"]].merge(one, on="adm2_pcode", how="left")
    vmax = max(m["displaced_forecast"].max(), m["displaced_iom"].max()) / 1000
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    panels = [("displaced_forecast", "forecast: people displaced", "Reds", vmax),
              ("displaced_iom", "iom: people displaced by floods", "Reds", vmax),
              ("cattle_forecast", "forecast: cattle in flooded area", "Oranges", None),
              ("food_insecure_displaced", "forecast: displaced and IPC 3+", "Purples", None)]
    for ax, (col, title, cmap, vm) in zip(axes.ravel(), panels):
        m.assign(k=m[col] / 1000).plot(column="k", cmap=cmap, vmin=0, vmax=vm, legend=True, ax=ax, edgecolor="gray",
                                       linewidth=0.3, missing_kwds={"color": "lightgray"},
                                       legend_kwds={"label": "thousands", "shrink": 0.6})
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"flood season jul-dec {map_year}, {model} forecast {h} month ahead (gray = no flood data)")
    plt.tight_layout()
    plt.savefig(f"images/displacement_map_lead{h}.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(one["displaced_forecast"] + 1, one["displaced_iom"] + 1)
    for _, r in one.nlargest(8, "displaced_iom").iterrows():
        ax.annotate(r["adm2_name"], (r["displaced_forecast"] + 1, r["displaced_iom"] + 1), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forecast people displaced (+1)")
    ax.set_ylabel("iom people displaced by floods (+1)")
    ax.set_title(f"per county, flood season {map_year}, {model} {h} month ahead")
    ax.grid(alpha=0.3)
    plt.savefig(f"images/displacement_vs_iom_lead{h}.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8, 5))
first = results[results["lead"] == 1]
ax.plot(first["year"], first["iom"] / 1000, marker="o", label="iom")
for (h, model), g in results.groupby(["lead", "model"]):
    ax.plot(g["year"], g["forecast"] / 1000, marker="o", label=f"forecast {h} month ahead ({model})")
ax.set_xlabel("year")
ax.set_ylabel("people displaced by floods, jul-dec (thousands)")
ax.set_title("whole country per year, rate fitted on the other years")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()
plt.savefig("images/displacement_years.png", dpi=150, bbox_inches="tight")
