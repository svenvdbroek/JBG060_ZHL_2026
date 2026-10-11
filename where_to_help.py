import pandas as pd
import matplotlib.pyplot as plt
from processing_data.loading_impact_data import load_admin_boundaries
import os


season = [7, 8, 9, 10, 11, 12]
map_year = 2025

fc = pd.read_csv("processing_data/floods/impact_forecasts.csv")
fc = fc[fc["target_month"].isin(season)].rename(columns={"target_year": "year"})
df = fc.groupby(["adm2_pcode", "adm2_name", "adm1_name", "year", "lead", "model"]).agg(
    p_flood=("p_flood", "max"), share_expected=("pred", "max"), share_actual=("target", "max")).reset_index()

e = pd.read_csv("processing_data/floods/exposure.csv")
e = e.groupby(["adm2_pcode", "year"])[["people_total", "cattle_total"]].first().reset_index()
df = df.merge(e, on=["adm2_pcode", "year"])

d = pd.read_csv("processing_data/floods/displacement.csv")
d = d[["adm2_pcode", "year", "ipc3_share", "displaced_iom"]].drop_duplicates(["adm2_pcode", "year"])
df = df.merge(d, on=["adm2_pcode", "year"])

df["people_hit"] = df["share_expected"] * df["people_total"]
df["cattle_hit"] = df["share_expected"] * df["cattle_total"]
df["priority"] = df["people_hit"] * df["ipc3_share"]

checks = {"forecast": "people_hit", "pop": "people_total"}
rows = []
for (h, y), g in df.groupby(["lead", "year"]):
    r = {"lead": h, "model": g["model"].iloc[0], "year": y, "iom_displaced": int(g["displaced_iom"].sum())}
    for name, col in checks.items():
        r[f"corr_{name}"] = round(g[col].corr(g["displaced_iom"], method="spearman"), 2)
    for name, col in checks.items():
        r[f"top10_{name}"] = int((g.nlargest(10, col)["displaced_iom"] > 0).sum())
    rows.append(r)
print("rank correlation with iom displacement, and how many of the top 10 counties really had displacement:")
print(pd.DataFrame(rows).to_string(index=False))

cols = ["adm2_name", "adm1_name", "lead", "model", "p_flood", "share_expected", "people_total", "people_hit",
        "ipc3_share", "priority", "cattle_hit", "displaced_iom"]
df = df.sort_values(["lead", "year", "priority"], ascending=[True, True, False])
df[["adm2_pcode", "year"] + cols].to_csv("processing_data/floods/where_to_help.csv", index=False)
for h, g in df[df["year"] == map_year].groupby("lead"):
    print(f"top 15 counties to help in {map_year}, {g['model'].iloc[0]} forecast {h} month ahead:")
    print(g[cols].head(15).round(3).to_string(index=False))

admin1, admin2 = load_admin_boundaries()
os.makedirs("images", exist_ok=True)
panels = [("p_flood", "chance of a flood", "Blues", ""),
          ("people_hit", "people expected in flood water", "Reds", "thousands"),
          ("priority", "priority: people in flood water and IPC 3+", "Purples", "thousands"),
          ("displaced_iom", "what happened: displaced by floods (iom)", "Greys", "thousands")]
for h, one in df[df["year"] == map_year].groupby("lead"):
    model = one["model"].iloc[0]
    m = admin2[["adm2_pcode", "geometry"]].merge(one, on="adm2_pcode", how="left")
    m["people_hit"] = m["people_hit"] / 1000
    m["priority"] = m["priority"] / 1000
    m["displaced_iom"] = m["displaced_iom"] / 1000
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    for ax, (col, title, cmap, label) in zip(axes.ravel(), panels):
        m.plot(column=col, cmap=cmap, vmax=m[col].quantile(0.97), legend=True, ax=ax, edgecolor="gray", linewidth=0.3,
               missing_kwds={"color": "lightgray"}, legend_kwds={"label": label, "shrink": 0.6})
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"where to help, flood season jul-dec {map_year}, {model} forecast {h} month ahead (gray = no data)")
    plt.tight_layout()
    plt.savefig(f"images/where_to_help_lead{h}.png", dpi=150, bbox_inches="tight")
