import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.transform import from_origin
from processing_data.loading_impact_data import load_admin_boundaries
import os


admin1, admin2 = load_admin_boundaries()
shapes = [(geom, i + 1) for i, geom in enumerate(admin2.geometry)]
counties = admin2[["adm2_pcode"]].copy()

block = 12
with rasterio.open("raw_data/worldpop/ssd_pop_2020_CN_100m_R2025A_v1.tif") as src:
    h, w = src.height // block * block, src.width // block * block
    pop = np.zeros((h // block, w // block))
    step = block * 100
    for r0 in range(0, h, step):
        n = min(step, h - r0)
        a = src.read(1, window=Window(0, r0, w, n))
        a[a < 0] = 0
        pop[r0 // block:(r0 + n) // block] = a.reshape(n // block, block, w // block, block).sum(axis=(1, 3))
    transform = from_origin(src.transform.c, src.transform.f, src.res[0] * block, src.res[1] * block)
county_raster = rasterize(shapes, out_shape=pop.shape, transform=transform, dtype="int16")
counties["people"] = np.bincount(county_raster.ravel(), weights=pop.ravel(), minlength=len(admin2) + 1)[1:]
print("people in south sudan (worldpop 2020):", round(counties["people"].sum() / 1e6, 1), "million")

res = 0.025
lat_top, lon_left = 12.3, 24.0
grid = rasterize(shapes, out_shape=(356, 480), transform=from_origin(lon_left, lat_top, res, res), dtype="int16")
rr, cc = np.where(grid > 0)
with rasterio.open("raw_data/farmland/geonode__cattle_gha.tif") as src:
    cattle = src.read(1)
    cattle[cattle < 0] = 0
    cr, ccol = rasterio.transform.rowcol(src.transform, lon_left + (cc + 0.5) * res, lat_top - (rr + 0.5) * res)
    share = (res / src.res[0]) ** 2
counties["cattle"] = np.bincount(grid[rr, cc], weights=cattle[np.array(cr), np.array(ccol)] * share,
                                 minlength=len(admin2) + 1)[1:]
print("cattle:", round(counties["cattle"].sum() / 1e6, 1), "million")

crop = pd.read_csv("processing_data/crops/crop_by_county.csv")[["adm2_pcode", "crop_km2"]]
counties = counties.merge(crop, on="adm2_pcode")

fc = pd.read_csv("processing_data/floods/forecasts.csv")
fc = fc[fc["lead"] == 1].merge(counties, on="adm2_pcode")
for c in ["people", "cattle", "crop_km2"]:
    fc[f"{c}_forecast"] = fc["pred"] * fc[c]
    fc[f"{c}_actual"] = fc["target"] * fc[c]
fc.to_csv("processing_data/floods/impact_forecasts.csv", index=False)

monthly = fc.groupby(["target_year", "target_month"])[
    ["people_forecast", "people_actual", "crop_km2_forecast", "crop_km2_actual"]].sum()
monthly.index = pd.to_datetime(pd.DataFrame({"year": monthly.index.get_level_values(0),
                                             "month": monthly.index.get_level_values(1), "day": 1}))
print("\ncorrelation forecast vs actual people in flooded area (whole country per month):",
      round(monthly["people_forecast"].corr(monthly["people_actual"]), 3))

worst = fc.groupby(["target_year", "target_month"])["people_actual"].sum().idxmax()
month = fc[(fc["target_year"] == worst[0]) & (fc["target_month"] == worst[1])].copy()
month["people_diff"] = month["people_forecast"] - month["people_actual"]
print(f"\nworst month: {worst[1]}/{worst[0]}. top 10 counties by people in the forecast flood area:")
print(month.sort_values("people_forecast", ascending=False)[
    ["adm2_name", "adm1_name", "p_flood", "pred", "target", "people_forecast", "people_actual", "crop_km2_forecast"]
].head(10).round(3).to_string(index=False))

os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(monthly.index, monthly["people_actual"] / 1000, label="what happened")
ax.plot(monthly.index, monthly["people_forecast"] / 1000, label="forecast 1 month before")
ax.set_ylabel("people in flooded area (thousands)")
ax.set_title("People living in the flooded part of their county, whole country per month")
ax.grid(alpha=0.3)
ax.legend()
plt.savefig("images/impact_people.png", dpi=150, bbox_inches="tight")

m = admin2[["adm2_pcode", "geometry"]].merge(month, on="adm2_pcode", how="left")
vmax = max(m["people_forecast"].max(), m["people_actual"].max()) / 1000
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
m.plot(column="p_flood", cmap="Blues", vmin=0, vmax=1, legend=True, ax=axes[0], edgecolor="gray", linewidth=0.3,
       missing_kwds={"color": "lightgray"}, legend_kwds={"label": "chance of >2% of county flooded"})
axes[0].set_title("forecast: chance of a flood")
for ax, col, title in [(axes[1], "people_forecast", "forecast: people in flooded area"),
                       (axes[2], "people_actual", "what happened: people in flooded area")]:
    m.assign(k=m[col] / 1000).plot(column="k", cmap="Blues", vmin=0, vmax=vmax, legend=True, ax=ax,
                                   edgecolor="gray", linewidth=0.3, missing_kwds={"color": "lightgray"},
                                   legend_kwds={"label": "thousands of people"})
    ax.set_title(title)
for ax in axes:
    ax.set_axis_off()
fig.suptitle(f"{worst[1]}/{worst[0]}, forecast made 1 month before (gray = no flood data)")
plt.tight_layout()
plt.savefig("images/impact_map.png", dpi=150, bbox_inches="tight")
