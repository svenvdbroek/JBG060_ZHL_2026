import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.transform import from_origin
from processing_data.loading_impact_data import load_admin_boundaries
import os
import time


admin1, admin2 = load_admin_boundaries()
shapes = [(geom, i + 1) for i, geom in enumerate(admin2.geometry)]
n = len(admin2)

res = 1 / 480
lat_top, lat_bot = 10.0, 3.4
lon_left, lon_right = 24.0, 36.0
nrows = int(round((lat_top - lat_bot) / res))
ncols = int(round((lon_right - lon_left) / res))
county_raster = rasterize(shapes, out_shape=(nrows, ncols), transform=from_origin(lon_left, lat_top, res, res),
                          dtype="int16")


def load_pop(year, block=5):
    with rasterio.open(f"raw_data/worldpop/ssd_pop_{year}_CN_100m_R2025A_v1.tif") as src:
        h, w = src.height // block * block, src.width // block * block
        pop = np.zeros((h // block, w // block))
        step = block * 200
        for r0 in range(0, h, step):
            k = min(step, h - r0)
            a = src.read(1, window=Window(0, r0, w, k))
            a[a < 0] = 0
            pop[r0 // block:(r0 + k) // block] = a.reshape(k // block, block, w // block, block).sum(axis=(1, 3))
        return pop, src.transform.c, src.transform.f, src.res[0] * block


def value_at(grid, r, c):
    g, x0, y0, gres = grid
    gr = ((y0 - (lat_top - (r + 0.5) * res)) / gres).astype(np.int64)
    gc = ((lon_left + (c + 0.5) * res - x0) / gres).astype(np.int64)
    ok = (gr >= 0) & (gr < g.shape[0]) & (gc >= 0) & (gc < g.shape[1])
    out = np.zeros(len(r))
    out[ok] = g[gr[ok], gc[ok]] * (res / gres) ** 2
    return out


pops = {2020: load_pop(2020), 2025: load_pop(2025)}
with rasterio.open("raw_data/farmland/geonode__cattle_gha.tif") as src:
    cattle = src.read(1)
    cattle[cattle < 0] = 0
    cattle = (cattle, src.transform.c, src.transform.f, src.res[0])

totals = {"people_2020": np.zeros(n + 1), "people_2025": np.zeros(n + 1), "cattle": np.zeros(n + 1)}
for r0 in range(0, nrows, 400):
    rr, cc = np.mgrid[r0:min(r0 + 400, nrows), 0:ncols]
    rr, cc = rr.ravel(), cc.ravel()
    ids = county_raster[rr, cc]
    for name, grid in [("people_2020", pops[2020]), ("people_2025", pops[2025]), ("cattle", cattle)]:
        totals[name] += np.bincount(ids, weights=value_at(grid, rr, cc), minlength=n + 1)

counties = admin2[["adm2_pcode"]].copy()
counties["county_id"] = np.arange(1, n + 1)
for name in totals:
    counties[name] = totals[name][1:]
print("people in flood map area (2025):", round(counties["people_2025"].sum() / 1e6, 1), "million")
print("cattle in flood map area:", round(counties["cattle"].sum() / 1e6, 1), "million")

out = []
t0 = time.time()
for year in range(2012, 2026):
    pop = pops[2020] if year <= 2022 else pops[2025]
    parts = []
    for kind in ["recurring", "unusual"]:
        for tile in ["h20v08", "h21v08"]:
            df = pd.read_parquet(f"raw_data/flood_masks/compact_{kind}/flood_events_{tile}_{year}.parquet",
                                 columns=["date", "lat", "lon"])
            r = np.floor((lat_top - df["lat"].values) / res).astype(np.int64)
            c = np.floor((df["lon"].values - lon_left) / res).astype(np.int64)
            months = pd.to_datetime(df["date"].cat.categories).month.values[df["date"].cat.codes.values]
            inside = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
            parts.append(pd.DataFrame({"r": r[inside], "c": c[inside], "month": months[inside]}))
    px = pd.concat(parts, ignore_index=True).drop_duplicates()
    r, c = px["r"].values, px["c"].values
    px["county_id"] = county_raster[r, c]
    px["people_exposed"] = value_at(pop, r, c)
    px["cattle_exposed"] = value_at(cattle, r, c)
    px = px[px["county_id"] > 0]
    g = px.groupby(["county_id", "month"])[["people_exposed", "cattle_exposed"]].sum().reset_index()
    g["year"] = year
    out.append(g)
    print(year, "done,", round(time.time() - t0), "s")

out = pd.concat(out, ignore_index=True).merge(counties[["county_id", "adm2_pcode"]], on="county_id")
cm = pd.read_csv("processing_data/floods/county_month.csv")
cm = cm[cm["year"] >= 2012][["adm2_pcode", "adm2_name", "adm1_name", "year", "month", "coverage", "frac_flooded"]]
cm = cm.merge(out.drop(columns="county_id"), on=["adm2_pcode", "year", "month"], how="left").fillna(0)
cm = cm.merge(counties.drop(columns="county_id"), on="adm2_pcode")
cm["people_total"] = np.where(cm["year"] <= 2022, cm["people_2020"], cm["people_2025"])
cm = cm.drop(columns=["people_2020", "people_2025"]).rename(columns={"cattle": "cattle_total"})
cm.to_csv("processing_data/floods/exposure.csv", index=False)

print("share of people in flooded pixels vs share of land flooded (all county-months):")
print("people:", round(cm["people_exposed"].sum() / (cm["frac_flooded"] * cm["people_total"]).sum(), 3))
print("cattle:", round(cm["cattle_exposed"].sum() / (cm["frac_flooded"] * cm["cattle_total"]).sum(), 3))
