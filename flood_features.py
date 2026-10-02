import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from processing_data.loading_impact_data import load_admin_boundaries
import os


era5_dir = "raw_data/rainfall and runoff"
years = range(2000, 2026)

admin1, admin2 = load_admin_boundaries()
target = pd.read_csv("processing_data/floods/county_month.csv")

res = 0.025
lat_top, lat_bot = 12.3, 3.4
lon_left, lon_right = 24.0, 36.0
nrows = int(round((lat_top - lat_bot) / res))
ncols = int(round((lon_right - lon_left) / res))
transform = from_origin(lon_left, lat_top, res, res)
shapes = [(geom, i + 1) for i, geom in enumerate(admin2.geometry)]
county_raster = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, dtype="int16")

rr, cc = np.where(county_raster > 0)
pts = pd.DataFrame({
    "county_id": county_raster[rr, cc],
    "lat": lat_top - (rr + 0.5) * res,
    "lon": lon_left + (cc + 0.5) * res,
})
pts["era_lat"] = (pts["lat"] * 4).round() / 4
pts["era_lon"] = (pts["lon"] * 4).round() / 4
weights = pts.groupby(["county_id", "era_lat", "era_lon"]).size().rename("n").reset_index()
weights["w"] = weights["n"] / weights.groupby("county_id")["n"].transform("sum")

upstream = {"lat_min": -3.0, "lat_max": 2.0, "lon_min": 29.0, "lon_max": 36.0}

monthly = []
up = []
for year in years:
    path = f"{era5_dir}/ERA5_{year}.nc"
    if not os.path.exists(path):
        print("missing", path)
        continue
    ds = xr.open_dataset(path)[["tp", "ro"]]
    ds = ds.resample(valid_time="1MS").sum()
    df = ds.to_dataframe().reset_index()
    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    df = df.rename(columns={"latitude": "era_lat", "longitude": "era_lon"})
    df["era_lat"] = df["era_lat"].round(2)
    df["era_lon"] = df["era_lon"].round(2)

    m = df.merge(weights, on=["era_lat", "era_lon"])
    m["tp"] = m["tp"] * m["w"]
    m["ro"] = m["ro"] * m["w"]
    monthly.append(m.groupby(["county_id", "year", "month"])[["tp", "ro"]].sum().reset_index())

    box = df[(df["era_lat"].between(upstream["lat_min"], upstream["lat_max"])) &
             (df["era_lon"].between(upstream["lon_min"], upstream["lon_max"]))]
    up.append(box.groupby(["year", "month"])[["tp", "ro"]].mean().rename(
        columns={"tp": "tp_upstream", "ro": "ro_upstream"}).reset_index())
    print(year, "done")

monthly = pd.concat(monthly, ignore_index=True)
up = pd.concat(up, ignore_index=True)

for c in ["tp", "ro"]:
    monthly[c] = monthly[c] * 1000
for c in ["tp_upstream", "ro_upstream"]:
    up[c] = up[c] * 1000

counties = admin2[["adm2_pcode"]].copy()
counties["county_id"] = np.arange(1, len(admin2) + 1)
counties["lat"] = admin2.geometry.centroid.y
counties["lon"] = admin2.geometry.centroid.x
monthly = monthly.merge(counties, on="county_id").drop(columns="county_id")

data = target.merge(monthly, on=["adm2_pcode", "year", "month"], how="inner")
data = data.merge(up, on=["year", "month"], how="left")
data = data.sort_values(["adm2_pcode", "year", "month"]).reset_index(drop=True)

g = data.groupby("adm2_pcode")
for c in ["tp", "ro", "tp_upstream", "ro_upstream"]:
    for lag in [1, 2, 3]:
        data[f"{c}_lag{lag}"] = g[c].shift(lag)
    data[f"{c}_sum3"] = g[c].transform(lambda s: s.rolling(3).sum())
    data[f"{c}_sum6"] = g[c].transform(lambda s: s.rolling(6).sum())

data["frac_lag1"] = g["frac_flooded"].shift(1)
data["frac_lag2"] = g["frac_flooded"].shift(2)
data["frac_lag12"] = g["frac_flooded"].shift(12)

missions = ("TOPEX", "POSDN", "JASN1", "JASN2", "JASN3", "SEN6A")
lakes = []
for name in ["victoria", "Kyoga"]:
    rows = []
    with open(f"raw_data/Water levels lakes/water_level_{name}.txt", encoding="latin-1") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] in missions:
                rows.append((parts[2], float(parts[14])))
    df = pd.DataFrame(rows, columns=["date", "level"])
    df = df[df["level"] < 9000]  # 9999.99 means missing
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    lakes.append(df.set_index("date")["level"].rename(f"lake_{name.lower()}"))
albert = xr.open_dataset("raw_data/Water levels lakes/water_level_altimetry_Albert.nc").to_dataframe()
albert["datetime"] = pd.to_datetime(albert["datetime"])
lakes.append(albert.set_index("datetime")["water_level"].rename("lake_albert"))

q = pd.read_csv("raw_data/Darthmouth Flood Observatory/100205_discharge.csv", parse_dates=["Date"])
lakes.append(q.set_index("Date")["Discharge (m3/s)"].rename("discharge"))

river = pd.concat([s.resample("MS").mean() for s in lakes], axis=1).ffill(limit=2)
river = river.loc["1999":]
for c in list(river.columns):
    river[f"{c}_chg1"] = river[c].diff(1)
    river[f"{c}_chg3"] = river[c].diff(3)
    river[f"{c}_chg12"] = river[c].diff(12)
river["year"] = river.index.year
river["month"] = river.index.month
data = data.merge(river, on=["year", "month"], how="left")

crop = pd.read_csv("processing_data/crops/crop_by_county.csv")[["adm2_pcode", "crop_pct", "range_pct"]]
data = data.merge(crop, on="adm2_pcode", how="left")

data = data.dropna(subset=[c for c in data.columns if c not in river.columns])
data.to_csv("processing_data/floods/model_table.csv", index=False)
print(len(data), "rows,", data["year"].min(), "-", data["year"].max())
print(data.columns.tolist())
