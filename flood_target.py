import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from processing_data.loading_impact_data import load_admin_boundaries
import os
import time


admin1, admin2 = load_admin_boundaries()

res = 1 / 480
lat_top, lat_bot = 10.0, 3.4
lon_left, lon_right = 24.0, 36.0
nrows = int(round((lat_top - lat_bot) / res))
ncols = int(round((lon_right - lon_left) / res))

transform = from_origin(lon_left, lat_top, res, res)
shapes = [(geom, i + 1) for i, geom in enumerate(admin2.geometry)]
county_raster = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, dtype="int16")

pixel_km2 = (res * 111) ** 2
county_px = np.bincount(county_raster.ravel(), minlength=len(admin2) + 1)[1:]
coverage = county_px * pixel_km2 / admin2["area_sqkm"].values

counties = admin2[["adm2_name", "adm1_name", "adm2_pcode"]].copy()
counties["county_id"] = np.arange(1, len(admin2) + 1)
counties["pixels"] = county_px
counties["coverage"] = coverage.round(3)
print("counties with less than 90% covered by the flood tiles:")
print(counties[counties["coverage"] < 0.9][["adm2_name", "adm1_name", "coverage"]].to_string(index=False))

tiles = ["h20v08", "h21v08"]
years = range(2000, 2026)
out = []
t0 = time.time()

for year in years:
    rec = []
    unu = []
    for tile in tiles:
        for kind, store in [("recurring", rec), ("unusual", unu)]:
            path = f"raw_data/flood_masks/compact_{kind}/flood_events_{tile}_{year}.parquet"
            df = pd.read_parquet(path, columns=["date", "lat", "lon"])
            r = np.floor((lat_top - df["lat"].values) / res).astype(np.int32)
            c = np.floor((df["lon"].values - lon_left) / res).astype(np.int32)
            inside = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
            r, c = r[inside], c[inside]
            county = county_raster[r, c]
            keep = county > 0
            month = pd.to_datetime(df["date"].cat.categories)[df["date"].cat.codes.values[inside][keep]].month
            store.append(pd.DataFrame({
                "county_id": county[keep],
                "month": np.asarray(month, dtype=np.int8),
                "pixel": r[keep].astype(np.int64) * ncols + c[keep],
            }))

    rec = pd.concat(rec, ignore_index=True)
    unu = pd.concat(unu, ignore_index=True)

    days = pd.concat([rec, unu]).groupby(["county_id", "month"]).size().rename("pixel_days")

    rec = rec.drop_duplicates(["pixel", "month"])
    unu = unu.drop_duplicates(["pixel", "month"])
    both = pd.concat([rec, unu]).drop_duplicates(["pixel", "month"])

    flooded = both.groupby(["county_id", "month"]).size().rename("px_flooded")
    unusual = unu.groupby(["county_id", "month"]).size().rename("px_unusual")

    year_df = pd.concat([flooded, unusual, days], axis=1).fillna(0).astype(int).reset_index()
    year_df["year"] = year
    out.append(year_df)
    print(year, "done,", round(time.time() - t0), "s so far")

out = pd.concat(out, ignore_index=True)

full = pd.MultiIndex.from_product([counties["county_id"], years, range(1, 13)], names=["county_id", "year", "month"])
out = out.set_index(["county_id", "year", "month"]).reindex(full, fill_value=0).reset_index()
out = out.merge(counties, on="county_id")

out["frac_flooded"] = out["px_flooded"] / out["pixels"]
out["frac_unusual"] = out["px_unusual"] / out["pixels"]
out["flood_days"] = out["pixel_days"] / out["pixels"]

cols = ["adm2_pcode", "adm2_name", "adm1_name", "year", "month", "pixels", "coverage",
        "px_flooded", "px_unusual", "frac_flooded", "frac_unusual", "flood_days"]
out = out[cols].sort_values(["adm2_pcode", "year", "month"])

os.makedirs("processing_data/floods", exist_ok=True)
out.to_csv("processing_data/floods/county_month.csv", index=False)
print(len(out), "rows saved, total time", round(time.time() - t0), "s")

by_year = out.groupby("year")["frac_flooded"].mean()
print("\nmean fraction flooded per year (all counties, all months):")
print(by_year.round(4).to_string())
