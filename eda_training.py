import traceback
from pathlib import Path
import numpy as np
import pandas as pd

YEARS = np.arange(2015, 2026)
RAW = Path("./raw_data")
PANEL = Path("./processed_data_rf/flood_panel.parquet")
MODEL_TABLE = Path("processing_data/floods/model_table.csv")
HORIZONS = [1, 2, 3, 6]
MODEL_FEATURES = ["frac_flooded", "frac_unusual", "flood_days",
                  "frac_lag1", "frac_lag2", "frac_lag12", "frac_max12", "frac_mean12",
                  "frac_target_lastyear", "lat", "lon", "target_month"]
FORECASTS = {
    "forecasts_xgboost": ("processing_data/floods/forecasts.csv",
                          ["target", "flood", "pred", "p_flood", "persistence", "climatology", "p_clim"]),
    "forecasts_rf": ("processing_data/floods/forecasts_rf.csv",
                     ["target", "flood", "pred_rf", "p_rf", "persistence", "climatology", "p_clim"]),
    "forecasts_lr": ("processing_data/floods/forecasts_lr.csv",
                     ["target", "flood", "p_lr", "p_clim"]),
}

detail = []   # rows: dataset, column, rows, missing
errors = []


def add(name, df, cols=None):
    cols = list(cols) if cols is not None else list(df.columns)
    miss = df[cols].isna().sum()
    for c in cols:
        detail.append({"dataset": name, "column": c, "rows": len(df), "missing": int(miss[c])})


def add_parts(name, parts):
    """Several frames that belong to one dataset (e.g. one per station): sum per column."""
    rows = {}
    for df in parts:
        for c in df.columns:
            r = rows.setdefault(c, [0, 0])
            r[0] += len(df)
            r[1] += int(df[c].isna().sum())
    for c, (n, m) in rows.items():
        detail.append({"dataset": name, "column": c, "rows": n, "missing": m})


def add_dataset_xr(name, ds):
    for v in ds.data_vars:
        detail.append({"dataset": name, "column": v, "rows": int(ds[v].size),
                       "missing": int(ds[v].isnull().sum().compute())})


def raster_counts(path):
    import rasterio
    total = miss = 0
    with rasterio.open(path) as src:
        total = src.width * src.height
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win, masked=True)
            miss += int(np.ma.getmaskarray(arr).sum() + np.isnan(arr.filled(0)).sum())
    return total, miss


def safe(label, fn):
    print(f"checking {label} ...", flush=True)
    try:
        fn()
    except Exception as e:  # keep going, report at the end
        errors.append({"dataset": label, "error": f"{type(e).__name__}: {e}"})
        traceback.print_exc(limit=1)


# raw datasets
def check_raw():
    from processing_data.loading import (
        load_dartmouth_data, load_lake_stations, load_rainfall_runoff, load_flood_masks,
    )
    from processing_data.loading_impact_data import load_admin_boundaries, load_health_facilities

    safe("flood_masks", lambda: add("flood_masks", load_flood_masks(YEARS)))
    safe("era5_rainfall_runoff", lambda: add_dataset_xr("era5_rainfall_runoff", load_rainfall_runoff(YEARS)))
    safe("river_discharge", lambda: add_parts("river_discharge", load_dartmouth_data().values()))

    def lakes():
        for lake, df in load_lake_stations().items():
            add(f"lake_{lake}", df)
    safe("lake_levels", lakes)

    def admin():
        a1, a2 = load_admin_boundaries()
        add("admin1_boundaries", a1)
        add("admin2_boundaries", a2)
    safe("admin_boundaries", admin)

    safe("health_facilities", lambda: add("health_facilities", load_health_facilities()))


def check_rasters():
    singles = {
        "raster_crop_mask": RAW / "farmland" / "asap_mask_crop_v04.tif",
        "raster_rangeland_mask": RAW / "farmland" / "asap_mask_rangeland_v04.tif",
        "raster_cattle": RAW / "farmland" / "geonode__cattle_gha.tif",
    }
    for name, path in singles.items():
        def one(name=name, path=path):
            total, miss = raster_counts(path)
            detail.append({"dataset": name, "column": "pixels", "rows": total, "missing": miss})
        safe(name, one)

    def pop():
        total = miss = 0
        for y in YEARS:
            t, m = raster_counts(RAW / "worldpop" / f"ssd_pop_{y}_CN_100m_R2025A_v1.tif")
            total += t
            miss += m
        detail.append({"dataset": "raster_worldpop_2015_2025", "column": "pixels",
                       "rows": total, "missing": miss})
    safe("raster_worldpop", pop)


def check_panel():
    if PANEL.exists():
        add("flood_panel (RF training data)", pd.read_parquet(PANEL))
    else:
        errors.append({"dataset": "flood_panel", "error": f"not found: {PANEL}"})


# model_table.csv
def check_model_table():
    raw = pd.read_csv(MODEL_TABLE)
    add("model_table: raw", raw)

    data = raw[raw["coverage"] > 0.5].sort_values(["adm2_pcode", "year", "month"]).reset_index(drop=True)
    data["t"] = data["year"] * 12 + data["month"] - 1
    add("model_table: after coverage>0.5", data)
    print(f"  model_table rows: raw={len(raw)}, after filter={len(data)}")

    g = data.groupby("adm2_pcode")["frac_flooded"]
    data["frac_max12"] = g.transform(lambda s: s.rolling(12, min_periods=1).max())
    data["frac_mean12"] = g.transform(lambda s: s.rolling(12, min_periods=1).mean())

    for h in HORIZONS:
        future = data[["adm2_pcode", "t", "frac_flooded"]].copy()
        future["t"] -= h
        df = data.merge(future.rename(columns={"frac_flooded": "target"}), on=["adm2_pcode", "t"])
        last_year = data[["adm2_pcode", "t", "frac_flooded"]].copy()
        last_year["t"] += 12 - h
        df = df.merge(last_year.rename(columns={"frac_flooded": "frac_target_lastyear"}),
                      on=["adm2_pcode", "t"], how="left")
        df["target_month"] = (df["t"] + h) % 12 + 1
        add(f"model_table: lead {h} (model features + target)", df, MODEL_FEATURES + ["target"])
        print(f"  lead {h}: {len(df)} usable rows, {len(data) - len(df)} dropped (no target month)")


# forecast outputs
def check_forecasts():
    keys = ["adm2_pcode", "target_year", "target_month", "lead"]
    frames = {}
    for name, (path, cols) in FORECASTS.items():
        if not Path(path).exists():
            errors.append({"dataset": name, "error": f"not found: {path} (run the model script first)"})
            continue
        df = pd.read_csv(path)
        frames[name] = df
        add(name, df, keys + cols)
    if len(frames) == len(FORECASTS):
        sets = {n: set(map(tuple, f[keys].values)) for n, f in frames.items()}
        common = set.intersection(*sets.values())
        print(f"  forecast rows in common: {len(common)}; " +
              ", ".join(f"only in {n}: {len(s - common)}" for n, s in sets.items()))


# run
if __name__ == "__main__":
    safe("raw datasets", check_raw)
    safe("rasters", check_rasters)
    safe("flood_panel", check_panel)
    safe("model_table", check_model_table)
    safe("forecast files", check_forecasts)

    det = pd.DataFrame(detail)
    det["pct_missing"] = (det["missing"] / det["rows"].replace(0, np.nan) * 100).round(2)
    det.to_csv("missing_detail.csv", index=False)

    summ = (det.assign(has_missing=det["missing"] > 0)
               .groupby("dataset", sort=False)
               .agg(rows=("rows", "max"), columns=("column", "count"),
                    columns_with_missing=("has_missing", "sum"),
                    missing_cells=("missing", "sum"),
                    total_cells=("rows", "sum")))
    summ["pct_cells_missing"] = (summ["missing_cells"] / summ["total_cells"] * 100).round(3)
    summ = summ.drop(columns="total_cells").reset_index()
    summ.to_csv("missing_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n Missing values per dataset")
    print(summ.to_string(index=False))

    with_missing = det[det["missing"] > 0].sort_values(["dataset", "pct_missing"], ascending=[True, False])
    if len(with_missing):
        print("\n Columns with missing values")
        print(with_missing.to_string(index=False))

    if errors:
        print("\n Could not check")
        print(pd.DataFrame(errors).to_string(index=False))

    print("\nSaved missing_summary.csv and missing_detail.csv")