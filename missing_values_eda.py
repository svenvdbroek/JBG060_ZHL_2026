import traceback
from pathlib import Path
import numpy as np
import pandas as pd

YEARS = np.arange(2015, 2026)
WINDOW = (pd.Timestamp("2015-01-01"), pd.Timestamp("2025-12-31"))
N_WINDOW_MONTHS = 132
RAW = Path("./raw_data")
MODEL_TABLE = Path("processing_data/floods/model_table.csv")
SENTINELS = [-9999, -999, -99, 9999, 99999]

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
errors = []


def safe(label, fn):
    print(f"\n {label}", flush=True)
    try:
        fn()
    except Exception as e:
        errors.append((label, f"{type(e).__name__}: {e}"))
        traceback.print_exc(limit=2)


def longest_run(s):
    s = s.dropna()
    if s.empty:
        return 0
    return int(s.groupby((s != s.shift()).cumsum()).size().max())


def to_dt_index(df):
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        for c in df.columns:
            if "date" in str(c).lower() or "time" in str(c).lower():
                idx = pd.DatetimeIndex(pd.to_datetime(df[c]))
                break
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(idx))
    return idx


def time_stats(name, series, idx):
    idx = pd.DatetimeIndex(idx).dropna().sort_values().unique()
    row = {"dataset": name, "series": series, "n_obs": len(idx)}
    if len(idx) < 2:
        return row
    d = pd.Series(idx).diff().dropna()
    step = d.median()
    in_win = idx[(idx >= WINDOW[0]) & (idx <= WINDOW[1])]
    row.update({
        "first": idx[0].date(), "last": idx[-1].date(), "typical_step": str(step),
        "expected_obs": int((idx[-1] - idx[0]) / step) + 1,
        "missing_steps": int((idx[-1] - idx[0]) / step) + 1 - len(idx),
        "n_gaps": int((d > step * 1.5).sum()), "longest_gap": str(d.max()),
        "obs_2015_2025": len(in_win),
        "months_with_data_2015_2025": int(in_win.to_period("M").nunique()),
        "pct_window_months_covered": round(in_win.to_period("M").nunique() / N_WINDOW_MONTHS * 100, 1),
    })
    return row


def value_stats(name, series, df):
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c]
        n = s.notna().sum()
        if n == 0:
            continue
        rows.append({
            "dataset": name, "series": series, "column": c, "n_valid": int(n),
            "sentinel_values": int(s.isin(SENTINELS).sum()),
            "zeros": int((s == 0).sum()), "pct_zero": round((s == 0).sum() / n * 100, 2),
            "negatives": int((s < 0).sum()),
            "longest_constant_run": longest_run(s),
            "duplicate_index": int(df.index.duplicated().sum()),
        })
    return rows


# time series
ts_rows, val_rows = [], []


def check_discharge_and_lakes():
    from processing_data.loading import load_dartmouth_data, load_lake_stations
    for sid, df in load_dartmouth_data().items():
        ts_rows.append(time_stats("river_discharge", sid, to_dt_index(df)))
        val_rows.extend(value_stats("river_discharge", sid, df))
    for lake, df in load_lake_stations().items():
        ts_rows.append(time_stats(f"lake_{lake}", lake, to_dt_index(df)))
        val_rows.extend(value_stats(f"lake_{lake}", lake, df))


def check_era5():
    from processing_data.loading import load_rainfall_runoff
    ds = load_rainfall_runoff(YEARS)
    ts_rows.append(time_stats("era5_rainfall_runoff", "all", ds["valid_time"].to_index()))
    for v in ds.data_vars:
        s = ds[v]
        n = int(s.size)
        zeros = int((s == 0).sum().compute())
        neg = int((s < 0).sum().compute())
        val_rows.append({"dataset": "era5_rainfall_runoff", "series": "all", "column": v, "n_valid": n,
                         "zeros": zeros, "pct_zero": round(zeros / n * 100, 2), "negatives": neg})


def report_timeseries():
    ts = pd.DataFrame(ts_rows)
    val = pd.DataFrame(val_rows)
    ts.to_csv("hidden_timeseries_coverage.csv", index=False)
    val.to_csv("hidden_value_checks.csv", index=False)
    if len(ts):
        print("Time coverage per series (sorted by lowest coverage of 2015-2025):")
        cols = [c for c in ["dataset", "series", "first", "last", "typical_step", "missing_steps",
                            "n_gaps", "longest_gap", "months_with_data_2015_2025",
                            "pct_window_months_covered"] if c in ts.columns]
        print(ts.sort_values("pct_window_months_covered")[cols].head(25).to_string(index=False))
        if "river_discharge" in set(ts["dataset"]):
            d = ts[ts["dataset"] == "river_discharge"]
            print(f"\nDischarge: {len(d)} stations, "
                  f"{int((d['pct_window_months_covered'] < 50).sum())} cover < 50% of the 2015-2025 months, "
                  f"{int((d['pct_window_months_covered'].fillna(0) == 0).sum())} have no data in the window")
    if len(val):
        flag = val[(val.get("sentinel_values", 0) > 0) | (val.get("negatives", 0) > 0) |
                   (val.get("longest_constant_run", 0) >= 30)]
        print("\nSuspicious values (sentinels, negatives, or constant run >= 30):")
        print(flag.to_string(index=False) if len(flag) else "none found")


# flood masks
def check_flood_masks():
    from processing_data.loading import load_flood_masks
    df = load_flood_masks(YEARS)
    df["date"] = pd.to_datetime(df["date"])
    month = df["date"].dt.to_period("M")
    per_month = df.groupby(month)["date"].nunique()
    full = pd.period_range("2015-01", "2025-12", freq="M")
    per_month = per_month.reindex(full, fill_value=0)
    per_month.rename("distinct_dates_with_floods").to_csv("hidden_floodmask_dates_per_month.csv")
    zero = per_month[per_month == 0]
    print(f"months with NO flood record: {len(zero)} of {len(full)}")
    if len(zero):
        print("  ", ", ".join(map(str, zero.index)))
    print("distinct flood dates per month: min", per_month.min(), "median", per_month.median(),
          "max", per_month.max())
    per_year = df.groupby(df["date"].dt.year)["date"].nunique()
    print("\ndistinct dates per year:\n", per_year.to_string())
    dups = int(df.duplicated(["date", "lat", "lon"]).sum())
    print(f"\nduplicate (date, lat, lon) rows: {dups}")
    if "tile" in df.columns:
        print("rows per tile:\n", df["tile"].value_counts().to_string())
    print("NOTE: an absent flood record can mean 'no flood' or 'not observed' (cloud/no overpass);"
          " a month with very few distinct dates is suspicious.")


# model_table
def check_model_table():
    raw = pd.read_csv(MODEL_TABLE)
    print("rows:", len(raw))
    dups = int(raw.duplicated(["adm2_pcode", "year", "month"]).sum())
    print("duplicate (county, year, month) rows:", dups)

    ym = raw[["year", "month"]].drop_duplicates()
    expected = len(ym)
    per_county = raw.groupby("adm2_pcode").size()
    incomplete = per_county[per_county < expected]
    print(f"months in table: {expected}; counties: {raw['adm2_pcode'].nunique()}; "
          f"counties with missing months: {len(incomplete)}")
    if len(incomplete):
        print(incomplete.sort_values().head(15).to_string())

    print("missing county-month rows in total:",
          raw["adm2_pcode"].nunique() * expected - len(raw))

    if "coverage" in raw.columns:
        print("\ncoverage (share of county observed by satellite):")
        print(raw["coverage"].describe().round(3).to_string())
        print("rows with coverage <= 0.5:", int((raw["coverage"] <= 0.5).sum()),
              f"({(raw['coverage'] <= 0.5).mean() * 100:.1f}%)")
        by_year = raw.groupby("year")["coverage"].agg(lambda s: (s <= 0.5).mean() * 100).round(1)
        print("% rows with coverage <= 0.5 per year:\n", by_year.to_string())

    num = raw.select_dtypes(include="number")
    sent = {c: int(num[c].isin(SENTINELS).sum()) for c in num.columns if num[c].isin(SENTINELS).any()}
    print("\nsentinel values per column:", sent or "none")
    if "frac_flooded" in raw.columns:
        print(f"frac_flooded == 0 in {(raw['frac_flooded'] == 0).mean() * 100:.1f}% of rows")
    ordered = raw.sort_values(["adm2_pcode", "year", "month"])
    lake_cols = [x for x in raw.columns if x.startswith("lake_")]
    stuck = {c: int(ordered.groupby("adm2_pcode")[c].apply(longest_run).max()) for c in lake_cols}
    if stuck:
        print("longest constant run per county in lake columns (months):", stuck)


# discharge station per county
def check_station_distance():
    from processing_data.loading import load_dartmouth_data
    from processing_data.loading_impact_data import load_admin_boundaries

    info = pd.read_excel(RAW / "Darthmouth Flood Observatory" / "information.xlsx")
    st = info.set_index("area id")[["latitude", "longitude"]]
    st = st[~st.index.duplicated(keep="first")]
    _, admin2 = load_admin_boundaries()
    discharge = load_dartmouth_data()

    def km(lat1, lon1, lat2, lon2):
        p = np.pi / 180
        a = (np.sin((lat2 - lat1) * p / 2) ** 2 +
             np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
        return 12742 * np.arcsin(np.sqrt(a))

    rows = []
    for _, r in admin2.iterrows():
        nid = np.hypot(st["latitude"] - r["center_lat"], st["longitude"] - r["center_lon"]).idxmin()
        idx = to_dt_index(discharge[nid])
        idx = idx[(idx >= WINDOW[0]) & (idx <= WINDOW[1])]
        rows.append({"adm2_name": r["adm2_name"], "station": nid,
                     "distance_km": round(km(r["center_lat"], r["center_lon"],
                                             st.loc[nid, "latitude"], st.loc[nid, "longitude"]), 1),
                     "station_months_2015_2025": int(idx.to_period("M").nunique())})
    out = pd.DataFrame(rows)
    out.to_csv("hidden_station_per_county.csv", index=False)
    print(out["distance_km"].describe().round(1).to_string())
    print(f"\ncounties > 100 km from their station: {(out['distance_km'] > 100).sum()} of {len(out)}")
    print(f"counties > 200 km from their station: {(out['distance_km'] > 200).sum()}")
    print(f"counties whose station has < 60 of {N_WINDOW_MONTHS} months of data in 2015-2025: "
          f"{(out['station_months_2015_2025'] < 60).sum()}")
    print("\ncounties per station:\n", out["station"].value_counts().to_string())


if __name__ == "__main__":
    safe("1a. discharge + lake time series", check_discharge_and_lakes)
    safe("1b. ERA5 time coverage", check_era5)
    report_timeseries()
    safe("2. flood masks", check_flood_masks)
    safe("3. model_table.csv", check_model_table)
    safe("4. discharge station per county", check_station_distance)

    if errors:
        print("\n Could not check")
        for label, e in errors:
            print(f"{label}: {e}")
    print("\nSaved hidden_*.csv files")