'''
Builds a dataset to able to use for the RF-model
'''

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pathlib import Path

from processing_data.loading import (
    load_dartmouth_data,
    load_lake_stations,
    load_rainfall_runoff,
    load_flood_masks,
)
from processing_data.loading_impact_data import (
    load_admin_boundaries,
    load_health_facilities,
)

YEARS = np.arange(2015, 2026)
RAW_DATA_DIR = Path("./raw_data")
OUT_DIR = Path("./processed_data_rf")
OUT_DIR.mkdir(exist_ok=True)


# Admin boundaries / spatial helpers

def load_admin2() -> gpd.GeoDataFrame:
    _, admin2 = load_admin_boundaries()
    return admin2.to_crs(epsg=4326)


def assign_points_to_counties(lons, lats, admin2: gpd.GeoDataFrame) -> np.ndarray:
    """Vectorised point-in-polygon join; returns adm2_name per point ('None' if outside any county)."""
    points = gpd.GeoDataFrame(
        {"lon": lons, "lat": lats},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, admin2[["adm2_name", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["adm2_name"].reindex(points.index).fillna("None").values


def full_panel_skeleton(admin2: gpd.GeoDataFrame, years: np.ndarray) -> pd.DataFrame:
    """
    All (county, month) combinations, so months with zero rain/floods are kept as 0, not dropped.
    """
    months = pd.period_range(f"{years.min()}-01", f"{years.max()}-12", freq="M")
    counties = admin2["adm2_name"].unique()
    return pd.MultiIndex.from_product([counties, months], names=["adm2_name", "month"]).to_frame(index=False)


# Target: flood occurrence per county-month

def build_flood_target(years: np.ndarray, admin2: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    flood_type == 0 is "recurring" and flood_type == 1 is "unusual". Both counts are
    kept separate: "flooded" = any flood at all, "unusual_flood" = the harder, more
    interesting prediction target.
    """
    print("Loading flood masks...")
    df = load_flood_masks(years)
    if df.empty:
        raise RuntimeError("No flood mask data loaded - check raw_data/flood_masks path")

    df["adm2_name"] = assign_points_to_counties(df["lon"].values, df["lat"].values, admin2)
    df = df[df["adm2_name"] != "None"]
    df["month"] = df["date"].dt.to_period("M")

    agg = (
        df.groupby(["adm2_name", "month"])
        .agg(
            flood_pixel_count=("flood_type", "size"),
            unusual_pixel_count=("flood_type", "sum"),
        )
        .reset_index()
    )
    agg["flooded"] = (agg["flood_pixel_count"] > 0).astype(int)
    agg["unusual_flood"] = (agg["unusual_pixel_count"] > 0).astype(int)
    return agg


# Hazard features: rainfall & runoff (ERA5, nearest 0.25 deg grid cell per county)

def build_rainfall_runoff_features(years: np.ndarray, admin2: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Loading rainfall/runoff")
    ds = load_rainfall_runoff(years)
    ds_monthly = ds.resample(valid_time="1MS").sum().compute()

    records = []
    for _, row in admin2.iterrows():
        cell = ds_monthly.sel(latitude=row["center_lat"], longitude=row["center_lon"], method="nearest")
        months = pd.to_datetime(cell["valid_time"].values).to_period("M")
        records.append(pd.DataFrame({
            "adm2_name": row["adm2_name"],
            "month": months,
            "precip_mm": cell["tp"].values * 1000,  # m/day -> mm accumulated over the month
            "runoff_mm": cell["ro"].values * 1000,
        }))
    return pd.concat(records, ignore_index=True)


# Hazard features: river discharge (nearest Dartmouth station per county)

def build_discharge_features(admin2: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Loading river discharge stations...")
    info = pd.read_excel(RAW_DATA_DIR / "Darthmouth Flood Observatory" / "information.xlsx")
    discharge = load_dartmouth_data()
    station_coords = info.set_index("area id")[["latitude", "longitude"]]

    records = []
    for _, row in admin2.iterrows():
        dists = np.hypot(station_coords["latitude"] - row["center_lat"],
                          station_coords["longitude"] - row["center_lon"])
        nearest_id = dists.idxmin()
        df = discharge[nearest_id]
        discharge_col = [c for c in df.columns if "discharge" in c.lower()][0]
        monthly = df[discharge_col].resample("MS").mean()
        records.append(pd.DataFrame({
            "adm2_name": row["adm2_name"],
            "month": monthly.index.to_period("M"),
            "discharge_m3s": monthly.values,
            "discharge_station_id": nearest_id,
        }))
    return pd.concat(records, ignore_index=True)


# Hazard features: upstream lake levels (national signal - same value for every county
# in a given month, since these lakes feed the Nile/Sudd upstream of all of South Sudan)

def build_lake_features() -> pd.DataFrame:
    print("Loading lake water levels...")
    lakes = load_lake_stations()
    level_cols = {"Albert": "water_level", "Kyoga": "height_egm2008", "victoria": "height_egm2008"}

    out = None
    for name, df in lakes.items():
        monthly = df[level_cols[name]].resample("MS").mean()
        s = monthly.to_frame(f"lake_{name.lower()}_level")
        s["month"] = s.index.to_period("M")
        s = s.reset_index(drop=True)
        out = s if out is None else out.merge(s, on="month", how="outer")
    return out


# Static exposure features from rasters (population, cropland, rangeland, cattle)

def zonal_stat(raster_path: Path, admin2: gpd.GeoDataFrame, stat: str = "sum") -> np.ndarray:
    """Zonal statistic (sum or mean) of a raster within each county polygon."""
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        geoms = admin2.to_crs(src.crs).geometry if admin2.crs != src.crs else admin2.geometry
        values = []
        for geom in geoms:
            try:
                arr, _ = rio_mask(src, [geom], crop=True, nodata=nodata)
            except ValueError:
                values.append(np.nan) 
                continue
            mask = (arr == nodata) if nodata is not None else None
            arr = arr.astype(np.float32)
            if mask is not None:
                arr[mask] = np.nan
            values.append(np.nansum(arr) if stat == "sum" else np.nanmean(arr))
    return np.array(values)


def build_static_exposure_features(admin2: gpd.GeoDataFrame):
    print("Computing zonal statistics for cropland, rangeland, cattle, population...")
    admin2 = admin2.copy()
    admin2["crop_pct"] = zonal_stat(RAW_DATA_DIR / "farmland" / "asap_mask_crop_v04.tif", admin2, "mean")
    admin2["rangeland_pct"] = zonal_stat(RAW_DATA_DIR / "farmland" / "asap_mask_rangeland_v04.tif", admin2, "mean")
    admin2["cattle_count"] = zonal_stat(RAW_DATA_DIR / "farmland" / "geonode__cattle_gha.tif", admin2, "sum")

    pop_by_year = {
        year: zonal_stat(RAW_DATA_DIR / "worldpop" / f"ssd_pop_{year}_CN_100m_R2025A_v1.tif", admin2, "sum")
        for year in range(2015, 2026)
    }
    pop_df = (
        pd.DataFrame(pop_by_year, index=admin2["adm2_name"])
        .reset_index()
        .melt(id_vars="adm2_name", var_name="year", value_name="population")
    )
    pop_df["year"] = pop_df["year"].astype(int)

    static = admin2[["adm2_name", "area_sqkm", "crop_pct", "rangeland_pct", "cattle_count"]]
    return static, pop_df


def build_health_facility_features(admin2: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Loading health facilities...")
    hf = load_health_facilities().to_crs(epsg=4326)
    joined = gpd.sjoin(hf, admin2[["adm2_name", "geometry"]], how="left", predicate="within")
    counts = joined.groupby("adm2_name").size().rename("health_facility_count")
    out = admin2[["adm2_name"]].merge(counts, on="adm2_name", how="left")
    out["health_facility_count"] = out["health_facility_count"].fillna(0)
    return out


# Assemble full panel

def main():
    admin2 = load_admin2()
    print(f"Loaded {len(admin2)} counties")

    panel = full_panel_skeleton(admin2, YEARS)

    flood = build_flood_target(YEARS, admin2)
    panel = panel.merge(flood, on=["adm2_name", "month"], how="left")
    fill_cols = ["flood_pixel_count", "unusual_pixel_count", "flooded", "unusual_flood"]
    panel[fill_cols] = panel[fill_cols].fillna(0)

    rain = build_rainfall_runoff_features(YEARS, admin2)
    panel = panel.merge(rain, on=["adm2_name", "month"], how="left")

    discharge = build_discharge_features(admin2)
    panel = panel.merge(discharge, on=["adm2_name", "month"], how="left")

    lakes = build_lake_features()
    panel = panel.merge(lakes, on="month", how="left")

    static, pop_df = build_static_exposure_features(admin2)
    panel = panel.merge(static, on="adm2_name", how="left")
    panel["year"] = panel["month"].dt.year
    panel = panel.merge(pop_df, on=["adm2_name", "year"], how="left")

    hf = build_health_facility_features(admin2)
    panel = panel.merge(hf, on="adm2_name", how="left")

    panel = panel.sort_values(["adm2_name", "month"]).reset_index(drop=True)
    out_path = OUT_DIR / "flood_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved panel with {len(panel)} rows and {panel.shape[1]} columns to {out_path}")
    print(panel.head())


if __name__ == "__main__":
    main()