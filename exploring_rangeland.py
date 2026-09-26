from processing_data.loading_impact_data import (
    load_farmland_mask,
    load_admin_boundaries
)

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import pandas as pd


bbox = {
    "lon_min": 24.10,
    "lat_min": 3.47,
    "lon_max": 36,
    "lat_max": 12.3}

rangeland = load_farmland_mask(
    bbox,
    "rangeland")

admin1, admin2 = load_admin_boundaries()

rangeland_path = "./raw_data/farmland/asap_mask_rangeland_v04.tif"

results = []

with rasterio.open("./raw_data/farmland/asap_mask_rangeland_v04.tif") as src:
    print("NoData:", src.nodata)
    print("CRS:", src.crs)
    print("Resolution:", src.res)

    admin1_raster_crs = admin1.to_crs(src.crs)

    for _, state in admin1_raster_crs.iterrows():

        state_name = state["adm1_name"]

        # Clip raster to this state polygon
        clipped, transform = mask(
            src,
            [mapping(state.geometry)],
            crop=True,
            filled=False)

        values = clipped[0]

        values = values.compressed().astype(float)

        values = values[
            (values >= 0) &
            (values <= 100)]

        #  statistics computation
        results.append({
            "adm1_name": state_name,
            "mean_rangeland_pct": np.mean(values),
            "median_rangeland_pct": np.median(values),
            "max_rangeland_pct": np.max(values)})

rangeland_by_state = pd.DataFrame(results)

#Calculation of state area in km2

admin1_area = admin1.to_crs("EPSG:6933").copy()

admin1_area["area_km2"] = (
    admin1_area.geometry.area / 1_000_000
)

state_area = admin1_area[
    ["adm1_name", "area_km2"]
]

# Combine state area and rangeland statistics


rangeland_stats = rangeland_by_state.merge(
    state_area,
    on="adm1_name"
)

rangeland_stats["estimated_rangeland_km2"] = (
    rangeland_stats["area_km2"]
    * rangeland_stats["mean_rangeland_pct"]
    / 100
)

rangeland_stats = rangeland_stats.sort_values(
    "mean_rangeland_pct",
    ascending=False
)

print(rangeland_stats)

#print(rangeland)
#print(rangeland.shape)
#print("Total pixels:", rangeland.size)
#print("Zero pixels:", np.sum(rangeland.values == 0))
#print(np.percentile(rangeland.values,[0, 25, 50, 75, 90]))
#print(np.nanmin(rangeland.values))
#print(np.nanmax(rangeland.values))
#print(np.isnan(rangeland.values).sum())

fig, ax = plt.subplots(figsize=(10, 8))

rangeland.plot(
    ax=ax,
    vmin=0,
    vmax=100,
    cbar_kwargs={
        "label": "Rangeland coverage (%)"})

admin1.boundary.plot(
    ax=ax,
    edgecolor="red",
    linewidth=1.2
)

ax.set_title("Rangeland coverage", fontweight = "bold")

for _, row in admin1.iterrows():

    point = row.geometry.representative_point()

    ax.text(
        point.x,
        point.y,
        row["adm1_name"],
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        path_effects=[
            pe.withStroke(
                linewidth=2.5,
                foreground="white"
            )
        ]
    )




plt.show()

