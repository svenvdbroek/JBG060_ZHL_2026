import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from processing_data.loading import load_flood_masks
from processing_data.loading_impact_data import load_admin_boundaries
import os

df = load_flood_masks(np.array([2020]))  # Change to the year of interest
df_unique = df.drop_duplicates(subset=['lat', 'lon', 'flood_type'])

admin1, admin2 = load_admin_boundaries()

gdf = gpd.GeoDataFrame(
    df_unique,
    geometry=[Point(xy) for xy in zip(df_unique["lon"], df_unique["lat"])],
    crs="EPSG:4326",
)

# keep only points inside South Sudan
gdf = gpd.clip(gdf, admin2)

fig, ax = plt.subplots(figsize=(8, 8))
admin2.boundary.plot(ax=ax, color="black", linewidth=0.5)

recurring = gdf[gdf["flood_type"] == 0]
unusual = gdf[gdf["flood_type"] == 1]

unusual.plot(ax=ax, color="red", markersize=2, label="Unusual")
recurring.plot(ax=ax, color="blue", markersize=2, label="Recurring")

ax.legend(loc="upper right")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Flood pixels in South Sudan (2025)")

os.makedirs("images", exist_ok=True)
plt.savefig("images/flood_map_2025.png", dpi=150, bbox_inches="tight")