import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from processing_data.loading import load_flood_masks
from processing_data.loading_impact_data import load_admin_boundaries
import os

admin1, admin2 = load_admin_boundaries()

years = range(2000, 2026)
unusual_counts = []
recurring_counts = []

for year in years:
    df = load_flood_masks(np.array([year]))
    df_unique = df.drop_duplicates(subset=['lat', 'lon', 'flood_type'])

    gdf = gpd.GeoDataFrame(
        df_unique,
        geometry=[Point(xy) for xy in zip(df_unique["lon"], df_unique["lat"])],
        crs="EPSG:4326",
    )
    gdf = gpd.clip(gdf, admin2)

    unusual_count = len(gdf[gdf['flood_type'] == 1])
    recurring_count = len(gdf[gdf['flood_type'] == 0])

    unusual_counts.append(unusual_count)
    recurring_counts.append(recurring_count)

plt.figure(figsize=(10, 6))
plt.plot(years, unusual_counts, color='red', label='Unusual')
plt.plot(years, recurring_counts, color='blue', label='Recurring')
plt.xlabel('Year')
plt.ylabel('Number of floods')
plt.title('Recurring vs Unusual Floods in South Sudan (2000–2025)')
plt.legend()

os.makedirs('images', exist_ok=True)
plt.savefig('images/flood_type_trend.png', dpi=150, bbox_inches='tight')

for i in range(2000,2004):

    df_i = load_flood_masks(np.array([i]))
    print(f'year {i}: {df_i['date'].nunique()}')