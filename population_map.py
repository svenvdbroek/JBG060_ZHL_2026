import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from processing_data.loading_impact_data import load_admin_boundaries
import os

admin1, admin2 = load_admin_boundaries()

with rasterio.open("raw_data/worldpop/ssd_pop_2025_CN_100m_R2025A_v1.tif") as src:
    pop = src.read(1)
    pop = np.where(pop < 0, 0, pop)

    block = 20
    h, w = pop.shape
    h_trim, w_trim = h - h % block, w - w % block
    pop_trim = pop[:h_trim, :w_trim]
    pop_agg = pop_trim.reshape(h_trim // block, block, w_trim // block, block).sum(axis=(1, 3))

    rows, cols = np.where(pop_agg > 0)
    values = pop_agg[rows, cols]

    xs, ys = rasterio.transform.xy(src.transform, rows * block + block // 2, cols * block + block // 2)

fig, ax = plt.subplots(figsize=(8, 8))
admin2.boundary.plot(ax=ax, color="black", linewidth=0.5)

sizes = (values / values.max()) * 300
sc = ax.scatter(xs, ys, c=values, cmap="viridis", s=sizes, alpha=0.6, norm=mcolors.LogNorm())
plt.colorbar(sc, ax=ax, label="Population per cell")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Population distribution in South Sudan (2025)")

os.makedirs("images", exist_ok=True)
plt.savefig("images/population_map_2025.png", dpi=150, bbox_inches="tight")