import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_origin
from processing_data.loading_impact_data import load_admin_boundaries, load_farmland_mask
import os

admin1, admin2 = load_admin_boundaries()

bbox = {"lon_min": 24, "lat_min": 3.4, "lon_max": 36, "lat_max": 12.3}
crop = load_farmland_mask(bbox, "crop")
rangeland = load_farmland_mask(bbox, "rangeland")

lats = crop.latitude.values
lons = crop.longitude.values
res = lons[1] - lons[0]
transform = from_origin(lons[0] - res / 2, lats[0] + res / 2, res, res)

shapes = [(geom, i + 1) for i, geom in enumerate(admin2.geometry)]
county_raster = rasterize(shapes, out_shape=crop.shape, transform=transform)
in_ssd = county_raster > 0

crop_vals = crop.values
range_vals = rangeland.values

c = crop_vals[in_ssd]
r = range_vals[in_ssd]

print("pixels in south sudan:", in_ssd.sum())
print("crop min/max:", c.min(), c.max())
print("rangeland min/max:", r.min(), r.max())
print("any empty values:", np.isnan(c).any(), np.isnan(r).any())
print("% of pixels with no crop:", round((c == 0).mean() * 100, 1))
print("% of pixels with no rangeland:", round((r == 0).mean() * 100, 1))
print("pixels where crop + rangeland > 100:", ((c + r) > 100).sum())

os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(c[c > 0], bins=20, color="tab:blue")
ax[0].set_title("Crop % (only pixels with some crop)")
ax[0].set_xlabel("% of pixel")
ax[1].hist(r[r > 0], bins=20, color="tab:green")
ax[1].set_title("Rangeland % (only pixels with some rangeland)")
ax[1].set_xlabel("% of pixel")
plt.savefig("images/crop_rangeland_hist.png", dpi=150, bbox_inches="tight")

pixel_km2 = (res * 111) ** 2

crop_km2 = crop_vals / 100 * pixel_km2
crop_km2[~in_ssd] = 0
range_km2 = range_vals / 100 * pixel_km2
range_km2[~in_ssd] = 0

total_area = in_ssd.sum() * pixel_km2
print("pixel area ~", round(pixel_km2, 3), "km2")
print("total area:", round(total_area), "km2")
print("cropland:", round(crop_km2.sum()), "km2 =", round(crop_km2.sum() / total_area * 100, 2), "% of the country")
print("rangeland:", round(range_km2.sum()), "km2 =", round(range_km2.sum() / total_area * 100, 1), "% of the country")

crop_plot = np.where(in_ssd & (crop_vals > 0), crop_vals, np.nan)
range_plot = np.where(in_ssd, range_vals, np.nan)
extent = [lons[0], lons[-1], lats[-1], lats[0]]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
im = ax[0].imshow(crop_plot, extent=extent, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
admin1.boundary.plot(ax=ax[0], color="gray", linewidth=0.5)
plt.colorbar(im, ax=ax[0], label="% cropland")
ax[0].set_title("Cropland")

im = ax[1].imshow(range_plot, extent=extent, cmap="Greens", vmin=0, vmax=100, interpolation="nearest")
admin1.boundary.plot(ax=ax[1], color="gray", linewidth=0.5)
plt.colorbar(im, ax=ax[1], label="% rangeland")
ax[1].set_title("Rangeland")
plt.savefig("images/crop_rangeland_map.png", dpi=150, bbox_inches="tight")

crop_km2_da = crop.copy(data=crop_km2)
blocks = crop_km2_da.coarsen(latitude=12, longitude=12, boundary="trim").sum()
blocks_df = blocks.to_dataframe(name="crop_km2").reset_index()
blocks_df = blocks_df[blocks_df["crop_km2"] > 0.5].sort_values("crop_km2")
print(len(blocks_df), "blocks with more than 0.5 km2 of crop")

fig, ax = plt.subplots(figsize=(10, 8))
admin2.boundary.plot(ax=ax, color="lightgray", linewidth=0.3)
admin1.boundary.plot(ax=ax, color="gray", linewidth=0.8)
sc = ax.scatter(blocks_df["longitude"], blocks_df["latitude"], s=blocks_df["crop_km2"] * 4,
                c=blocks_df["crop_km2"], cmap="Blues", vmin=0, vmax=30, edgecolors="k", linewidths=0.2)
for i, row in admin1.iterrows():
    ax.text(row["center_lon"], row["center_lat"], row["adm1_name"], fontsize=8, ha="center")
plt.colorbar(sc, label="km2 of cropland in block")
ax.set_title("Cropland hotspots in South Sudan")
plt.savefig("images/crop_hotspots.png", dpi=150, bbox_inches="tight")

os.makedirs("processing_data/crops", exist_ok=True)
blocks_df.to_csv("processing_data/crops/crop_blocks_6km.csv", index=False)
