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

# give every pixel the number of the county it is in (0 = outside south sudan)
shapes = [(geom, i + 1) for i, geom in enumerate(admin2.geometry)]
county_raster = rasterize(shapes, out_shape=crop.shape, transform=transform)
in_ssd = county_raster > 0

# pixel is 0.00446 degrees, 1 degree is about 111 km -> roughly 0.5 x 0.5 km
pixel_km2 = (res * 111) ** 2

crop_km2 = crop.values / 100 * pixel_km2
crop_km2[~in_ssd] = 0
range_km2 = rangeland.values / 100 * pixel_km2
range_km2[~in_ssd] = 0

counties = admin2[["adm2_name", "adm1_name", "adm2_pcode", "geometry"]].copy()

crop_list = []
range_list = []
area_list = []
for i in range(len(admin2)):
    mask = county_raster == i + 1
    crop_list.append(crop_km2[mask].sum())
    range_list.append(range_km2[mask].sum())
    area_list.append(mask.sum() * pixel_km2)

counties["area_km2"] = area_list
counties["crop_km2"] = crop_list
counties["range_km2"] = range_list
counties["crop_pct"] = counties["crop_km2"] / counties["area_km2"] * 100
counties["range_pct"] = counties["range_km2"] / counties["area_km2"] * 100

print("my area / official area:", (counties["area_km2"] / admin2["area_sqkm"]).round(3).describe()[["min", "max"]].tolist())
print(counties.drop(columns="geometry").sort_values("crop_km2", ascending=False).head(15))

states = counties.groupby("adm1_name")[["area_km2", "crop_km2", "range_km2"]].sum()
states["crop_pct"] = states["crop_km2"] / states["area_km2"] * 100
states["share_of_all_crop"] = states["crop_km2"] / states["crop_km2"].sum() * 100
print(states.sort_values("crop_km2", ascending=False))

sorted_crop = counties["crop_km2"].sort_values(ascending=False)
cumsum = sorted_crop.cumsum() / sorted_crop.sum() * 100
print("biggest county alone:", round(cumsum.iloc[0], 1), "% of all cropland")
print("top 3 counties:", round(cumsum.iloc[2], 1), "%")
print("top 10 counties:", round(cumsum.iloc[9], 1), "%")

os.makedirs("images", exist_ok=True)

top = counties.sort_values("crop_km2", ascending=False).head(15)
top2 = counties.sort_values("crop_pct", ascending=False).head(15)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(top["adm2_name"], top["crop_km2"])
ax[0].invert_yaxis()
ax[0].set_title("Top 15 counties - cropland (km2)")
ax[1].barh(top2["adm2_name"], top2["crop_pct"])
ax[1].invert_yaxis()
ax[1].set_title("Top 15 counties - cropland (% of county)")
plt.tight_layout()
plt.savefig("images/crop_top_counties.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9, 7))
counties.plot(column="crop_pct", cmap="Blues", vmax=5, legend=True, ax=ax, edgecolor="gray", linewidth=0.3,
              legend_kwds={"label": "% of county that is cropland (colors capped at 5)"})
ax.set_title("Cropland % per county")
plt.savefig("images/crop_pct_county.png", dpi=150, bbox_inches="tight")

# for the flood overlay later
os.makedirs("processing_data/crops", exist_ok=True)
counties.drop(columns="geometry").to_csv("processing_data/crops/crop_by_county.csv", index=False)
