from processing_data.loading_impact_data import load_cattle, load_admin_boundaries
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import matplotlib.patheffects as pe



df, longitudes, latitudes = load_cattle()

#print(df.shape)
#print(df.head())
#print(latitudes[:5])
#print(latitudes[-5:])

#print(longitudes.min(), longitudes.max())
#print(latitudes.min(), latitudes.max())



admin1, admin2 = load_admin_boundaries()

ssd_cattle = df.loc[
    (df.index >= 3.47) & (df.index <= 12.3),
    (df.columns >= 24.10) & (df.columns <= 36)]


ssd_cattle_plot = ssd_cattle.sort_index()

vmax = np.nanpercentile(ssd_cattle.values, 99)

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    ssd_cattle_plot.values,
    extent=[
        ssd_cattle_plot.columns.min(),
        ssd_cattle_plot.columns.max(),
        ssd_cattle_plot.index.min(),
        ssd_cattle_plot.index.max()
    ],
    origin="lower",
    aspect="auto",
    vmin=0,
    vmax=vmax
)

# Draw state boundaries
admin1.boundary.plot(
    ax=ax,
    edgecolor="red",
    linewidth=1.1
)

plt.colorbar(
    im,
    ax=ax,)

ax.set_title("Cattle population map")

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
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Creating grid of all cattle pixel coordinates
lon_grid, lat_grid = np.meshgrid(
    ssd_cattle.columns.values,
    ssd_cattle.index.values
)

# Converting every cattle pixel to a geographic point
cattle_points = gpd.GeoDataFrame(
    {
        "cattle": ssd_cattle.values.ravel()
    },
    geometry=gpd.points_from_xy(
        lon_grid.ravel(),
        lat_grid.ravel()
    ),
    crs="EPSG:4326"
)

cattle_states = gpd.sjoin(
    cattle_points,
    admin1[["adm1_name", "geometry"]],
    how="inner",
    predicate="within"
)

print("Cells inside South Sudan:", len(cattle_states))

print(
    "Missing cells inside South Sudan:",
    cattle_states["cattle"].isna().sum()
)



print("Zero cattle cells:", (cattle_states["cattle"] == 0).sum())

print(np.nanpercentile(cattle_states["cattle"].to_numpy(), 99))

print(np.nanmax(cattle_states["cattle"].to_numpy()))

missing_by_state = (
    cattle_states
    .groupby("adm1_name")["cattle"]
    .agg(
        total_cells="size",
        missing_cells=lambda x: x.isna().sum(),
        zero_cells=lambda x: (x == 0).sum()
    )
)
print(missing_by_state)

cattle_by_state = (
    cattle_states
    .groupby("adm1_name")["cattle"]
    .agg(
        total_cattle="sum",
        valid_pixels="count",
        average_cattle_per_pixel="mean"
    )
    .sort_values("average_cattle_per_pixel", ascending=False)
)

print(cattle_by_state)



state_cattle = (
    cattle_states
    .groupby("adm1_name")["cattle"]
    .sum()
    .reset_index(name="total_cattle"))

admin1_area = admin1.to_crs("EPSG:6933").copy()

admin1_area["area_km2"] = (admin1_area.geometry.area / 1_000_000)
state_area = admin1_area[["adm1_name", "area_km2"]]

state_stats = state_cattle.merge(
    state_area,
    on="adm1_name")

state_stats["cattle_per_km2"] = (
    state_stats["total_cattle"]
    / state_stats["area_km2"])

state_stats = state_stats.sort_values(
    "cattle_per_km2",
    ascending=False
)

print(state_stats)

total_cattle_country = cattle_states["cattle"].sum()
total_area_country = state_stats["area_km2"].sum()

cattle_density_country = (
    total_cattle_country / total_area_country
)

print(f"Total area: {total_area_country:,.0f} km²")
print(f"Estimated cattle population: {total_cattle_country:,.0f}")
print(f"Estimated cattle density: {cattle_density_country:.2f} cattle/km²")



plt.show()


