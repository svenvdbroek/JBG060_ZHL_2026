from processing_data.loading_impact_data import (load_ipc_data, load_admin_boundaries)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import StrMethodFormatter




ipc = load_ipc_data()

admin1, admin2 = load_admin_boundaries()

county_state_map = (
    admin2[["adm2_name", "adm1_name"]]
    .drop_duplicates()
    .rename(columns={
        "adm2_name": "County",
        "adm1_name": "State"}))

ipc = ipc.rename(columns={
    "Estimated Population": "Estimated_pop",
    "Phase 3+ Pop": "IPC3+_pop",
    "Phase 3+ Share": "IPC3+_share"})

ipc["IPC3+_share_%"] = ipc["IPC3+_share"] * 100

#print(ipc.head(10))

#print(ipc.loc[ipc["Area_level"] == "State", ["Start Date", "Area", "IPC3+_share_%"]].sort_values("IPC3+_share_%", ascending=False).head(20))



# Saving plots
plots_dir = Path("plots/ipc")
plots_dir.mkdir(parents=True, exist_ok=True)

def save_plot(fig, filename, show=False):
    fig.tight_layout()
    fig.savefig(plots_dir / filename,
    dpi=300,
    bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


# Average IPC3+ population by county
county_avg = (ipc[ipc["Area_level"] == "County"].groupby("Area")["IPC3+_pop"].mean().sort_values(ascending=False))
#print(county_avg.head(20))


# Average IPC3+ share of total counties' population
county_avg_share = (ipc[ipc["Area_level"] == "County"].groupby("Area", as_index=False)["IPC3+_share_%"]
                    .mean().sort_values("IPC3+_share_%", ascending=False))

county_avg_share = county_avg_share.rename(columns={"Area": "County"})

county_avg_share["County"] = (
    county_avg_share["County"]
    .str.strip()
    .str.lower())

county_state_map["County"] = (
    county_state_map["County"]
    .str.strip()
    .str.lower())

county_avg_share = county_avg_share.merge(
    county_state_map,
    on="County",
    how="left")
#print("County average IPC3+ share:")
#print(county_avg_share.head(10))


# Adding state in the brackets for the respective county
county_avg_share["County_label"] = (
    county_avg_share["County"]
    + " ("
    + county_avg_share["State"].fillna("State unknown")
    + ")")

# Average IPC3+ population by state
state_avg = (ipc[ipc["Area_level"] == "State"].groupby("Area")["IPC3+_pop"].mean().sort_values(ascending=False))
#print(state_avg)

# Average IPC3+ share of total states' population
state_avg_share = (ipc[ipc["Area_level"] == "State"].groupby("Area")["IPC3+_share_%"].mean().sort_values(ascending=False))
print("State average IPC3+ share")
print(state_avg_share)

state_median_share = (
    ipc[ipc["Area_level"] == "State"].groupby("Area")["IPC3+_share_%"].median().sort_values(ascending=False))
#print(state_median_share.head(20))



# County ranking plot
top20 = (county_avg_share.head(20).sort_values("IPC3+_share_%"))

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    top20["County_label"],
    top20["IPC3+_share_%"])

ax.set_xlabel("Average population in IPC Phase 3+ (%)")
ax.set_ylabel("County")
ax.set_title("Top 20 Counties with highest average IPC Phase 3+ share")

save_plot(fig, "ipc_phase3_share_by_county.png")



# State ranking plot
state_plot = state_avg_share.sort_values()

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    state_plot.index,
    state_plot.values)

ax.set_xlabel("Average population in IPC Phase 3+ (%)")
ax.set_ylabel("State")
ax.set_title("Average IPC Phase 3+ share by state")

plt.tight_layout()

save_plot(fig, "ipc_phase3_share_by_state.png")



# State time series plot
selected_states = [
    "Northern Bahr el Ghazal",
    "Warrap",
    "Jonglei", 
    "Unity", 
    "Upper Nile",
    "Western Equatoria"]

state_time = ipc[
    (ipc["Area_level"] == "State") &
    (ipc["Area"].isin(selected_states))].copy()

state_time = state_time.sort_values("Start Date")

fig, ax = plt.subplots(figsize=(10, 6))

for state in selected_states:
    subset = state_time[state_time["Area"] == state]

    ax.plot(
        subset["Start Date"],
        subset["IPC3+_share_%"],
        marker="o",
        label=state)

ax.set_xlabel("IPC time period")
ax.set_ylabel("Population in IPC Phase 3+ (%)")
ax.set_title("IPC Phase 3+ share over time in selected states")

ax.legend()
save_plot(fig, "ipc_phase3_share_over_time_selected_states.png")



# County time series plot
selected_counties = [
    "Pibor",
    "Canal/pigi",
    "Fangak", 
    "Rubkona", 
    "Luakpiny/nasir",
    "Aweil south"]

county_time = ipc[
    (ipc["Area_level"] == "County") &
    (ipc["Area"].isin(selected_counties))].copy()

county_time = county_time.sort_values("Start Date")

fig, ax = plt.subplots(figsize=(10, 6))

for county in selected_counties:
    subset = county_time[county_time["Area"] == county]

    ax.plot(
        subset["Start Date"],
        subset["IPC3+_share_%"],
        marker="o",
        label=county)

ax.set_xlabel("IPC time period")
ax.set_ylabel("Population in IPC Phase 3+ (%)")
ax.set_title("IPC Phase 3+ share over time in selected counties")

ax.legend()

save_plot(fig, "ipc_phase3_share_over_time_selected_counties.png")
plt.show()




