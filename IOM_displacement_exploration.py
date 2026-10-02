import pandas as pd


file_path = "raw_data/External_datasets/IOM_DTM_SSD_EventTracking_Dataset_January- December 2025_20260121.xlsx"

df_IOM = pd.read_excel(file_path, sheet_name="DTM_Event Tracking_Dataset")

idps_pop = df_IOM[df_IOM["Affected population category"] == "IDPs"].copy()

# The states where displaced persons moved to
target_states_to_move = idps_pop[idps_pop["Event Location:  State (Admin 1)"].
                         isin(["Jonglei", "Unity", "Western Bahr el Ghazal", "Western Equatoria"])].copy()

target_states_to_move = target_states_to_move[[
    "Event SSID",
    "Assessment Date",
    "Event Location:  State (Admin 1)",
    "Event Location:  County (Admin 2)",
    "Arrival Location: State (Admin 1)",
    "Arrival Location: County (Admin 2)",
    "No. of Individuals",
    "Movement Trigger"]]

# Reasons that caused their displacement to this state
trigger_by_state_destination = (target_states_to_move.groupby(["Event Location:  State (Admin 1)", "Movement Trigger"])
                    ["No. of Individuals"].sum().reset_index())

print(trigger_by_state_destination)

state_total_pop_movement = (target_states_to_move.groupby(["Event Location:  State (Admin 1)"])["No. of Individuals"].sum())

flood_pop_state = (target_states_to_move[target_states_to_move["Movement Trigger"] == "Disaster (Flooding)"].
                   groupby(["Event Location:  State (Admin 1)"])["No. of Individuals"].sum())


share_of_flood_state = (flood_pop_state / state_total_pop_movement) * 100

#print(share_of_flood_state)



target_states_to_origin = idps_pop[idps_pop["Arrival Location: State (Admin 1)"].
                         isin(["Jonglei", "Unity", "Western Bahr el Ghazal", "Western Equatoria"])].copy()


trigger_state_origin = (target_states_to_origin.groupby(["Arrival Location: State (Admin 1)", "Movement Trigger"])
                    ["No. of Individuals"].sum().reset_index())

#print(trigger_state_origin)



origin_county_for_flooding = (target_states_to_origin[target_states_to_origin["Movement Trigger"] == "Disaster (Flooding)"].
                              groupby(["Arrival Location: State (Admin 1)", "Arrival Location: County (Admin 2)"])["No. of Individuals"]
                              .sum().sort_values(ascending=False))

#print(origin_county_for_flooding)


destination_counties_for_flooding = (target_states_to_move[target_states_to_move["Movement Trigger"] == "Disaster (Flooding)"].
                                groupby(["Event Location:  State (Admin 1)", "Event Location:  County (Admin 2)"])["No. of Individuals"]
                                .sum().sort_values(ascending=False))

#print(destination_counties_for_flooding)



flood_flows = (target_states_to_origin[target_states_to_origin["Movement Trigger"] == "Disaster (Flooding)"]
               .groupby([ "Arrival Location: State (Admin 1)",
                        "Arrival Location: County (Admin 2)",
                        "Event Location:  State (Admin 1)",
                        "Event Location:  County (Admin 2)"])
                        ["No. of Individuals"].sum().reset_index()
                        .sort_values("No. of Individuals", ascending=False))
#print(flood_flows)

fangak_flood = idps_pop[
    (idps_pop["Movement Trigger"] == "Disaster (Flooding)") &
    (idps_pop["Arrival Location: State (Admin 1)"] == "Jonglei") &
    (idps_pop["Arrival Location: County (Admin 2)"] == "Fangak")].copy()


print(fangak_flood["No. of Individuals"].sum())

fangak_payam_flows = (
    fangak_flood
    .groupby([
        "Arrival Location: Payam (Admin 3)",
        "Event Location:  Payam (Admin 3)"
    ])["No. of Individuals"]
    .sum()
    .reset_index()
    .sort_values("No. of Individuals", ascending=False))

#print(fangak_payam_flows)

location_type_fagak = (fangak_flood.groupby("Location type")["No. of Individuals"].sum())

#print(location_type_fagak)

location_type_states = (target_states_to_origin.groupby(["Arrival Location: State (Admin 1)", "Location type"])["No. of Individuals"].sum())
#print(location_type_states)

transport_fangak = (fangak_flood.groupby("Mode of Transport")["No. of Individuals"].sum())
print(transport_fangak)

transport_states = (target_states_to_origin.groupby(["Arrival Location: State (Admin 1)", "Mode of Transport"])["No. of Individuals"].sum())
#print(transport_states)