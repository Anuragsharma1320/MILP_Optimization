import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("produkt_tu_stunde_03126.txt", sep=";")
df = df[df["TT_TU"] != -999]

df["time"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H")
df = df[df["time"].dt.year == 2025].reset_index(drop=True)

# Temperature
T = df["TT_TU"].values

# -------------------------------
# Thermal inertia (FIXED)
# -------------------------------
theta = np.zeros_like(T)

for i in range(len(T)):
    if i >= 3:
        theta[i] = (
            T[i] +
            0.5 * T[i-1] +
            0.25 * T[i-2] +
            0.125 * T[i-3]
        ) / 1.875
    else:
        theta[i] = T[i]

# -------------------------------
# Heat demand (IMPROVED MODEL)
# -------------------------------
T_base = 18
h = np.maximum(0, T_base - theta)

# -------------------------------
# Weekday factor
# -------------------------------
df["weekday"] = df["time"].dt.weekday
F = np.where(df["weekday"] < 5, 1.0, 0.9)

# -------------------------------
# Hourly factor
# -------------------------------
hour = df["time"].dt.hour

SF = []
for h_ in hour:
    if 6 <= h_ <= 9:
        SF.append(1.3)
    elif 18 <= h_ <= 22:
        SF.append(1.2)
    elif 0 <= h_ <= 5:
        SF.append(0.9)
    else:
        SF.append(0.8)

SF = np.array(SF)

# -------------------------------
# Combine
# -------------------------------
Q_raw = h * F * SF

# -------------------------------
# Scale
# -------------------------------
annual_demand_MWh = 2000 * 13  # 26000

Q_MWh = Q_raw * (annual_demand_MWh / Q_raw.sum())
Q_MW = Q_MWh

# -------------------------------
# Heat pump
# -------------------------------
COP = 3.0
P_HP_MW = Q_MW / COP

# -------------------------------
# Save
# -------------------------------
df_out = pd.DataFrame({
    "time": df["time"],
    "heat_demand_MW": Q_MW,
    "hp_electric_load_MW": P_HP_MW
})

df_out.to_csv("heat_hp_profile_2025.csv", index=False)

print("Done!")

# Debug
print("Peak heat:", Q_MW.max())
print("Peak HP:", P_HP_MW.max())