import numpy as np
import pandas as pd

def generate_heat_demand(year=2023, annual_demand=100e6):


    if __name__ == "__main__":
        df = generate_heat_demand()
        df.to_csv("heat_demand.csv", index=False)
        print("Heat demand profile generated and saved as heat_demand.csv")
    # Time index
    hours = pd.date_range(f"{year}-01-01", periods=8760, freq="H")

    # Synthetic outdoor temperature
    T_mean = 10
    T_amp = 10
    T_out = T_mean + T_amp * np.sin(2 * np.pi * (hours.dayofyear / 365))
    T_out += np.random.normal(0, 2, size=8760)

    # Heating demand (degree-day model)
    T_base = 18
    Q_raw = np.maximum(0, T_base - T_out)

    # Daily profile
    daily_profile = []
    for h in hours.hour:
        if 6 <= h <= 9:
            daily_profile.append(1.3)
        elif 18 <= h <= 22:
            daily_profile.append(1.2)
        elif 0 <= h <= 5:
            daily_profile.append(0.9)
        else:
            daily_profile.append(0.8)

    Q = Q_raw * np.array(daily_profile)

    # Scale to annual demand
    Q_final = Q * (annual_demand / Q.sum())

    df = pd.DataFrame({
        "time": hours,
        "temperature_C": T_out,
        "heat_demand_kWh": Q_final
    })

    return df
