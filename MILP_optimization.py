import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus

# --- Load Data ---
# Ensure your Excel file is in the same directory
data = pd.read_excel('MG_Data_MILP (3).xlsx')
PV = data['PV'].values
Wind = data['Wind'].values
Load = data['Demand'].values
Price = data['Price'].values  # Base price profile (€/kWh)

n = len(PV)
dt = 0.25  # 15-min resolution

# --- Buy/Sell Tariffs ---
Price_buy = Price
Price_sell = 0.60 * Price

# --- Parameters ---
BESS_capacity = 150
SOC_min = 0.20 * BESS_capacity
SOC_max = 0.90 * BESS_capacity
eta_ch, eta_dis = 0.90, 0.90
P_bess_max = 75

EV_capacity = 200
EV_SOC_max = 0.95 * EV_capacity
EV_P_max = 100
eta_ev_ch, eta_ev_dis = 0.95, 0.95

# Initial SoCs
E0_BESS = 0.50 * BESS_capacity
E0_EV = 0.80 * EV_capacity

# --- Availability Windows ---
time_hours = np.arange(n) * dt
charge_intervals = [[0, 8], [11, 16], [22, 24]]
discharge_intervals = [[18, 21]]

a_ch = np.zeros(n)
a_dis = np.zeros(n)

for start, end in charge_intervals:
    a_ch[(time_hours >= start) & (time_hours < end)] = 1
for start, end in discharge_intervals:
    a_dis[(time_hours >= start) & (time_hours < end)] = 1

# EV SOC Policy
EV_SOC_min_base = 0.60 * EV_capacity
EV_SOC_min_curve = np.full(n, EV_SOC_min_base)
# Ready window 8:00-11:00
EV_SOC_min_curve[(time_hours >= 8) & (time_hours < 11)] = 0.80 * EV_capacity

# --- Optimization Setup ---
prob = LpProblem("Microgrid_Optimization", LpMinimize)

# Decision Variables
p_grid_buy = [LpVariable(f"p_grid_buy_{t}", lowBound=0) for t in range(n)]
p_grid_sell = [LpVariable(f"p_grid_sell_{t}", lowBound=0) for t in range(n)]
p_bess_ch = [LpVariable(f"p_bess_ch_{t}", lowBound=0, upBound=P_bess_max) for t in range(n)]
p_bess_dis = [LpVariable(f"p_bess_dis_{t}", lowBound=0, upBound=P_bess_max) for t in range(n)]
soc_bess = [LpVariable(f"soc_bess_{t}", lowBound=SOC_min, upBound=SOC_max) for t in range(n)]

p_ev_ch = [LpVariable(f"p_ev_ch_{t}", lowBound=0, upBound=EV_P_max) for t in range(n)]
p_ev_dis = [LpVariable(f"p_ev_dis_{t}", lowBound=0, upBound=EV_P_max) for t in range(n)]
soc_ev = [LpVariable(f"soc_ev_{t}", lowBound=EV_SOC_min_curve[t], upBound=EV_SOC_max) for t in range(n)]

u_ch = [LpVariable(f"u_ch_{t}", cat='Binary') for t in range(n)]
u_dis = [LpVariable(f"u_dis_{t}", cat='Binary') for t in range(n)]

# Objective Function
prob += lpSum([p_grid_buy[t] * Price_buy[t] - p_grid_sell[t] * Price_sell[t] for t in range(n)])

# Constraints
for t in range(n):
    # 1) Energy Balance
    prob += p_grid_buy[t] - p_grid_sell[t] - p_bess_ch[t] + p_bess_dis[t] - p_ev_ch[t] + p_ev_dis[t] == Load[t] - PV[
        t] - Wind[t]

    # 2) BESS Dynamics
    if t == 0:
        prob += soc_bess[t] == E0_BESS + (p_bess_ch[t] * eta_ch - p_bess_dis[t] / eta_dis) * dt
    else:
        prob += soc_bess[t] == soc_bess[t - 1] + (p_bess_ch[t] * eta_ch - p_bess_dis[t] / eta_dis) * dt

    # 3) EV Dynamics
    if t == 0:
        prob += soc_ev[t] == E0_EV + (p_ev_ch[t] * eta_ev_ch - p_ev_dis[t] / eta_ev_dis) * dt
    else:
        prob += soc_ev[t] == soc_ev[t - 1] + (p_ev_ch[t] * eta_ev_ch - p_ev_dis[t] / eta_ev_dis) * dt

    # 4) Binary Constraints (Simultaneous ch/dis prevention & Availability)
    prob += u_ch[t] + u_dis[t] <= 1
    prob += p_ev_ch[t] <= EV_P_max * u_ch[t] * a_ch[t]
    prob += p_ev_dis[t] <= EV_P_max * u_dis[t] * a_dis[t]

# Final SOC Constraints
prob += soc_bess[n - 1] >= E0_BESS
prob += soc_ev[n - 1] >= 0.80 * EV_capacity

# Solve
prob.solve()

# --- Extract Results ---
res_p_buy = np.array([value(p_grid_buy[t]) for t in range(n)])
res_p_sell = np.array([value(p_grid_sell[t]) for t in range(n)])
res_soc_bess = np.array([value(soc_bess[t]) for t in range(n)])

# --- Plotting (Example for BESS) ---
plt.figure(figsize=(10, 6))
plt.step(time_hours, res_soc_bess / BESS_capacity * 100, label='BESS SOC %')
plt.axhline(SOC_min / BESS_capacity * 100, color='r', linestyle='--', label='Min SOC')
plt.title('BESS State of Charge')
plt.xlabel('Time (hours)')
plt.ylabel('SOC %')
plt.legend()
plt.grid(True)
plt.show()