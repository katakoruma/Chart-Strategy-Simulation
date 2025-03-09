import numpy as np
import matplotlib.pyplot as plt 
from performance_simulation import PerformanceSimulation

yearly_return = 1.07

daily_return = 1.001
daily_loss = 1 - 0.01


gain_phase = 0.7
loss_phase = 1 - gain_phase

dt = 15
time = 261

mode = "constant_timesteps"
#mode = "constant_gain"



sim = PerformanceSimulation(yearly_return, daily_return, daily_loss, gain_phase, loss_phase, time, dt, mode)

sim.load_data()
data = sim.import_data_np

random_swing_performance_analyse, trade_dates_random  = sim.random_swing_trade_ana(set='data', trades=10, trade_dates=None)
swing_performance_analyse, trade_dates = sim.swing_trade_ana(set='data', smooth_period=1, trades=20, hold_time=14, time_after_reversel=0, trade_dates=None)

print("Buy and hold return: ", data[-1])
print("Random swing trade return analyse: ", random_swing_performance_analyse[-1])
print("Swing trade return analyse: ", swing_performance_analyse[-1])


plt.plot(data, label="Buy and hold")
plt.plot(swing_performance_analyse, label="Swing trade analyse")
plt.plot(random_swing_performance_analyse, label="Random swing trade analyse")
plt.axhline(1, color="black", linestyle="--")   

plt.xlabel("Time")
plt.ylabel("Performance")

plt.grid()
plt.legend()

plt.show()