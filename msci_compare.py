import numpy as np
import matplotlib.pyplot as plt 
from performance_simulation import *

dt = 15
time = 261 * 5


year = 0
limit = slice(year*time, (year+1)*time)



sim = ChartImport(time=time, dt=dt)

sim.load_data(limit=limit)
data = sim.import_data_np

#trade_dates_random = np.array([[0, 1, 51, 56, 62, 75, 93, 123, 146, 158, 160, 175, 179, 200, 206, 211, 236, 241, 245, 257]])

random_swing_performance_analyse, trade_dates_random = sim.random_swing_trade_ana(set='data', trades=50, trade_dates=None)
swing_performance_analyse, trade_dates = sim.swing_trade_ana(set='data', smooth_period=1, trades=100, hold_time=14, time_after_reversel=0, trade_dates=None)

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