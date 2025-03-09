import numpy as np
import matplotlib.pyplot as plt 
from performance_simulation import *

#General parameters
dt = 15
time = 261 * 12
initial_capital = 100

#Data parameters
year = 0
limit = slice(year*261, year*261 + time)


#Swing tade parameters
trades = 20 * 7
hold_time = 30
time_after_reversel = 0
smooth_period = 40

trade_coast = 1
spread = 0.005


ci = ChartImport(time=time, dt=dt, initial_capital=initial_capital)

ci.load_data(limit=limit)
performance = ci.performance

#trade_dates_random = np.array([[0, 1, 51, 56, 62, 75, 93, 123, 146, 158, 160, 175, 179, 200, 206, 211, 236, 241, 245, 257]])

random_swing_performance_analyse, trade_dates_random  = ci.random_swing_trade_ana(performance, trades=trades, trade_dates=None, trade_coast=trade_coast, spread=spread)
swing_performance_analyse, trade_dates = ci.swing_trade_ana(performance, smooth_period=smooth_period, trades=trades, hold_time=hold_time, time_after_reversel=time_after_reversel, trade_dates=None, trade_coast=trade_coast, spread=spread)

print("Buy and hold return: ", performance[-1])
print("Random swing trade return analyse: ", random_swing_performance_analyse[-1])
print("Swing trade return analyse: ", swing_performance_analyse[-1])


plt.plot(performance, label="Buy and hold")
plt.plot(swing_performance_analyse, label="Swing trade analyse")
plt.plot(random_swing_performance_analyse, label="Random swing trade analyse")
plt.axhline(1, color="black", linestyle="--")   

plt.xlabel("Time")
plt.ylabel("Performance")

plt.grid()
plt.legend()

plt.show()