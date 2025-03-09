
import numpy as np
import matplotlib.pyplot as plt 
from performance_simulation import *

#General parameters
dt = 15
time = 261 * 5
initial_capital = 100


#Simulation parameters
yearly_return = 1.07**5

daily_return = 1.001
daily_loss = 1 - 0.01

gain_phase = 0.7
loss_phase = 1 - gain_phase

mode = "constant_timesteps"
#mode = "constant_gain"


#Swing tade parameters
trades = 20 * 5
hold_time = 90
time_after_reversel = 0
smooth_period = 1

trade_coast = 1
spread = 0.005


sim = ChartSimulation(yearly_return, daily_return, daily_loss, gain_phase, loss_phase, mode, time, dt, initial_capital)

performance, phase = sim.simulate_performance()

random_swing_performance_analyse, trade_dates_random  = sim.random_swing_trade_ana(performance, trades=trades, trade_dates=None, trade_coast=trade_coast, spread=spread)
swing_performance_analyse, trade_dates = sim.swing_trade_ana(performance, smooth_period=smooth_period, trades=trades, hold_time=hold_time, time_after_reversel=time_after_reversel, trade_dates=None, trade_coast=trade_coast, spread=spread)

print("Buy and hold return: ", performance[-1])
print("Random swing trade return analyse: ", random_swing_performance_analyse[-1])
print("Swing trade return analyse: ", swing_performance_analyse[-1])
print("Best return: ", performance[0] * daily_return**(np.sum(phase == 1)) ) 

plt.plot(performance, label="Buy and hold")
plt.plot(swing_performance_analyse, label="Swing trade analyse")
plt.plot(random_swing_performance_analyse, label="Random swing trade analyse")
#plt.axhline(1, color="black", linestyle="--")   

plt.xlabel("Time")
plt.ylabel("Performance")

plt.grid()
plt.legend()

#plt.yscale("log")

plt.show()