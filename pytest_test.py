import pytest
import numpy as np
from performance_simulation import *

@pytest.fixture
def simulation():
    yearly_return = 1.07
    daily_return = 1.001
    daily_loss = 1 - 0.01
    gain_phase = 0.7
    loss_phase = 1 - gain_phase
    dt = 15
    time = 261
    mode = "constant_timesteps"
    return ChartSimulation(yearly_return, daily_return, daily_loss, gain_phase, loss_phase, mode, time, dt)

@pytest.fixture
def chart_load_data():
    ci = ChartImport()
    return ci.load_data(path='990100 - MSCI World Index.csv-2.csv')

def test_simulate_performance(simulation):
    performance, phase = simulation.simulate_performance()
    assert len(performance) == simulation.time + 1
    assert len(phase) == simulation.time
    assert performance[0] == 1

def test_random_swing_trade(simulation):
    performance, phase = simulation.simulate_performance()
    swing_performance, trade_dates = simulation.random_swing_trade(phase, trades=5)
    assert len(swing_performance) == simulation.time + 1
    assert len(trade_dates) == 10

def test_swing_trade(simulation):
    performance, phase = simulation.simulate_performance()
    swing_performance, trade_dates = simulation.swing_trade(phase, trades=20, hold_time=14, time_after_reversel=0)
    assert len(swing_performance) == simulation.time + 1
    assert len(trade_dates) <= 40

def test_load_data(chart_load_data):
    data, performance = chart_load_data
    assert not data.empty
    assert 'Value' in data.columns
    assert len(data) > 0

def test_load_data_normalized(chart_load_data):
    data, performance = chart_load_data
    assert data['Value'].iloc[0] == 1

def test_random_swing_trade_ana(simulation):
    performance, phase = simulation.simulate_performance()
    swing_performance, trade_dates = simulation.random_swing_trade_ana(performance, trades=5)
    assert len(swing_performance) == simulation.time + 1
    assert len(trade_dates) == 10

def test_swing_trade_ana(simulation):
    performance, phase = simulation.simulate_performance()
    swing_performance, trade_dates = simulation.swing_trade_ana(performance, smooth_period=1, trades=20, hold_time=14, time_after_reversel=0)
    assert len(swing_performance) == simulation.time + 1
    assert len(trade_dates) <= 40

def test_equivalence_swing_trade(simulation):
    performance, phase = simulation.simulate_performance()

    random_swing_performance, trade_dates_random = simulation.random_swing_trade(phase, trades=5)
    swing_performance, trade_dates = simulation.swing_trade(phase, trades=20,  hold_time=14, time_after_reversel=0)

    random_swing_performance_analyse, trade_dates_random  = simulation.random_swing_trade_ana(performance, trades=5, trade_dates=trade_dates_random)
    swing_performance_analyse, trade_dates = simulation.swing_trade_ana(performance, smooth_period=1, trades=20, hold_time=14, time_after_reversel=0, trade_dates=trade_dates)

    assert np.allclose(random_swing_performance, random_swing_performance_analyse)
    assert np.allclose(swing_performance, swing_performance_analyse)

def test_trade_dates(simulation):

    pa = PerformanceAnalyzer(time=36)

    smooth_period = 3
    trades = 5
    hold_time = 10

    chart = np.array([0,1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1])
    chart_random = chart + np.random.normal(0, 1, len(chart))
    swing_performance, trade_dates = pa.swing_trade_ana(data=chart_random, smooth_period=smooth_period, trades=trades, hold_time=hold_time)

    print("Trade dates: ", trade_dates)
    assert np.all(trade_dates > smooth_period/2) and np.all(trade_dates < len(chart) - hold_time/2)
    assert trade_dates[0] == 2
    assert trade_dates[1] == 12
    assert trade_dates[2] in range(18,20)
    assert trade_dates[3] in range(28,30)
