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

def test_load_data(simulation):
    ci = ChartImport()
    data = ci.load_data()
    assert not data.empty
    assert 'Value' in data.columns
    assert len(data) > 0

def test_load_data_normalized(simulation):
    ci = ChartImport()
    data = ci.load_data(normalize=True)
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

def equivalence_swing_trade(simulation):
    performance, phase = simulation.simulate_performance()

    random_swing_performance, trade_dates_random = sim.random_swing_trade(phase, trades=5)
    swing_performance, trade_dates = sim.swing_trade(phase, trades=20,  hold_time=14, time_after_reversel=0)

    random_swing_performance_analyse, trade_dates_random  = sim.random_swing_trade_ana(performance, trades=5, trade_dates=trade_dates_random)
    swing_performance_analyse, trade_dates = sim.swing_trade_ana(performance, smooth_period=1, trades=20, hold_time=14, time_after_reversel=0, trade_dates=trade_dates)

    assert np.allclose(random_swing_performance, random_swing_performance_analyse)
    assert np.allclose(trade_swing_performancedates, swing_performance_analyse)