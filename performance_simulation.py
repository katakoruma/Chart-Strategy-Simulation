import numpy as np 
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit
import multiprocessing
import warnings

class PerformanceAnalyzer(object):

    def __init__(self, 
                 time=261, 
                 length_of_year=261,
                 initial_investment=1, 
                 saving_plan=0,
                 saving_plan_period=22,
                 smooth_period=5, 
                 max_trades=20,  
                 hold_time=[14,3,7,0], 
                 time_after_reversel=0, 
                 trade_cost=0, 
                 spread=0, 
                 asset_cost=0,
                 tax_rate=0,
                 tax_allowance=0,
                 *args, **kwargs):
        
        self.time = time
        self.length_of_year = length_of_year

        self.initial_investment = float(initial_investment)
        self.saving_plan_period = saving_plan_period
        self.saving_plan = saving_plan
    
        self.smooth_period = smooth_period
        self.max_trades = max_trades
        self.hold_time = hold_time
        self.time_after_reversel = time_after_reversel

        self.trade_cost = trade_cost
        self.spread = spread
        self.asset_cost = float(asset_cost)
        self.tax_rate = float(tax_rate)
        self.tax_allowance = float(tax_allowance)

    @property
    def trade_cost(self):
        return self.__trade_cost
    
    @trade_cost.setter
    def trade_cost(self, trade_cost):
        if type(trade_cost) == float or type(trade_cost) == int:
            trade_cost = float(trade_cost)
            self.__trade_cost = np.array([trade_cost, trade_cost])
        elif type(trade_cost) == list or type(trade_cost) == np.ndarray:
            assert len(trade_cost) == 2, 'Trade cost must be a list or array of length 2'
            trade_cost = np.array([float(trade_cost[0]), float(trade_cost[1])])
            self.__trade_cost = trade_cost
        else:
            raise ValueError('Trade cost must be either a float, an integer, a list or an array')
        assert np.all(self.__trade_cost >= 0), 'Trade cost must be greater than or equal to 0'

    @property
    def spread(self):
        return self.__spread
    
    @spread.setter
    def spread(self, spread):
        if type(spread) == float or type(spread) == int:
            spread = float(spread)
            self.__spread = np.array([spread, spread])
        elif type(spread) == list or type(spread) == np.ndarray:
            assert len(spread) == 2, 'Spread must be a list or array of length 2'
            spread = np.array([float(spread[0]), float(spread[1])])
            self.__spread = spread
        else:
            raise ValueError('Spread must be either a float, an integer, a list or an array')
        assert np.all(self.__spread  >= 0), 'Spread must be greater than or equal to 0'


    @property
    def saving_plan(self):  
        return self.__saving_plan

    @saving_plan.setter
    def saving_plan(self, saving_plan):
         
        self.investet_over_time = np.array([self.initial_investment])

        if type(saving_plan) == dict:

            assert 1 in saving_plan.keys(), 'The saving plan must start at 1'
            assert all([i <= self.time//self.saving_plan_period for i in saving_plan.keys()]), 'There must be less saving plan entries than the number of saving plan periods'

            saving_plan = {k: float(v) for k, v in saving_plan.items()}

            changing_executions = list(saving_plan.keys())
            changing_executions.sort()

            saving_plan[self.time//self.saving_plan_period+1] = saving_plan[changing_executions[-1]]
            changing_executions.append(self.time//self.saving_plan_period+1)

            self.total_investment = self.initial_investment + np.sum([saving_plan[changing_executions[i]] * (changing_executions[i+1] - changing_executions[i]) for i in range(len(changing_executions)-1)])
            
            self.saving_plan_sched = np.array([self.initial_investment])
            for i in range(self.time-1):
                if i % self.saving_plan_period == 0 and i != 0:
                    if i//self.saving_plan_period in changing_executions:
                        current_save = saving_plan[i//self.saving_plan_period]
                    self.saving_plan_sched = np.append(self.saving_plan_sched, current_save )
                else:
                    self.saving_plan_sched = np.append(self.saving_plan_sched, 0)

            self.investet_over_time = np.cumsum(self.saving_plan_sched)

        elif type(saving_plan) == float or type(saving_plan) == int:
            saving_plan = float(saving_plan)
            self.total_investment = self.initial_investment + saving_plan * (self.time//self.saving_plan_period)

            self.saving_plan_sched = np.array([self.initial_investment])
            for i in range(self.time-1):
                if i % self.saving_plan_period == 0 and i != 0:
                    self.saving_plan_sched = np.append(self.saving_plan_sched, saving_plan )
                else:
                    self.saving_plan_sched = np.append(self.saving_plan_sched, 0)

            self.investet_over_time = np.cumsum(self.saving_plan_sched)
        
        else:
            raise ValueError('Saving plan must be either a float, an integer or a dictionary')
        
        assert np.isclose(self.total_investment, self.investet_over_time[-1]), 'Something went wrong. The total investment does not match the sum of the saving plan and the initial investment'

        self.__saving_plan = saving_plan


    def random_swing_trade(self, 
                           data=None, 
                           trade_dates=None, 
                           max_trades=None, 
                           hold_time=None,
                           trade_cost=None, 
                           spread=None, 
                           saving_plan=None, 
                           saving_plan_period=None, 
                           asset_cost=None,
                           tax_rate=None, 
                           tax_allowance=None,
                           return_full_arr=True,
                           *args, **kwargs):

        if max_trades is None:
            max_trades = self.max_trades
        if trade_cost is None:
            trade_cost = self.trade_cost
        if hold_time is None:
            hold_time = self.hold_time
        if spread is None:
            spread = self.spread
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
        if asset_cost is None:
            asset_cost = self.asset_cost
        if tax_rate is None:
            tax_rate = self.tax_rate
        if tax_allowance is None:
            tax_allowance = self.tax_allowance
        if data is None:
            data = self.performance

        if trade_dates is None:

            trade_dates = np.array([])
            i = 0

            while i in range(self.time) and len(trade_dates) < max_trades-1:

                trade_dates = np.append(trade_dates, i)
                i += max(1,round(np.random.normal(hold_time[0], hold_time[2])))
                trade_dates = np.append(trade_dates, i)
                i += max(1,round(np.random.normal(hold_time[1], hold_time[3])))


        if isinstance(saving_plan, dict):
            saving_plan_keys = np.array(list(saving_plan.keys()))  
            saving_plan_values = np.array(list(saving_plan.values())) 
        else:
            saving_plan_values = [saving_plan]
            saving_plan_keys = [0]

        self.random_swing_performance, self.random_swing_ttwror, self.random_swing_transaction_cost, self.random_swing_tax, self.random_swing_asset_cost = compute_performance(
                                                                                                                                                                                    initial_investment=self.initial_investment, 
                                                                                                                                                                                    length_of_year=self.length_of_year, 
                                                                                                                                                                                    time=self.time, 
                                                                                                                                                                                    data=data, 
                                                                                                                                                                                    trade_dates=trade_dates, 
                                                                                                                                                                                    trade_cost=trade_cost, 
                                                                                                                                                                                    spread=spread, 
                                                                                                                                                                                    saving_plan_arr=saving_plan_values,
                                                                                                                                                                                    saving_plan_keys=saving_plan_keys, 
                                                                                                                                                                                    saving_plan_period=saving_plan_period, 
                                                                                                                                                                                    asset_cost=asset_cost, 
                                                                                                                                                                                    tax_rate=tax_rate, 
                                                                                                                                                                                    tax_allowance=tax_allowance
                                                                                                                                                                                    )
        
        if return_full_arr:
            return self.random_swing_performance,  self.random_swing_ttwror, self.random_swing_transaction_cost, self.random_swing_tax, self.random_swing_asset_cost
        else:
            return self.random_swing_performance[-1],  self.random_swing_ttwror[-1], self.random_swing_transaction_cost, self.random_swing_tax, self.random_swing_asset_cost


    def swing_trade(self, 
                    data=None, 
                    trade_dates=None, 
                    smooth_period=None, 
                    max_trades=None, 
                    hold_time=None, 
                    time_after_reversel=None, 
                    trade_cost=None, 
                    spread=None, 
                    saving_plan=None, 
                    saving_plan_period=None, 
                    asset_cost=None,
                    tax_rate=None, 
                    tax_allowance=None,
                    return_full_arr=True,
                    *args, **kwargs):

        if smooth_period is None:
            smooth_period = self.smooth_period
        if max_trades is None:
            max_trades = self.max_trades
        if hold_time is None:
            hold_time = self.hold_time
        if time_after_reversel is None:
            time_after_reversel = self.time_after_reversel
        if trade_cost is None:
            trade_cost = self.trade_cost
        if spread is None:
            spread = self.spread
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
        if asset_cost is None:
            asset_cost = self.asset_cost
        if tax_rate is None:
            tax_rate = self.tax_rate
        if tax_allowance is None:
            tax_allowance = self.tax_allowance
        if data is None:
            data = self.performance

        data_smooth = smooth(data, smooth_period)
        data_trend = np.gradient(data_smooth)

        if trade_dates is None:

            trade_dates = np.array([0])
            i = 0

            while i in range(self.time) and len(trade_dates) < max_trades:

                if i > smooth_period/2 and i < self.time - smooth_period/2:
                    if data_trend[i] > 0 and len(trade_dates) % 2 == 0:
                        trade_dates = np.append(trade_dates, i + time_after_reversel)
                        i += hold_time[0]
                    elif data_trend[i] < 0 and len(trade_dates) % 2 == 1:
                        trade_dates = np.append(trade_dates, i + time_after_reversel)
                        i += hold_time[1]
                    else:
                        i += 1
                else:
                    i += 1


        if isinstance(saving_plan, dict):
            saving_plan_keys = np.array(list(saving_plan.keys()))  
            saving_plan_values = np.array(list(saving_plan.values())) 
        else:
            saving_plan_values = [saving_plan]
            saving_plan_keys = [0]
            
        self.swing_trade_performance, self.swing_trade_ttwror, self.swing_trade_transaction_cost, self.swing_trade_tax, self.swing_trade_asset_cost = compute_performance(
                                                                                                                                                 initial_investment=self.initial_investment, 
                                                                                                                                                 length_of_year=self.length_of_year, 
                                                                                                                                                 time=self.time, 
                                                                                                                                                 data=data, 
                                                                                                                                                 trade_dates=trade_dates, 
                                                                                                                                                 trade_cost=trade_cost, 
                                                                                                                                                 spread=spread, 
                                                                                                                                                 saving_plan_arr=saving_plan_values,
                                                                                                                                                 saving_plan_keys=saving_plan_keys, 
                                                                                                                                                 saving_plan_period=saving_plan_period, 
                                                                                                                                                 asset_cost=asset_cost, 
                                                                                                                                                 tax_rate=tax_rate, 
                                                                                                                                                 tax_allowance=tax_allowance
                                                                                                                                                 )

        if return_full_arr:
            return self.swing_trade_performance, self.swing_trade_ttwror, self.swing_trade_transaction_cost, self.swing_trade_tax, self.swing_trade_asset_cost
        else:
            return self.swing_trade_performance[-1], self.swing_trade_ttwror[-1], self.swing_trade_transaction_cost, self.swing_trade_tax, self.swing_trade_asset_cost
    
    def buy_and_hold(self, 
                     data=None, 
                     trade_cost=None, 
                     spread=None, 
                     saving_plan=None, 
                     saving_plan_period=None, 
                     asset_cost=None,
                     tax_rate=None, 
                     tax_allowance=None,
                     return_full_arr=True,
                     *args, **kwargs):

        if trade_cost is None:
            trade_cost = self.trade_cost
        if spread is None:
            spread = self.spread
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
        if asset_cost is None:
            asset_cost = self.asset_cost
        if tax_rate is None:
            tax_rate = self.tax_rate
        if tax_allowance is None:
            tax_allowance = self.tax_allowance
        if data is None:
            data = self.performance
            
        trade_dates=np.array([0, self.time-1])
        #trade_dates = np.sort([-1, self.time])

        if isinstance(saving_plan, dict):
            saving_plan_keys = np.array(list(saving_plan.keys()))  
            saving_plan_values = np.array(list(saving_plan.values())) 
        else:
            saving_plan_values = [saving_plan]
            saving_plan_keys = [0]
            
        self.buy_and_hold_performance, self.buy_and_hold_ttwror, self.buy_and_hold_transaction_cost, self.buy_and_hold_tax, self.buy_and_hold_asset_cost = compute_performance(
                                                                                                                                                                                    initial_investment=self.initial_investment, 
                                                                                                                                                                                    length_of_year=self.length_of_year, 
                                                                                                                                                                                    time=self.time, 
                                                                                                                                                                                    data=data, 
                                                                                                                                                                                    trade_dates=trade_dates, 
                                                                                                                                                                                    trade_cost=trade_cost, 
                                                                                                                                                                                    spread=spread, 
                                                                                                                                                                                    saving_plan_arr=saving_plan_values, 
                                                                                                                                                                                    saving_plan_keys=saving_plan_keys,
                                                                                                                                                                                    saving_plan_period=saving_plan_period, 
                                                                                                                                                                                    asset_cost=asset_cost, 
                                                                                                                                                                                    tax_rate=tax_rate, 
                                                                                                                                                                                    tax_allowance=tax_allowance
                                                                                                                                                                                    )

        if return_full_arr:
            return self.buy_and_hold_performance, self.buy_and_hold_ttwror, self.buy_and_hold_transaction_cost, self.buy_and_hold_tax, self.buy_and_hold_asset_cost
        else:
            return self.buy_and_hold_performance[-1], self.buy_and_hold_ttwror[-1], self.buy_and_hold_transaction_cost, self.buy_and_hold_tax, self.buy_and_hold_asset_cost


    def internal_rate_of_return(self, performance=None, initial_investment=None, trade_dates=None, saving_plan=None, saving_plan_period=None, time=None, length_of_year=None, *args, **kwargs):
        
        if performance == 'buy_and_hold':
            performance = self.buy_and_hold_performance[-1]
        elif performance == 'swing_trade':
            performance = self.swing_trade_performance[-1]
        elif performance == 'random_swing_trade':
            performance = self.random_swing_performance[-1]
        elif type(performance) == float:
            raise ValueError('Performance must be either buy_and_hold, swing_trade or random_swing_trade or a float')
        
        if initial_investment is None:
            initial_investment = self.initial_investment
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
        if time is None:
            time = self.time
        if length_of_year is None:
            length_of_year = self.length_of_year

        if trade_dates is None:
            trade_dates = [i for i in range(self.time) if i % saving_plan_period == 0 and i != 0]


        def eq_return(x):

            if type(saving_plan) == dict:
                changing_executions = list(saving_plan.keys())
                changing_executions.sort()

                return (initial_investment * x**(time/length_of_year) 
                        + np.sum([saving_plan[changing_executions[j]] 
                                  * np.sum([x**(time/length_of_year - i/length_of_year) for i in trade_dates[changing_executions[j]:changing_executions[j+1]]]) 
                          for j in range(len(changing_executions)-1)])
                        - performance )

            elif type(saving_plan) == float or type(saving_plan) == int:
                return (initial_investment * x**(time/length_of_year) 
                        + saving_plan * np.sum([x**(time/length_of_year - i/length_of_year) for i in trade_dates]) 
                        - performance )

        yearly_performance = (performance/initial_investment)**((length_of_year/time))

        return fsolve(eq_return, yearly_performance)[0]

    
    def plot_performance(self, log=False, *args, **kwargs):
        
        if hasattr(self, 'dates'):
            dates = self.dates
        else:
            dates = np.arange(self.time)

        plt.plot(dates, self.performance, label="Index")
        plt.plot(dates, self.investet_over_time, label="Investment over time")
        plt.plot(dates, self.buy_and_hold_performance, label="Buy and hold")
        plt.plot(dates, self.swing_trade_performance, label="Swing trade analyse")
        plt.plot(dates, self.random_swing_performance, label="Random swing trade analyse")
        #plt.axhline(1, color="black", linestyle="--")   

        plt.xlabel("Time")
        plt.ylabel("Performance")

        plt.grid()
        plt.legend()

        if log:
            plt.yscale("log")

        plt.show()
    
    def print_results(self, accuracy=1, *args, **kwargs):

        print("Initial investment: ", f"{self.initial_investment:,}")
        print("Total investment: ", f"{self.total_investment:,}")
        print()

        strategies = ['Index Performance', 'Buy and Hold', 'Swing Trade', 'Random Swing Trade']
        metrics = ['Absolute Return', 'Relative Performance (%)', 'TTWROR (%)', 'Yearly Performance (%)', 'Internal Rate of Return (%)', 'Taxes', 'Transaction Cost', 'Asset Cost']

        data = {
            'Index Performance': [
            self.performance[-1],
            factor_to_percentage(self.performance[-1] / self.initial_investment),
            factor_to_percentage(self.performance[-1] / self.initial_investment),
            factor_to_percentage((self.performance[-1] / self.initial_investment) ** (self.length_of_year / self.time)),
            factor_to_percentage((self.performance[-1] / self.initial_investment) ** (self.length_of_year / self.time)),
            None,  # Taxes not applicable
            None,  # Transaction Cost not applicable
            None   # Asset Cost not applicable
            ],
            'Buy and Hold': [
            self.buy_and_hold_performance[-1],
            factor_to_percentage(self.buy_and_hold_performance[-1] / self.total_investment),
            factor_to_percentage(self.buy_and_hold_ttwror[-1]),
            factor_to_percentage((self.buy_and_hold_performance[-1] / self.total_investment) ** (self.length_of_year / self.time)),
            factor_to_percentage(self.internal_rate_of_return('buy_and_hold')),
            self.buy_and_hold_tax,
            self.buy_and_hold_transaction_cost,
            self.buy_and_hold_asset_cost
            ],
            'Swing Trade': [
            self.swing_trade_performance[-1],
            factor_to_percentage(self.swing_trade_performance[-1] / self.total_investment),
            factor_to_percentage(self.swing_trade_ttwror[-1]),
            factor_to_percentage((self.swing_trade_performance[-1] / self.total_investment) ** (self.length_of_year / self.time)),
            factor_to_percentage(self.internal_rate_of_return('swing_trade')),
            self.swing_trade_tax,
            self.swing_trade_transaction_cost,
            self.swing_trade_asset_cost
            ],
            'Random Swing Trade': [
            self.random_swing_performance[-1],
            factor_to_percentage(self.random_swing_performance[-1] / self.total_investment),
            factor_to_percentage(self.random_swing_ttwror[-1]),
            factor_to_percentage((self.random_swing_performance[-1] / self.total_investment) ** (self.length_of_year / self.time)),
            factor_to_percentage(self.internal_rate_of_return('random_swing_trade')),
            self.random_swing_tax,
            self.random_swing_transaction_cost,
            self.random_swing_asset_cost
            ]
        }

        self.results_df = pd.DataFrame(data, index=metrics)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(self.results_df.map(lambda x: f"{x:,.{accuracy}f}" if isinstance(x, (int, float)) else x))
    


    def print_parameters(self):
    
        print("General parameters: \n")
        print("Time: ", self.time)

        print("\nTrade parameters: \n")
        print("Max trades: ", self.max_trades)
        print("Smooth period", self.smooth_period)
        print("Hold time: ", self.hold_time)
        print("Time after reversel: ", self.time_after_reversel)
        print("Trade coast: ", self.trade_cost[0])
        print("Spread: ", self.spread)
        print("Initial investment: ", self.initial_investment)
        print("Saving plan: ", self.saving_plan)
        print("Saving plan period: ", self.saving_plan_period)
        print("\n")


@njit(parallel=False)
def compute_performance(
    initial_investment, 
    length_of_year, 
    time, 
    data, 
    trade_dates, 
    trade_cost, 
    spread, 
    saving_plan_arr,
    saving_plan_keys,
    saving_plan_period, 
    asset_cost, 
    tax_rate, 
    tax_allowance, 
    consider_loss_for_taxes=True
):

    swing_performance = np.zeros(time, dtype=np.float64)
    swing_performance[0] = initial_investment
    ttwror = np.zeros(time, dtype=np.float64)
    ttwror[0] = 1.0
    ttwror_factor = 1.0

    data_gradient = data[1:] - data[:-1]
    trade_dates = np.sort(trade_dates)
    payed_tax = 0.0
    payed_transaction_cost = 0.0
    payed_asset_cost = 0.0
    unused_tax_allowance = tax_allowance

    saving_plan = 0.0

    for i in range(time-1):
        if saving_plan_arr is not None:
            for j in range(len(saving_plan_keys)):
                if  i // saving_plan_period == saving_plan_keys[j]:
                    saving_plan = saving_plan_arr[j]
                    break

        if i % length_of_year == 0:
            unused_tax_allowance = tax_allowance

        if swing_performance[i] <= 0:
            for j in range(i, time-1):
                swing_performance[j] = 0
                ttwror[j] = 0
            break
        elif np.sum(trade_dates <= i) % 2 == 1:
            if i in trade_dates:
                swing_performance[i+1] = (swing_performance[i] - trade_cost[0]) * (1 - spread[0])
                value_at_last_trade = [swing_performance[i+1], swing_performance[i]]
                payed_transaction_cost = payed_transaction_cost + trade_cost[0] + (swing_performance[i] - trade_cost[0]) * spread[0]
            else:
                swing_performance[i+1] = swing_performance[i]
            if saving_plan != 0 and i % saving_plan_period == 0 and i != 0:
                value_at_last_trade = [value_at_last_trade[0] + (saving_plan - trade_cost[1]) * (1 - spread[1]), swing_performance[i+1] + saving_plan]
                swing_performance[i+1] += (saving_plan - trade_cost[1]) * (1 - spread[1])
                ttwror_factor = ttwror[i]
                payed_transaction_cost += trade_cost[1] + (saving_plan - trade_cost[1]) * spread[1]
            payed_asset_cost += swing_performance[i+1] * (1 + data_gradient[i] / data[i]) * (1 - (1 - asset_cost) ** (1 / length_of_year))
            swing_performance[i+1] *= (1 + data_gradient[i] / data[i]) * (1 - asset_cost) ** (1 / length_of_year)
        else:
            if i in trade_dates:
                swing_performance[i+1] = swing_performance[i] - trade_cost[0]
                payed_transaction_cost += trade_cost[0]
                if tax_rate != 0 and (swing_performance[i+1] > value_at_last_trade[0] or consider_loss_for_taxes):
                    taxable_profit = swing_performance[i+1] - value_at_last_trade[0]
                    if taxable_profit > unused_tax_allowance:
                        taxable_profit -= unused_tax_allowance
                        unused_tax_allowance = 0
                        tax = tax_rate * taxable_profit
                        swing_performance[i+1] -= tax
                        payed_tax += tax
                    else:
                        unused_tax_allowance -= taxable_profit
            else:
                swing_performance[i+1] = swing_performance[i]
            if saving_plan != 0 and i % saving_plan_period == 0 and i != 0:
                swing_performance[i+1] += saving_plan
                value_at_last_trade[1] = swing_performance[i+1]
                ttwror_factor = ttwror[i]

        ttwror[i+1] = swing_performance[i+1] / value_at_last_trade[1] * ttwror_factor

    return swing_performance, ttwror, payed_transaction_cost, payed_tax, payed_asset_cost


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def factor_to_percentage(factor):
    if type(factor) == list or type(factor) == np.ndarray:
        factor = np.array(factor)
    return (factor - 1) * 100


class ChartSimulation(PerformanceAnalyzer):

        def __init__(self,dt=15, yearly_return=1.07, daily_return=1.001, daily_loss=0.999, gain_phase=0.7, loss_phase=0.3, mode="fixed_gain_phase", *args, **kwargs):
            # Call the parent class's __init__ method to initialize inherited attributes
            super().__init__(*args, **kwargs)
            
            # Initialize additional attributes specific to ChartSimulation
            self.dt = dt
            self.yearly_return = yearly_return
            self.daily_return = daily_return
            self.daily_loss = daily_loss
            self.gain_phase = gain_phase
            self.loss_phase = loss_phase
            self.mode = mode
            # Additional initialization logic for ChartSimulation

        def simulate_performance(self, *args, **kwargs):

            if self.mode == "fixed_gain_phase":
                self.daily_return = self.yearly_return**(1/self.length_of_year/(2*self.gain_phase-1))
                self.daily_loss = 1/self.daily_return

            elif self.mode == "fixed_return":
                self.gain_phase = np.log(self.yearly_return**(1/self.length_of_year)/self.daily_loss) / np.log(self.daily_return/self.daily_loss)
                self.loss_phase = 1 - self.gain_phase 
            else:
                raise ValueError("Mode must be either fixed_gain_phase or fixed_return")

            self.expected_total_return = self.daily_return**(self.gain_phase * self.time) * self.daily_loss**(self.loss_phase * self.time)
            
            performance = np.array([self.initial_investment])
            phase = np.zeros(self.time)

            rnd = np.random.choice([0, 1], p=[self.loss_phase, self.gain_phase], size=self.time//self.dt)

            for i in range(self.time//self.dt):
                phase[i*self.dt:max((i+1)*self.dt, self.time)] = rnd[i]

            for i in range(self.time-1):
                performance = np.append(performance, performance[-1] * self.daily_return if phase[i] == 1 else performance[-1] * self.daily_loss)

            self.performance = performance
            self.phase = phase

            return performance, phase
        
        def print_parameters(self):

            print("Simulation parameters: \n")
            print("Yearly return: ", self.yearly_return)
            print("Expected total return: ", self.expected_total_return)
            print("Daily return: ", self.daily_return)
            print("Daily loss: ", self.daily_loss)
            print("Gain phase: ", self.gain_phase)
            print("Loss phase: ", self.loss_phase)
            print("\n")

            super().print_parameters()
        

class ChartImport(PerformanceAnalyzer):

    def __init__(self, 
                 path="data/msci_complete.csv", 
                 date_col="Date",
                 val_col="Close",
                 rebalancing_period=261,
                 limit=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.import_data_df = None

        self.path = path
        self.date_col = date_col
        self.val_col = val_col
        self.rebalancing_period = rebalancing_period
        if limit is None:
            self.limit = slice(self.time)
        else:
            assert type(limit) == slice, 'Limit must be a slice object'
            self.limit = limit

    @property
    def path(self):  
        return self.__path
    
    @path.setter
    def path(self, path):
        if type(path) == str:
            self.__path = path
        elif type(path) == list:
            for entry in path:
                assert type(entry) == dict, 'Path must be a list of dictionaries'
                assert 'path' in entry, 'Path must be a list of dictionaries with the key "path"'
                assert 'weight' in entry, 'Path must be a list of dictionaries with the key "weight"'
                assert 'limit' in entry, 'Path must be a list of dictionaries with the key "limit"'

            len_slice = lambda sl: (sl.stop - sl.start + (sl.step - 1)) // sl.step 

            assert np.sum([entry['weight'] for entry in path]) == 1, 'The sum of the weights must be equal to 1'
            assert np.all(len_slice(path[0]['limit']) == len_slice(path[i]['limit']) for i in range(1, len(path))), 'The size of the slices must be equal'

            self.__path = path
        else:
            raise ValueError('Path must be either a string or a dictionary')

    def load_data(
                    self, 
                    path=None, 
                    date_col=None, 
                    val_col=None, 
                    limit=None, 
                    normalize=True,  
                    time=None,
                    length_of_year=None,
                    rebalancing_period=None,
                    *args, **kwargs
                  ):
    
        if path is None:
            path = self.path
        if date_col is None:
            date_col = self.date_col
        if val_col is None:
            val_col = self.val_col
        if limit is None:
            limit = self.limit
        if time is None:
            time = self.time
        if length_of_year is None:
            length_of_year = self.length_of_year
        if rebalancing_period is None:
            rebalancing_period = self.rebalancing_period
        
        
        if type(path) == list:
            dataframes = []

            for pa in path:
                dataframes.append(pd.read_csv(pa['path'])[pa['limit']])
                dataframes[-1].reset_index(drop=True, inplace=True)
                dataframes[-1][date_col] = pd.to_datetime(dataframes[-1][date_col])
                dataframes[-1].sort_values(by=date_col, ascending=True, inplace=True)

                dataframes[-1][val_col] = dataframes[-1][val_col] / dataframes[-1].loc[0, val_col]

                if "factor" in pa.keys():
                    for i in range(len(dataframes[-1][val_col])):
                        dataframes[-1].loc[i, val_col] *= pa['factor']**(i/length_of_year)
                        

            if not np.all([dataframes[0][date_col].to_numpy() == dataframes[i][date_col].to_numpy() for i in range(1, len(dataframes))]):
                warnings.warn('The dates of the dataframes are not equal. The first dataframe will be used as the reference.')


            self.dates = dataframes[0][date_col].to_numpy()
            self.performance = np.zeros(time)
            self.performance[0] = self.initial_investment

            for i in range(1, time):

                self.performance[i] = self.performance[(i-1)//rebalancing_period * rebalancing_period] * np.sum([dataframes[j][val_col].to_numpy()[i] * path[j]['weight'] for j in range(len(path))])
            
                if i % rebalancing_period == 0 and i != 0:
                    for j in range(len(path)):
                        dataframes[j][val_col] /= dataframes[j].loc[i, val_col]

            self.import_data_df = pd.DataFrame(data=[self.dates, self.performance], index=[date_col, val_col]).T


        elif type(path) == str:
            self.import_data_df = pd.read_csv(path)

            self.import_data_df[date_col] = pd.to_datetime(self.import_data_df[date_col])
            self.import_data_df.sort_values(by=date_col, ascending=True, inplace=True) 

            # if normalize:
            #     self.import_data_df[val_col] = self.import_data_df[val_col] * self.initial_investment / self.import_data_df[val_col].iloc[0]

            self.performance = self.import_data_df[val_col].to_numpy()[limit]
            self.dates = self.import_data_df[date_col].to_numpy()[limit]

        else:
            raise ValueError('Path must be either a string or a list of dictionaries')

        if normalize:
            self.performance = self.performance * self.initial_investment / self.performance[0]

        return self.performance, self.dates
    
    def update_selection(self, date_col=None, val_col=None, limit=None, normalize=True, *args, **kwargs):

        if date_col is None:
            date_col = self.date_col
        if val_col is None:
            val_col = self.val_col
        if limit is None:
            limit = slice(self.time)
        

        self.performance = self.import_data_df[val_col].to_numpy()[limit]
        self.dates = self.import_data_df[date_col].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_investment / self.performance[0]

        return self.performance, self.dates
    
    def print_parameters(self):

        print("Data parameters: \n")
        print('path: ', self.path)
        print("\n")

        super().print_parameters()

def _parallel_sim_computation(i, sim):
    performance = sim.simulate_performance()[0]
    buy_and_hold = sim.buy_and_hold(performance, return_full_arr=False)
    random_swing = sim.random_swing_trade(performance, return_full_arr=False)
    swing_trade = sim.swing_trade(performance, return_full_arr=False)

    return performance[-1], *buy_and_hold, *random_swing, *swing_trade

def _parallel_imp_computation(i, imp, stepsize):
    performance = imp.update_selection(limit=slice(i*stepsize, imp.time + i*stepsize), normalize=True)[0]
    buy_and_hold = imp.buy_and_hold(performance, return_full_arr=False)
    random_swing = imp.random_swing_trade(performance, return_full_arr=False)
    swing_trade = imp.swing_trade(performance, return_full_arr=False)

    return performance[-1], *buy_and_hold, *random_swing, *swing_trade

class MonteCarloSimulation:

    def __init__(self, chartsim=None, chartimp=None, parallel=True, *args, **kwargs):
        self.chartsim = chartsim
        self.chartimp = chartimp
        self.parallel = parallel

    def mc_artificial_chart(self, n=1000, parallel=None, *args, **kwargs):

        if self.chartsim is None:
            self.chartsim = ChartSimulation(**kwargs)

        if parallel is None:
            parallel = self.parallel
        
        self.index_profit = np.zeros((n, self.chartsim.time))
        self.buy_and_hold_profit = np.zeros(n)
        self.random_swing_profit = np.zeros(n)
        self.swing_trade_profit = np.zeros(n)

        self.buy_and_hold_ttwror = np.zeros(n)
        self.random_swing_ttwror = np.zeros(n)
        self.swing_trade_ttwror = np.zeros(n)

        self.buy_and_hold_transaction_cost, self.buy_and_hold_tax, self.buy_and_hold_asset_cost = np.zeros(n), np.zeros(n), np.zeros(n)
        self.random_swing_transaction_cost, self.random_swing_tax, self.random_swing_asset_cost = np.zeros(n), np.zeros(n), np.zeros(n)
        self.swing_trade_transaction_cost, self.swing_trade_tax, self.swing_trade_asset_cost = np.zeros(n), np.zeros(n), np.zeros(n)

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(_parallel_sim_computation)(i, self.chartsim) for i in tqdm(range(n)))
        else:
            results = [_parallel_sim_computation(i, self.chartsim) for i in tqdm(range(n))]

        for i in range(n): (
            self.index_profit[i], 
            self.buy_and_hold_profit[i], 
            self.buy_and_hold_ttwror[i],
            self.buy_and_hold_transaction_cost[i], 
            self.buy_and_hold_tax[i],
            self.buy_and_hold_asset_cost[i],
            self.random_swing_profit[i], 
            self.random_swing_ttwror[i],
            self.random_swing_transaction_cost[i], 
            self.random_swing_tax[i], 
            self.random_swing_asset_cost[i],
            self.swing_trade_profit[i], 
            self.swing_trade_ttwror[i],
            self.swing_trade_transaction_cost[i], 
            self.swing_trade_tax[i],
            self.swing_trade_asset_cost[i]
                ) = results[i]

        self.index_profit = self.index_profit[:,-1]

        return self.index_profit, self.buy_and_hold_profit, self.random_swing_profit, self.swing_trade_profit
    
    def mc_import_chart(self, n=1000, stepsize=1, parallel=None, *args, **kwargs):

        if stepsize * n + self.chartimp.time > len(self.chartimp.import_data_df):
            raise ValueError(f"Stepsize * n + time is larger than the length of the data: {stepsize}, n: {n}, time: {self.chartimp.time}, data length: {len(self.chartimp.import_data_df)}")

        if self.chartimp is None:
            self.chartimp = ChartImport(**kwargs)
        
        if self.chartimp.import_data_df is None:
            self.chartimp.load_data(**kwargs)

        if parallel is None:
            parallel = self.parallel

        self.index_profit = np.zeros((n, self.chartimp.time))
        self.buy_and_hold_profit = np.zeros(n)
        self.random_swing_profit = np.zeros(n)
        self.swing_trade_profit = np.zeros(n)

        self.buy_and_hold_ttwror = np.zeros(n)
        self.random_swing_ttwror = np.zeros(n)
        self.swing_trade_ttwror = np.zeros(n)

        self.buy_and_hold_transaction_cost, self.buy_and_hold_tax, self.buy_and_hold_asset_cost = np.zeros(n), np.zeros(n), np.zeros(n)
        self.random_swing_transaction_cost, self.random_swing_tax, self.random_swing_asset_cost = np.zeros(n), np.zeros(n), np.zeros(n)
        self.swing_trade_transaction_cost, self.swing_trade_tax, self.swing_trade_asset_cost = np.zeros(n), np.zeros(n), np.zeros(n)

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(_parallel_imp_computation)(i, self.chartimp, stepsize) for i in tqdm(range(n)))
        else:
            results = [_parallel_imp_computation(i, self.chartimp, stepsize) for i in tqdm(range(n))]

        for i in range(n): (
            self.index_profit[i], 
            self.buy_and_hold_profit[i], 
            self.buy_and_hold_ttwror[i],
            self.buy_and_hold_transaction_cost[i], 
            self.buy_and_hold_tax[i],
            self.buy_and_hold_asset_cost[i],
            self.random_swing_profit[i], 
            self.random_swing_ttwror[i],
            self.random_swing_transaction_cost[i], 
            self.random_swing_tax[i], 
            self.random_swing_asset_cost[i],
            self.swing_trade_profit[i], 
            self.swing_trade_ttwror[i],
            self.swing_trade_transaction_cost[i], 
            self.swing_trade_tax[i],
            self.swing_trade_asset_cost[i]
                ) = results[i]

        self.index_profit = self.index_profit[:,-1]

        return self.index_profit, self.buy_and_hold_profit, self.random_swing_profit, self.swing_trade_profit

    
    def hist_performance(self, bins=50, limits=None, *args, **kwargs):

        if limits == 'minmax':
            limits = (min(np.min(self.index_profit), np.min(self.buy_and_hold_profit), np.min(self.random_swing_profit), np.min(self.swing_profit)), 
                      max(np.max(self.index_profit), np.max(self.buy_and_hold_profit), np.max(self.random_swing_profit), np.max(self.swing_profit)))

        plt.hist(self.index_profit, bins=bins, range=limits, alpha=0.5, label="Index Performance")
        plt.hist(self.buy_and_hold_profit, bins=bins, range=limits, alpha=0.5, label="Buy and hold performance")
        plt.hist(self.swing_profit, bins=bins, range=limits, alpha=0.5, label="Swing trade performance")
        plt.hist(self.random_swing_profit, bins=bins, range=limits, alpha=0.5, label="Random swing trade performance")

        plt.xlabel("Performance")
        plt.ylabel("Frequency")
        plt.title("Performance distribution")

        plt.grid()
        plt.legend()
        plt.show()

    def print_results(self, accuracy=1, *args, **kwargs):

        if not self.chartsim is None:
            time = self.chartsim.time
            length_of_year = self.chartsim.length_of_year
            total_investment = self.chartsim.total_investment
            initial_investment = self.chartsim.initial_investment
            internal_rate_func = self.chartsim.internal_rate_of_return
            # print(f"Parameters of {self.chartsim.__class__.__name__}:\n")
            # self.chartsim.print_parameters()
            # print("\n")
        if not self.chartimp is None:
            time = self.chartimp.time
            length_of_year = self.chartimp.length_of_year
            total_investment = self.chartimp.total_investment
            initial_investment = self.chartimp.initial_investment
            internal_rate_func = self.chartimp.internal_rate_of_return
            # print(f"Parameters of {self.chartimp.__class__.__name__}: \n")
            # self.chartimp.print_parameters()
            # print("\n")

        print("Initial investment: ", f"{initial_investment:,}")
        print("Total investment: ", f"{total_investment:,}")
        print()

        metrics = ['Overall Return', 'Relative Performance (%)', 'TTWROR (%)', 'Yearly Performance (%)', 'Internal Rate of Return (%)', 'Taxes', 'Transaction Cost', 'Asset Cost']
        strategies = ['Index Performance', 'Buy and Hold', 'Swing Trade', 'Random Swing Trade']

        data = {}

        data['Index Performance'] = {
            'Overall Return': [self.index_profit.mean(), self.index_profit.std(), np.median(self.index_profit)],
            'Relative Performance (%)': [np.mean(factor_to_percentage(self.index_profit / initial_investment)), np.std(factor_to_percentage(self.index_profit / initial_investment)), np.median(factor_to_percentage(self.index_profit / initial_investment))],
            'TTWROR (%)': [np.mean(factor_to_percentage(self.index_profit / initial_investment)), np.std(factor_to_percentage(self.index_profit / initial_investment)), np.median(factor_to_percentage(self.index_profit / initial_investment))],
            'Yearly Performance (%)': [np.mean(factor_to_percentage((self.index_profit / initial_investment) ** (length_of_year / time))), np.std(factor_to_percentage((self.index_profit / initial_investment) ** (length_of_year / time))), np.median(factor_to_percentage((self.index_profit / initial_investment) ** (length_of_year / time)))],
            'Internal Rate of Return (%)': [np.mean(factor_to_percentage((self.index_profit / initial_investment) ** (length_of_year / time))), np.std(factor_to_percentage((self.index_profit / initial_investment) ** (length_of_year / time))), np.median(factor_to_percentage((self.index_profit / initial_investment) ** (length_of_year / time)))],
            'Taxes': [np.nan, np.nan, np.nan],
            'Transaction Cost': [np.nan, np.nan, np.nan],
            'Asset Cost': [np.nan, np.nan, np.nan]
        }

        internal_rates = [internal_rate_func(self.buy_and_hold_profit[i]) for i in range(len(self.buy_and_hold_profit))]
        data['Buy and Hold'] = {
            'Overall Return': [self.buy_and_hold_profit.mean(), self.buy_and_hold_profit.std(), np.median(self.buy_and_hold_profit)],
            'Relative Performance (%)': [np.mean(factor_to_percentage(self.buy_and_hold_profit / total_investment)), np.std(factor_to_percentage(self.buy_and_hold_profit / total_investment)), np.median(factor_to_percentage(self.buy_and_hold_profit / total_investment))],
            'TTWROR (%)': [np.mean(factor_to_percentage(self.buy_and_hold_ttwror)), np.std(factor_to_percentage(self.buy_and_hold_ttwror)), np.median(factor_to_percentage(self.buy_and_hold_ttwror))],
            'Yearly Performance (%)': [np.mean(factor_to_percentage((self.buy_and_hold_profit / total_investment) ** (length_of_year / time))), np.std(factor_to_percentage((self.buy_and_hold_profit / total_investment) ** (length_of_year / time))), np.median(factor_to_percentage((self.buy_and_hold_profit / total_investment) ** (length_of_year / time)))],
            'Internal Rate of Return (%)': [np.mean(factor_to_percentage(internal_rates)), np.std(factor_to_percentage(internal_rates)), np.median(factor_to_percentage(internal_rates))],
            'Taxes': [np.mean(self.buy_and_hold_tax), np.std(self.buy_and_hold_tax), np.median(self.buy_and_hold_tax)],
            'Transaction Cost': [np.mean(self.buy_and_hold_transaction_cost), np.std(self.buy_and_hold_transaction_cost), np.median(self.buy_and_hold_transaction_cost)],
            'Asset Cost': [np.mean(self.buy_and_hold_asset_cost), np.std(self.buy_and_hold_asset_cost), np.median(self.buy_and_hold_asset_cost)]
        }

        internal_rates = [internal_rate_func(self.swing_trade_profit[i]) for i in range(len(self.swing_trade_profit))]
        data['Swing Trade'] = {
            'Overall Return': [self.swing_trade_profit.mean(), self.swing_trade_profit.std(), np.median(self.swing_trade_profit)],
            'Relative Performance (%)': [np.mean(factor_to_percentage(self.swing_trade_profit / total_investment)), np.std(factor_to_percentage(self.swing_trade_profit / total_investment)), np.median(factor_to_percentage(self.swing_trade_profit / total_investment))],
            'TTWROR (%)': [np.mean(factor_to_percentage(self.swing_trade_ttwror)), np.std(factor_to_percentage(self.swing_trade_ttwror)), np.median(factor_to_percentage(self.swing_trade_ttwror))],
            'Yearly Performance (%)': [np.mean(factor_to_percentage((self.swing_trade_profit / total_investment) ** (length_of_year / time))), np.std(factor_to_percentage((self.swing_trade_profit / total_investment) ** (length_of_year / time))), np.median(factor_to_percentage((self.swing_trade_profit / total_investment) ** (length_of_year / time)))],
            'Internal Rate of Return (%)': [np.mean(factor_to_percentage(internal_rates)), np.std(factor_to_percentage(internal_rates)), np.median(factor_to_percentage(internal_rates))],
            'Taxes': [np.mean(self.swing_trade_tax), np.std(self.swing_trade_tax), np.median(self.swing_trade_tax)],
            'Transaction Cost': [np.mean(self.swing_trade_transaction_cost), np.std(self.swing_trade_transaction_cost), np.median(self.swing_trade_transaction_cost)],
            'Asset Cost': [np.mean(self.swing_trade_asset_cost), np.std(self.swing_trade_asset_cost), np.median(self.swing_trade_asset_cost)]
        }

        internal_rates = [internal_rate_func(self.random_swing_profit[i]) for i in range(len(self.random_swing_profit))]
        data['Random Swing Trade'] = {
            'Overall Return': [self.random_swing_profit.mean(), self.random_swing_profit.std(), np.median(self.random_swing_profit)],
            'Relative Performance (%)': [np.mean(factor_to_percentage(self.random_swing_profit / total_investment)), np.std(factor_to_percentage(self.random_swing_profit / total_investment)), np.median(factor_to_percentage(self.random_swing_profit / total_investment))],
            'TTWROR (%)': [np.mean(factor_to_percentage(self.random_swing_ttwror)), np.std(factor_to_percentage(self.random_swing_ttwror)), np.median(factor_to_percentage(self.random_swing_ttwror))],
            'Yearly Performance (%)': [np.mean(factor_to_percentage((self.random_swing_profit / total_investment) ** (length_of_year / time))), np.std(factor_to_percentage((self.random_swing_profit / total_investment) ** (length_of_year / time))), np.median(factor_to_percentage((self.random_swing_profit / total_investment) ** (length_of_year / time)))],
            'Internal Rate of Return (%)': [np.mean(factor_to_percentage(internal_rates)), np.std(factor_to_percentage(internal_rates)), np.median(factor_to_percentage(internal_rates))],
            'Taxes': [np.mean(self.random_swing_tax), np.std(self.random_swing_tax), np.median(self.random_swing_tax)],
            'Transaction Cost': [np.mean(self.random_swing_transaction_cost), np.std(self.random_swing_transaction_cost), np.median(self.random_swing_transaction_cost)],
            'Asset Cost': [np.mean(self.random_swing_asset_cost), np.std(self.random_swing_asset_cost), np.median(self.random_swing_asset_cost)]
        }

        index = pd.MultiIndex.from_product([strategies, ['Mean', 'Std', 'Median']], names=['Strategy', 'Metric'])
        df_data = []
        for metric in metrics:
            for strategy in strategies:
                df_data.extend(data[strategy][metric])
        df_data = np.array(df_data).reshape(8,12).T
        self.results_mc_df = pd.DataFrame(df_data, index=index, columns=metrics).T

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(self.results_mc_df.map(lambda x: f"{x:,.{accuracy}f}" if isinstance(x, (int, float)) else x))


if __name__ == "__main__":

    years = 5
    time = int(261 * years)
    saving_plan = {12*i+1: 500 * 1.02**(i*12) for i in range(0,years)}
    #saving_plan = 500
    sim = ChartSimulation(time=time, saving_plan=saving_plan)

    mc = MonteCarloSimulation()
    mc.mc_artificial_chart(n=500, parallel=False)

    mc.chartsim.simulate_performance()
    mc.chartsim.buy_and_hold()
    mc.chartsim.swing_trade()
    mc.chartsim.random_swing_trade()


    mc.chartsim.plot_performance()
    mc.chartsim.print_results()

    mc.hist_performance(bins=50)
    mc.print_results()
