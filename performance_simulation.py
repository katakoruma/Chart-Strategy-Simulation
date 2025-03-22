import numpy as np 
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

class PerformanceAnalyzer:

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
                 trade_coast=0, 
                 spread=0, 
                 tax_rate=0,
                 tax_allowance=0,
                 *args, **kwargs):
        
        self.time = time
        self.length_of_year = length_of_year

        self.initial_investment = initial_investment
        self.saving_plan = saving_plan
        self.saving_plan_period = saving_plan_period

        if type(saving_plan) == dict:

            assert 0 in saving_plan.keys(), 'The saving plan must start at 0'
            assert all([i <= time//saving_plan_period for i in saving_plan.keys()]), 'There must be less saving plan entries than the number of saving plan periods'

            changing_executions = list(saving_plan.keys())
            changing_executions.sort()

            saving_plan[time//saving_plan_period] = saving_plan[changing_executions[-1]]
            changing_executions.append(time//saving_plan_period)

            self.total_investment = initial_investment + np.sum([saving_plan[changing_executions[i]] * (changing_executions[i+1] - changing_executions[i]) for i in range(len(changing_executions)-1)])

        elif type(saving_plan) == float or type(saving_plan) == int:
            self.total_investment = initial_investment + saving_plan * (time//saving_plan_period)
        else:
            raise ValueError('Saving plan must be either a float, an integer or a dictionary')

        self.smooth_period = smooth_period
        self.max_trades = max_trades
        self.hold_time = hold_time
        self.time_after_reversel = time_after_reversel

        self.trade_coast = trade_coast
        self.spread = spread
        self.tax_rate = tax_rate
        self.tax_allowance = tax_allowance


    def random_swing_trade(self, 
                           data=None, 
                           trade_dates=None, 
                           max_trades=None, 
                           hold_time=None,
                           trade_coast=None, 
                           spread=None, 
                           saving_plan=None, 
                           saving_plan_period=None, 
                           tax_rate=None, 
                           tax_allowance=None,
                           *args, **kwargs):

        if max_trades is None:
            max_trades = self.max_trades
        if trade_coast is None:
            trade_coast = self.trade_coast
        if hold_time is None:
            hold_time = self.hold_time
        if spread is None:
            spread = self.spread
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
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


        self.random_swing_performance_analyse, self.random_swing_transaction_cost, self.random_swing_tax = self._compute_performance(data, trade_dates, trade_coast, spread, saving_plan, saving_plan_period, tax_rate, tax_allowance)
        
        # print('Random Swing Trade:', trade_dates)
        return self.random_swing_performance_analyse, self.random_swing_transaction_cost, self.random_swing_tax


    def swing_trade(self, 
                    data=None, 
                    trade_dates=None, 
                    smooth_period=None, 
                    max_trades=None, 
                    hold_time=None, 
                    time_after_reversel=None, 
                    trade_coast=None, 
                    spread=None, 
                    saving_plan=None, 
                    saving_plan_period=None, 
                    tax_rate=None, 
                    tax_allowance=None,
                    *args, **kwargs):

        if smooth_period is None:
            smooth_period = self.smooth_period
        if max_trades is None:
            max_trades = self.max_trades
        if hold_time is None:
            hold_time = self.hold_time
        if time_after_reversel is None:
            time_after_reversel = self.time_after_reversel
        if trade_coast is None:
            trade_coast = self.trade_coast
        if spread is None:
            spread = self.spread
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
        if tax_rate is None:
            tax_rate = self.tax_rate
        if tax_allowance is None:
            tax_allowance = self.tax_allowance
        if data is None:
            data = self.performance

        data_smooth = self._smooth(data, smooth_period)
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
            
        self.swing_performance_analyse, self.swing_transaction_cost, self.swing_tax = self._compute_performance(data, trade_dates, trade_coast, spread, saving_plan, saving_plan_period, tax_rate, tax_allowance)

        # print('Swing Trade:', trade_dates)
        return self.swing_performance_analyse, self.swing_transaction_cost, self.swing_tax
    
    def buy_and_hold(self, 
                     data=None, 
                     trade_coast=None, 
                     spread=None, 
                     saving_plan=None, 
                     saving_plan_period=None, 
                     tax_rate=None, 
                     tax_allowance=None,
                     *args, **kwargs):

        if trade_coast is None:
            trade_coast = self.trade_coast
        if spread is None:
            spread = self.spread
        if saving_plan is None:
            saving_plan = self.saving_plan
        if saving_plan_period is None:
            saving_plan_period = self.saving_plan_period
        if tax_rate is None:
            tax_rate = self.tax_rate
        if tax_allowance is None:
            tax_allowance = self.tax_allowance
        if data is None:
            data = self.performance
            
        trade_dates=np.array([0, self.time-1])
        #trade_dates = np.sort([-1, self.time])
            
        self.buy_and_hold_performance, self.buy_and_hold_transaction_cost, self.buy_and_hold_tax = self._compute_performance(data, trade_dates=trade_dates, trade_coast=trade_coast, spread=spread, saving_plan=saving_plan, saving_plan_period=saving_plan_period, tax_rate=tax_rate, tax_allowance=tax_allowance)

        return self.buy_and_hold_performance, self.buy_and_hold_transaction_cost, self.buy_and_hold_tax
    
    def _compute_performance(self, data, trade_dates, trade_coast, spread, saving_plan, saving_plan_period, tax_rate, tax_allowance, trade_coast_for_saving_plan=0, consider_loss_for_taxes=True):

        swing_performance = np.array([self.initial_investment])

        #data_gradient = np.gradient(data)
        data_gradient = data[1:] - data[:-1]

        trade_dates = np.sort(trade_dates)
        payed_tax = 0
        payed_transaction_cost = 0
        unused_tax_allowance = tax_allowance

        if type(saving_plan) == dict:
            saving_plan_dict = saving_plan
            saving_plan = saving_plan_dict[0]
        else:
            saving_plan_dict = None

        for i in range(self.time-1):

            if saving_plan_dict is not None: 
                if i/saving_plan_period in saving_plan_dict.keys(): # If we have a variable saving plan
                    saving_plan = saving_plan_dict[i/saving_plan_period]

            if i % self.length_of_year == 0: # Reset tax allowance after a year
                unused_tax_allowance = tax_allowance

            if swing_performance[-1] <= 0: # If we are broke
                swing_performance[-1] = 0
                swing_performance = np.append(swing_performance, np.zeros(self.time -1 -i))
                break
            elif np.sum(trade_dates <= i) % 2 == 1: # If we are in a trade or entering a trade
                if np.any(trade_dates == i): # If we are entering a trade
                    swing_performance = np.append(swing_performance, (swing_performance[-1]-trade_coast) * (1-spread))
                    value_at_last_trade = swing_performance[-1]
                    payed_transaction_cost += trade_coast + (swing_performance[-2]-trade_coast) * spread
                else: # If we are in a trade
                    swing_performance = np.append(swing_performance, swing_performance[-1] * (1 + data_gradient[i]/data[i]) )
                if saving_plan != 0 and i % saving_plan_period == 0 and i != 0: # If we have a saving plan
                    swing_performance[-1] = swing_performance[-1] + (saving_plan-trade_coast_for_saving_plan) * (1-spread) 
                    value_at_last_trade += (saving_plan-trade_coast_for_saving_plan) * (1-spread)
                    payed_transaction_cost += trade_coast_for_saving_plan + (saving_plan-trade_coast_for_saving_plan) * spread
            else:   # If we are not in a trade or exiting a trade
                if np.any(trade_dates == i): # If we are exiting a trade
                    swing_performance = np.append(swing_performance, swing_performance[-1] - trade_coast)
                    payed_transaction_cost += trade_coast
                    if tax_rate != 0 and (swing_performance[-1] > value_at_last_trade or consider_loss_for_taxes): # If we have to pay taxes and we made a profit or if we made a loss and the loss is offset against the tax allowance
                        taxable_profit = swing_performance[-1] - value_at_last_trade
                        if taxable_profit > unused_tax_allowance: # If thhe taxable profit exceeds the tax allowance
                            taxable_profit -= unused_tax_allowance
                            unused_tax_allowance = 0
                            tax = tax_rate * taxable_profit
                            swing_performance[-1] = swing_performance[-1] - tax  
                            payed_tax += tax       
                        else: # If the taxable profit is smaller than the tax allowance or the loss is offset against the tax allowance if a loss is considered for taxes
                            unused_tax_allowance -= taxable_profit
                
                else: # If we are not in a trade
                    swing_performance = np.append(swing_performance, swing_performance[-1])
                if saving_plan != 0 and i % saving_plan_period == 0 and i != 0:
                    swing_performance[-1] = swing_performance[-1] + saving_plan

        return swing_performance, payed_transaction_cost, payed_tax


    def _smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def internal_rate_of_return(self, performance=None, initial_investment=None, trade_dates=None, saving_plan=None, saving_plan_period=None, time=None, length_of_year=None, *args, **kwargs):
        
        if performance == 'buy_and_hold':
            performance = self.buy_and_hold_performance[-1]
        elif performance == 'swing_trade':
            performance = self.swing_performance_analyse[-1]
        elif performance == 'random_swing_trade':
            performance = self.random_swing_performance_analyse[-1]
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
        plt.plot(dates, self.buy_and_hold_performance, label="Buy and hold")
        plt.plot(dates, self.swing_performance_analyse, label="Swing trade analyse")
        plt.plot(dates, self.random_swing_performance_analyse, label="Random swing trade analyse")
        #plt.axhline(1, color="black", linestyle="--")   

        plt.xlabel("Time")
        plt.ylabel("Performance")

        plt.grid()
        plt.legend()

        if log:
            plt.yscale("log")

        plt.show()
    
    def print_results(self, accuracy=2, *args, **kwargs):

        print("Initial invetment: ", self.initial_investment)
        print("Total Investment: ", self.total_investment)
        print()

        absoulte_performance, relative_performance = round(self.performance[-1], accuracy), round(self.performance[-1]/self.initial_investment, accuracy)
        yearly_return = round((self.performance[-1]/self.initial_investment)**(self.length_of_year/self.time), accuracy)
        print(f"Index performance:") 
        print(f"    Absolute: {absoulte_performance}, Relative: {relative_performance}")
        print(f"    Yearly performance: {yearly_return}")
        print()

        absoulte_performance, relative_performance = round(self.buy_and_hold_performance[-1], accuracy), round(self.buy_and_hold_performance[-1]/self.total_investment, accuracy)
        yearly_return, internal_rate = round((self.buy_and_hold_performance[-1]/self.total_investment)**(self.length_of_year/self.time), accuracy), round(self.internal_rate_of_return('buy_and_hold'), accuracy)
        taxes, transaction_cost = round(np.sum(self.buy_and_hold_tax), accuracy), round(np.sum(self.buy_and_hold_transaction_cost), accuracy)
        print(f"Buy and hold return:")
        print(f"    Absolute: {absoulte_performance}, Relative: {relative_performance}")
        print(f"    Yearly performance: {yearly_return}, Internal rate of return: {internal_rate}")
        print(f"    Taxes: {taxes}, Transaction cost: {transaction_cost}")
        print()

        absoulte_performance, relative_performance = round(self.swing_performance_analyse[-1], accuracy), round(self.swing_performance_analyse[-1]/self.total_investment, accuracy)
        yearly_return, internal_rate = round((self.swing_performance_analyse[-1]/self.total_investment)**(self.length_of_year/self.time), accuracy), round(self.internal_rate_of_return('swing_trade'), accuracy)
        taxes, transaction_cost = round(np.sum(self.swing_tax), accuracy), round(np.sum(self.swing_transaction_cost), accuracy)
        print(f"Swing trade return:")
        print(f"    Absolute: {absoulte_performance}, Relative: {relative_performance}")
        print(f"    Yearly performance: {yearly_return}, Internal rate of return: {internal_rate}")
        print(f"    Taxes: {taxes}, Transaction cost: {transaction_cost}")
        print()

        absoulte_performance, relative_performance = round(self.random_swing_performance_analyse[-1], accuracy), round(self.random_swing_performance_analyse[-1]/self.total_investment, accuracy)
        yearly_return, internal_rate = round((self.random_swing_performance_analyse[-1]/self.total_investment)**(self.length_of_year/self.time), accuracy), round(self.internal_rate_of_return('random_swing_trade'), accuracy)
        taxes, transaction_cost = round(np.sum(self.random_swing_tax), accuracy), round(np.sum(self.random_swing_transaction_cost), accuracy)
        print(f"Random swing trade return:")
        print(f"    Absolute: {absoulte_performance}, Relative: {relative_performance}")
        print(f"    Yearly performance: {yearly_return}, Internal rate of return: {internal_rate}")
        print(f"    Taxes: {taxes}, Transaction cost: {transaction_cost}")
        print()

        if hasattr(self, 'daily_return'):
            print("Best return: ", round(self.performance[0] * self.daily_return**(np.sum(self.phase == 1)), accuracy))

    def print_parameters(self):
    
        print("General parameters: \n")
        print("Time: ", self.time)

        print("\nTrade parameters: \n")
        print("Max trades: ", self.max_trades)
        print("Smooth period", self.smooth_period)
        print("Hold time: ", self.hold_time)
        print("Time after reversel: ", self.time_after_reversel)
        print("Trade coast: ", self.trade_coast)
        print("Spread: ", self.spread)
        print("Initial invetment: ", self.initial_investment)
        print("Saving plan: ", self.saving_plan)
        print("Saving plan period: ", self.saving_plan_period)
        print("\n")




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

    def __init__(self, path="data/MSCI_World.csv", *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.import_data_df = None

        self.path = path

    def load_data(self, path=None, limit=None, normalize=True, *args, **kwargs):

        if limit is None:
            limit = slice(self.time)
        
        if path is None:
            path = self.path

        self.import_data_df = pd.read_csv(path)
        self.import_data_df = self.import_data_df
        self.import_data_df['Date'] = pd.to_datetime(self.import_data_df['Date'])

        if normalize:
            self.import_data_df['Value'] = self.import_data_df['Value'] * self.initial_investment / self.import_data_df['Value'].iloc[0]

        self.performance = self.import_data_df['Value'].to_numpy()[limit]
        self.dates = self.import_data_df['Date'].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_investment / self.performance[0]

        return  self.performance, self.dates
    
    def update_selection(self, limit=None, normalize=True, *args, **kwargs):

        if limit is None:
            limit = slice(self.time)

        self.performance = self.import_data_df['Value'].to_numpy()[limit]
        self.dates = self.import_data_df['Date'].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_investment / self.performance[0]

        return self.performance, self.dates
    
    def print_parameters(self):

        print("Data parameters: \n")
        print('path: ', self.path)
        print("\n")

        super().print_parameters()

def _parallel_sim_computation(i, sim):
    performance, _ = sim.simulate_performance()
    buy_and_hold_performance, buy_and_hold_transaction_cost, buy_and_hold_tax = sim.buy_and_hold(performance)
    random_swing_performance_analyse, random_swing_transaction_cost, random_swing_tax = sim.random_swing_trade(performance)
    swing_performance_analyse, swing_transaction_cost, swing_tax = sim.swing_trade(performance)

    return performance, buy_and_hold_performance, random_swing_performance_analyse, swing_performance_analyse, buy_and_hold_transaction_cost, buy_and_hold_tax, random_swing_transaction_cost, random_swing_tax, swing_transaction_cost, swing_tax

def _parallel_imp_computation(i, imp, stepsize):
    performance, _ = imp.update_selection(limit=slice(i*stepsize, imp.time + i*stepsize), normalize=True)
    buy_and_hold_performance, buy_and_hold_transaction_cost, buy_and_hold_tax = imp.buy_and_hold(performance)
    random_swing_performance_analyse, random_swing_transaction_cost, random_swing_tax = imp.random_swing_trade(performance)
    swing_performance_analyse, swing_transaction_cost, swing_tax = imp.swing_trade(performance)

    return performance, buy_and_hold_performance, random_swing_performance_analyse, swing_performance_analyse, buy_and_hold_transaction_cost, buy_and_hold_tax, random_swing_transaction_cost, random_swing_tax, swing_transaction_cost, swing_tax

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
        
        self.performance = np.zeros((n,  self.chartsim.time))
        self.buy_and_hold_performance = np.zeros((n, self.chartsim.time))
        self.random_swing_performance_analyse = np.zeros((n, self.chartsim.time))
        self.swing_performance_analyse = np.zeros((n, self.chartsim.time))

        self.buy_and_hold_transaction_cost, self.buy_and_hold_tax = np.zeros(n), np.zeros(n)
        self.random_swing_transaction_cost, self.random_swing_tax = np.zeros(n), np.zeros(n)
        self.swing_transaction_cost, self.swing_tax = np.zeros(n), np.zeros(n)

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(_parallel_sim_computation)(i, self.chartsim) for i in tqdm(range(n)))

            for i in range(n):
                (self.performance[i], 
                 self.buy_and_hold_performance[i], 
                 self.random_swing_performance_analyse[i], 
                 self.swing_performance_analyse[i], 
                 self.buy_and_hold_transaction_cost[i], 
                 self.buy_and_hold_tax[i], 
                 self.random_swing_transaction_cost[i], 
                 self.random_swing_tax[i], 
                 self.swing_transaction_cost[i], 
                 self.swing_tax[i]) = results[i]
        else:
            for i in tqdm(range(n)):
                self.performance[i], _ = self.chartsim.simulate_performance(**kwargs)
                self.buy_and_hold_performance[i], self.buy_and_hold_transaction_cost[i], self.buy_and_hold_tax[i] = self.chartsim.buy_and_hold(self.performance[i], **kwargs)
                self.random_swing_performance_analyse[i], self.random_swing_transaction_cost[i], self.random_swing_tax[i] = self.chartsim.random_swing_trade(self.performance[i], **kwargs)
                self.swing_performance_analyse[i], self.swing_transaction_cost[i], self.swing_tax[i] = self.chartsim.swing_trade(self.performance[i], **kwargs)

        self.index_performance = self.performance[:, -1]
        self.buy_and_hold_profit = self.buy_and_hold_performance[:, -1]
        self.random_swing_profit = self.random_swing_performance_analyse[:, -1]
        self.swing_profit = self.swing_performance_analyse[:, -1]

        return self.performance, self.buy_and_hold_profit, self.random_swing_performance_analyse, self.swing_performance_analyse
    
    def mc_import_chart(self, n=1000, stepsize=1, parallel=None, *args, **kwargs):

        if stepsize * n + self.chartimp.time > len(self.chartimp.import_data_df):
            raise ValueError(f"Stepsize * n + time is larger than the length of the data: {stepsize}, n: {n}, time: {self.chartimp.time}, data length: {len(self.chartimp.import_data_df)}")

        if self.chartimp is None:
            self.chartimp = ChartImport(**kwargs)
        
        if self.chartimp.import_data_df is None:
            self.chartimp.load_data(**kwargs)

        if parallel is None:
            parallel = self.parallel

        self.performance = np.zeros((n, self.chartimp.time))
        self.buy_and_hold_performance = np.zeros((n, self.chartimp.time))
        self.random_swing_performance_analyse = np.zeros((n, self.chartimp.time))
        self.swing_performance_analyse = np.zeros((n, self.chartimp.time))

        self.buy_and_hold_transaction_cost, self.buy_and_hold_tax = np.zeros(n), np.zeros(n)
        self.random_swing_transaction_cost, self.random_swing_tax = np.zeros(n), np.zeros(n)
        self.swing_transaction_cost, self.swing_tax = np.zeros(n), np.zeros(n)

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(_parallel_imp_computation)(i, self.chartimp, stepsize) for i in tqdm(range(n)))

            for i in range(n):
                (self.performance[i], 
                self.buy_and_hold_performance[i], 
                self.random_swing_performance_analyse[i], 
                self.swing_performance_analyse[i], 
                self.buy_and_hold_transaction_cost[i], 
                self.buy_and_hold_tax[i], 
                self.random_swing_transaction_cost[i], 
                self.random_swing_tax[i], 
                self.swing_transaction_cost[i], 
                self.swing_tax[i]) = results[i]
        
        else:
            for i in tqdm(range(n)):
                self.performance[i], _ = self.chartimp.update_selection(limit=slice(i*stepsize, self.chartimp.time + i*stepsize), normalize=True, **kwargs)
                self.buy_and_hold_performance[i], self.buy_and_hold_transaction_cost[i], self.buy_and_hold_tax[i] = self.chartimp.buy_and_hold(self.performance[i], **kwargs)
                self.random_swing_performance_analyse[i], self.random_swing_transaction_cost[i], self.random_swing_tax[i] = self.chartimp.random_swing_trade(self.performance[i], **kwargs)
                self.swing_performance_analyse[i], self.swing_transaction_cost[i], self.swing_tax[i] = self.chartimp.swing_trade(self.performance[i], **kwargs)

        self.index_performance = self.performance[:, -1]
        self.buy_and_hold_profit = self.buy_and_hold_performance[:, -1]
        self.random_swing_profit = self.random_swing_performance_analyse[:, -1]
        self.swing_profit = self.swing_performance_analyse[:, -1]

        return self.performance, self.buy_and_hold_profit, self.random_swing_performance_analyse, self.swing_performance_analyse

    
    def hist_performance(self, bins=50, limits=None, *args, **kwargs):

        if limits == 'minmax':
            limits = (min(np.min(self.index_performance), np.min(self.buy_and_hold_profit), np.min(self.random_swing_profit), np.min(self.swing_profit)), 
                      max(np.max(self.index_performance), np.max(self.buy_and_hold_profit), np.max(self.random_swing_profit), np.max(self.swing_profit)))

        plt.hist(self.index_performance, bins=bins, range=limits, alpha=0.5, label="Index Performance")
        plt.hist(self.buy_and_hold_profit, bins=bins, range=limits, alpha=0.5, label="Buy and hold performance")
        plt.hist(self.swing_profit, bins=bins, range=limits, alpha=0.5, label="Swing trade")
        plt.hist(self.random_swing_profit, bins=bins, range=limits, alpha=0.5, label="Random swing trade")

        plt.xlabel("Performance")
        plt.ylabel("Frequency")
        plt.title("Performance distribution")

        plt.grid()
        plt.legend()
        plt.show()

    def print_results(self, accuracy=2, *args, **kwargs):

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

        print("Initial invetment: ", initial_investment)
        print("Total money invested: ", total_investment)
        print()

        overall_return, overall_std, overall_median = round(self.index_performance.mean(), accuracy), round(self.index_performance.std(), accuracy), round(np.median(self.index_performance), accuracy)
        relative_performance, relative_std, relative_median = round(np.mean(self.index_performance/initial_investment), accuracy), round(np.std(self.index_performance/initial_investment), accuracy), round(np.median(self.index_performance/initial_investment), accuracy)
        yearly_return, yearly_std, yearly_median = round(np.mean((self.index_performance/initial_investment)**(length_of_year/time)), accuracy), round(np.std((self.index_performance/initial_investment)**(length_of_year/time)), accuracy), round(np.median((self.index_performance/initial_investment)**(length_of_year/time)), accuracy)
        print(f"Index performance:")
        print(f"  Overall return: {overall_return} +/- {overall_std} (Median: {overall_median})")
        print(f"  Relative performance: {relative_performance} +/- {relative_std} (Median: {relative_median})")
        print(f"  Yearly performance: {yearly_return} +/- {yearly_std} (Median: {yearly_median})")
        print()

        overall_return, overall_std, overall_median = round(self.buy_and_hold_profit.mean(), accuracy), round(self.buy_and_hold_profit.std(), accuracy), round(np.median(self.buy_and_hold_profit), accuracy)
        relative_performance, relative_std, relative_median = round(np.mean(self.buy_and_hold_profit/total_investment), accuracy), round(np.std(self.buy_and_hold_profit/total_investment), accuracy), round(np.median(self.buy_and_hold_profit/total_investment), accuracy)
        yearly_return, yearly_std, yearly_median = round(np.mean((self.buy_and_hold_profit/total_investment)**(length_of_year/time)), accuracy), round(np.std((self.buy_and_hold_profit/total_investment)**(length_of_year/time)), accuracy), round(np.median((self.buy_and_hold_profit/total_investment)**(length_of_year/time)), accuracy)
        internal_rates = [internal_rate_func(self.buy_and_hold_profit[i]) for i in range(len(self.buy_and_hold_profit))]
        internal_rate, internal_rate_std, internal_rate_median = round(np.mean(internal_rates), accuracy), round(np.std(internal_rates), accuracy), round(np.median(internal_rates), accuracy)
        taxes, taxes_std, taxes_median = round(np.mean(self.buy_and_hold_tax), accuracy), round(np.std(self.buy_and_hold_tax), accuracy), round(np.median(self.buy_and_hold_tax), accuracy)
        transaction_cost, transaction_cost_std, transaction_cost_median = round(np.mean(self.buy_and_hold_transaction_cost), accuracy), round(np.std(self.buy_and_hold_transaction_cost), accuracy), round(np.median(self.buy_and_hold_transaction_cost), accuracy) 
        print(f"Buy and hold return:") 
        print(f"  Overall return: {overall_return} +/- {overall_std} (Median: {overall_median})")
        print(f"  Relative performance: {relative_performance} +/- {relative_std} (Median: {relative_median})")
        print(f"  Yearly performance: {yearly_return} +/- {yearly_std} (Median: {yearly_median})")
        print(f"  Internal rate of return: {internal_rate} +/- {internal_rate_std} (Median: {internal_rate_median})")
        print(f"  Taxes: {taxes} +/- {taxes_std} (Median: {taxes_median})")
        print(f"  Transaction cost: {transaction_cost} +/- {transaction_cost_std} (Median: {transaction_cost_median})")
        print()

        overall_return, overall_std, overall_median = round(self.swing_profit.mean(), accuracy), round(self.swing_profit.std(), accuracy), round(np.median(self.swing_profit), accuracy)
        relative_performance, relative_std, relative_median = round(np.mean(self.swing_profit/total_investment), accuracy), round(np.std(self.swing_profit/total_investment), accuracy), round(np.median(self.swing_profit/total_investment), accuracy)
        yearly_return, yearly_std, yearly_median = round(np.mean((self.swing_profit/total_investment)**(length_of_year/time)), accuracy), round(np.std((self.swing_profit/total_investment)**(length_of_year/time)), accuracy), round(np.median((self.swing_profit/total_investment)**(length_of_year/time)), accuracy)
        internal_rates = [internal_rate_func(self.swing_profit[i]) for i in range(len(self.swing_profit))]
        internal_rate, internal_rate_std, internal_rate_median = round(np.mean(internal_rates), accuracy), round(np.std(internal_rates), accuracy), round(np.median(internal_rates), accuracy)
        taxes, taxes_std, taxes_median = round(np.mean(self.swing_tax), accuracy), round(np.std(self.swing_tax), accuracy), round(np.median(self.swing_tax), accuracy)
        transaction_cost, transaction_cost_std, transaction_cost_median = round(np.mean(self.swing_transaction_cost), accuracy), round(np.std(self.swing_transaction_cost), accuracy), round(np.median(self.swing_transaction_cost), accuracy)   
        print(f"Swing trade return:")
        print(f"  Overall return: {overall_return} +/- {overall_std} (Median: {overall_median})")
        print(f"  Relative performance: {relative_performance} +/- {relative_std} (Median: {relative_median})")
        print(f"  Yearly performance: {yearly_return} +/- {yearly_std} (Median: {yearly_median})")
        print(f"  Internal rate of return: {internal_rate} +/- {internal_rate_std} (Median: {internal_rate_median})")
        print(f"  Taxes: {taxes} +/- {taxes_std} (Median: {taxes_median})")
        print(f"  Transaction cost: {transaction_cost} +/- {transaction_cost_std} (Median: {transaction_cost_median})")
        print()

        overall_return, overall_std, overall_median = round(self.random_swing_profit.mean(), accuracy), round(self.random_swing_profit.std(), accuracy), round(np.median(self.random_swing_profit), accuracy)
        yearly_return, yearly_std, yearly_median = round(np.mean((self.random_swing_profit/total_investment)**(length_of_year/time)), accuracy), round(np.std((self.random_swing_profit/total_investment)**(length_of_year/time)), accuracy), round(np.median((self.random_swing_profit/total_investment)**(length_of_year/time)), accuracy)
        relative_performance, relative_std, relative_median = round(np.mean(self.random_swing_profit/total_investment), accuracy), round(np.std(self.random_swing_profit/total_investment), accuracy), round(np.median(self.random_swing_profit/total_investment), accuracy)
        internal_rates = [internal_rate_func(self.random_swing_profit[i]) for i in range(len(self.random_swing_profit))]
        internal_rate, internal_rate_std, internal_rate_median = round(np.mean(internal_rates), accuracy), round(np.std(internal_rates), accuracy), round(np.median(internal_rates), accuracy)
        taxes, taxes_std, taxes_median = round(np.mean(self.random_swing_tax), accuracy), round(np.std(self.random_swing_tax), accuracy), round(np.median(self.random_swing_tax), accuracy)
        transaction_cost, transaction_cost_std, transaction_cost_median = round(np.mean(self.random_swing_transaction_cost), accuracy), round(np.std(self.random_swing_transaction_cost), accuracy), round(np.median(self.random_swing_transaction_cost), accuracy)
        print(f"Random swing trade return:")
        print(f"  Overall return: {overall_return} +/- {overall_std} (Median: {overall_median})")
        print(f"  Relative performance: {relative_performance} +/- {relative_std} (Median: {relative_median})")
        print(f"  Yearly performance: {yearly_return} +/- {yearly_std} (Median: {yearly_median})")
        print(f"  Internal rate of return: {internal_rate} +/- {internal_rate_std} (Median: {internal_rate_median})")
        print(f"  Taxes: {taxes} +/- {taxes_std} (Median: {taxes_median})")
        print(f"  Transaction cost: {transaction_cost} +/- {transaction_cost_std} (Median: {transaction_cost_median})")
        print()


if __name__ == "__main__":

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
