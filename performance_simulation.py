import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

class PerformanceAnalyzer:

    def __init__(self, 
                 time=261, 
                 length_of_year=261,
                 initial_capital=1, 
                 saving_plan=0,
                 saving_plan_period=22,
                 set='simulation', 
                 smooth_period=5, 
                 max_trades=20,  
                 trades=5, 
                 hold_time=14, 
                 time_after_reversel=0, 
                 trade_coast=0, 
                 spread=0, 
                 tax_rate=0,
                 tax_allowance=0,
                 *args, **kwargs):
        
        self.time = time
        self.length_of_year = length_of_year

        self.initial_capital = initial_capital
        self.saving_plan = saving_plan
        self.saving_plan_period = saving_plan_period

        self.total_investment = initial_capital + saving_plan * time//saving_plan_period

        self.set = set
        self.smooth_period = smooth_period
        self.max_trades = max_trades
        self.trades = trades
        self.hold_time = hold_time
        self.time_after_reversel = time_after_reversel

        self.trade_coast = trade_coast
        self.spread = spread
        self.tax_rate = tax_rate
        self.tax_allowance = tax_allowance



    def random_swing_trade(self, phase=None, trades=5, trade_dates=None, *args, **kwargs):

        if phase is None:
            phase = self.phase

        if trade_dates is None:

            trade_dates = np.random.choice(np.arange(self.time), size=2*trades, replace=False)
            trade_dates = np.sort(trade_dates)

        swing_performance = np.array([self.initial_capital])

        for i in range(self.time):
            if np.sum(trade_dates <= i) % 2 == 1:
                swing_performance = np.append(swing_performance, swing_performance[-1] * self.daily_return if phase[i] == 1 else swing_performance[-1] * self.daily_loss)
            else:
                swing_performance = np.append(swing_performance, swing_performance[-1])
        
        # print('Random Swing Trade:', trade_dates)
        return swing_performance, trade_dates

    def swing_trade(self, phase=None, trade_dates=None, trades=20, hold_time=14, time_after_reversel=3, *args, **kwargs):

        if phase is None:
            phase = self.phase

        if trade_dates is None:
            trade_dates = np.array([])

            i = 0
            tr = 0
            while i in range(self.time) and tr < trades:

                if phase[i] == 1:
                    trade_dates = np.append(trade_dates, i + time_after_reversel)
                    trade_dates = np.append(trade_dates, i + time_after_reversel + hold_time)
                    i = i + time_after_reversel + hold_time
                    tr += 1
                else:
                    i += 1
            
        swing_performance = np.array([self.initial_capital])

        trade_dates = np.sort(trade_dates)
        for i in range(self.time):
            if np.sum(trade_dates <= i) % 2 == 1:
                swing_performance = np.append(swing_performance, swing_performance[-1] * self.daily_return if phase[i] == 1 else swing_performance[-1] * self.daily_loss)
            else:
                swing_performance = np.append(swing_performance, swing_performance[-1])
        
        # print('Swing Trade:', trade_dates)
        return swing_performance, trade_dates

    def random_swing_trade_ana(self, data=None, trade_dates=None, set=None, trades=None, trade_coast=None, spread=None, saving_plan=None, saving_plan_period=None, tax_rate=None, tax_allowance=None,
                               *args, **kwargs):

        if set is None:
            set = self.set
        if trades is None:
            trades = self.trades
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
        

        if data is None: # To be corrected
            if set == 'simulation':
                data = self.performance
            elif set == 'data':
                data = self.performance
            else:
                raise ValueError('Set must be either simulation or data')

        if trade_dates is None:

            trade_dates = np.random.choice(np.arange(self.time), size=2*trades, replace=False)
            trade_dates = np.sort(trade_dates)

        self.random_swing_performance_analyse = self._compute_swing_performance(data, trade_dates, trade_coast, spread, saving_plan, saving_plan_period, tax_rate, tax_allowance)
        
        # print('Random Swing Trade:', trade_dates)
        return self.random_swing_performance_analyse, trade_dates


    def swing_trade_ana(self, data=None, trade_dates=None, set=None, smooth_period=None, max_trades=None, hold_time=None, time_after_reversel=None, trade_coast=None, spread=None, saving_plan=None, saving_plan_period=None, tax_rate=None, tax_allowance=None,
                         *args, **kwargs):

        if set is None:
            set = self.set
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
        


        if data is None: # To be corrected
            if set == 'simulation':
                data = self.performance  
            elif set == 'data':
                data = self.performance
            else:
                raise ValueError('Set must be either simulation or data')

        data_smooth = self._smooth(data, smooth_period)
        data_trend = np.gradient(data_smooth)

        if trade_dates is None:

            trade_dates = np.array([0])

            i = 0
            while i in range(self.time) and len(trade_dates) < max_trades:

                if i > smooth_period/2 and i < self.time - smooth_period/2:
                    if data_trend[i] > 0 and len(trade_dates) % 2 == 0:
                        trade_dates = np.append(trade_dates, i + time_after_reversel)
                        i = i + hold_time
                    elif data_trend[i] < 0 and len(trade_dates) % 2 == 1:
                        trade_dates = np.append(trade_dates, i + time_after_reversel)
                        i = i + time_after_reversel
                    else:
                        i += 1
                else:
                    i += 1
            
        self.swing_performance_analyse = self._compute_swing_performance(data, trade_dates, trade_coast, spread, saving_plan, saving_plan_period, tax_rate, tax_allowance)
        
        # print('Swing Trade:', trade_dates)
        return self.swing_performance_analyse, trade_dates
    
    def buy_and_hold(self, data=None, set=None, trade_coast=None, spread=None, saving_plan=None, saving_plan_period=None, tax_rate=None, tax_allowance=None,
                     *args, **kwargs):

        if set is None:
            set = self.set
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
            if set == 'simulation':
                data = self.performance
            elif set == 'data':
                data = self.performance
            else:
                raise ValueError('Set must be either simulation or data')
            
        trade_dates=np.array([0, self.time-1])
        #trade_dates = np.sort([-1, self.time])
            
        self.buy_and_hold_performance = self._compute_swing_performance(data, trade_dates=trade_dates, trade_coast=trade_coast, spread=spread, saving_plan=saving_plan, saving_plan_period=saving_plan_period, tax_rate=tax_rate, tax_allowance=tax_allowance)

        return self.buy_and_hold_performance
    
    def _compute_swing_performance(self, data, trade_dates, trade_coast, spread, saving_plan, saving_plan_period, tax_rate, tax_allowance):

        swing_performance = np.array([self.initial_capital])

        #data_gradient = np.gradient(data)
        data_gradient = data[1:] - data[:-1]

        trade_dates = np.sort(trade_dates)
        for i in range(self.time-1):
            if i % self.length_of_year == 0:
                unused_tax_allowance = tax_allowance

            if swing_performance[-1] <= 0:
                swing_performance[-1] = 0
                swing_performance = np.append(swing_performance, np.zeros(self.time -1 -i))
                break
            elif np.sum(trade_dates <= i) % 2 == 1:
                if np.any(trade_dates == i):
                    swing_performance = np.append(swing_performance, (swing_performance[-1]-trade_coast) * (1-spread))
                    value_at_last_trade = swing_performance[-1]
                else:
                    swing_performance = np.append(swing_performance, swing_performance[-1] * (1 + data_gradient[i]/data[i]) )
                if saving_plan != 0 and i % saving_plan_period == 0 and i != 0:
                    swing_performance[-1] = swing_performance[-1] + (saving_plan-trade_coast) * (1-spread) 
                    value_at_last_trade += (saving_plan-trade_coast) * (1-spread)
            else:
                if np.any(trade_dates == i):
                    swing_performance = np.append(swing_performance, swing_performance[-1] - trade_coast)
                    if tax_rate != 0 and swing_performance[-1] > value_at_last_trade:
                        taxable_profit = swing_performance[-1] - value_at_last_trade
                        if taxable_profit > unused_tax_allowance:
                            taxable_profit -= unused_tax_allowance
                            unused_tax_allowance = 0
                            tax = tax_rate * taxable_profit
                            swing_performance[-1] = swing_performance[-1] - tax         
                        else:
                            unused_tax_allowance -= taxable_profit
                
                else:
                    swing_performance = np.append(swing_performance, swing_performance[-1])
                if saving_plan != 0 and i % saving_plan_period == 0 and i != 0:
                    swing_performance[-1] = swing_performance[-1] + saving_plan

        return swing_performance


    def _smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def plot_performance(self, log=False, *args, **kwargs):
        
        if hasattr(self, 'dates'):
            dates = self.dates
        else:
            dates = np.arange(self.time)

        plt.plot(dates, self.performance, label="Performance")
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

        print("Total money invested: ", self.total_investment)

        print(f"Index performance: Absolute: {round(self.performance[-1], accuracy)}, Relative: {round(self.performance[-1]/self.total_investment, accuracy)}")
        print(f"Buy and hold return: Absolute: {round(self.buy_and_hold_performance[-1], accuracy)}, Relative: {round(self.buy_and_hold_performance[-1]/self.total_investment, accuracy)}")
        print(f"Swing trade return: Absolute: {round(self.swing_performance_analyse[-1], accuracy)}, Relative: {round(self.swing_performance_analyse[-1]/self.total_investment, accuracy)}")
        print(f"Random swing trade return: Absolute: {round(self.random_swing_performance_analyse[-1], accuracy)}, Relative: {round(self.random_swing_performance_analyse[-1]/self.total_investment, accuracy)}")
        if hasattr(self, 'daily_return'):
            print("Best return: ", round(self.performance[0] * self.daily_return**(np.sum(self.phase == 1)), accuracy))

    def print_parameters(self):
    
        print("General parameters: \n")
        print("Time: ", self.time)
        print("Set: ", self.set)

        print("\nTrade parameters: \n")
        print("Max trades: ", self.max_trades)
        print("Trades: ", self.trades)
        print("Smooth period", self.smooth_period)
        print("Hold time: ", self.hold_time)
        print("Time after reversel: ", self.time_after_reversel)
        print("Trade coast: ", self.trade_coast)
        print("Spread: ", self.spread)
        print("Initial capital: ", self.initial_capital)
        print("Saving plan: ", self.saving_plan)
        print("Saving plan period: ", self.saving_plan_period)
        print("\n")




class ChartSimulation(PerformanceAnalyzer):

        def __init__(self,dt=15, yearly_return=1.07, daily_return=1.001, daily_loss=0.999, gain_phase=0.7, loss_phase=0.3, mode="constant_timesteps", *args, **kwargs):
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

            if self.mode == "constant_timesteps":
                self.daily_return = self.yearly_return**(1/self.length_of_year/(2*self.gain_phase-1))
                self.daily_loss = 1/self.daily_return

            elif self.mode == "constant_gain":
                self.gain_phase = np.log(self.yearly_return**(1/self.length_of_year)/self.daily_loss) / np.log(self.daily_return/self.daily_loss)
                self.loss_phase = 1 - self.gain_phase 

            self.expected_total_return = self.daily_return**(self.gain_phase * self.time) * self.daily_loss**(self.loss_phase * self.time)
            
            performance = np.array([self.initial_capital])
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
            self.import_data_df['Value'] = self.import_data_df['Value'] * self.initial_capital / self.import_data_df['Value'].iloc[0]

        self.performance = self.import_data_df['Value'].to_numpy()[limit]
        self.dates = self.import_data_df['Date'].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_capital / self.performance[0]

        return  self.performance, self.dates
    
    def update_selection(self, limit=None, normalize=True, *args, **kwargs):

        if limit is None:
            limit = slice(self.time)

        self.performance = self.import_data_df['Value'].to_numpy()[limit]
        self.dates = self.import_data_df['Date'].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_capital / self.performance[0]

        return self.performance, self.dates
    
    def print_parameters(self):

        print("Data parameters: \n")
        print('path: ', self.path)
        print("\n")

        super().print_parameters()

def _parallel_sim_computation(i, sim):
    performance, _ = sim.simulate_performance()
    buy_and_hold_performance = sim.buy_and_hold(performance)
    random_swing_performance_analyse, _ = sim.random_swing_trade_ana(performance)
    swing_performance_analyse, _ = sim.swing_trade_ana(performance)

    return performance, buy_and_hold_performance, random_swing_performance_analyse, swing_performance_analyse

def _parallel_imp_computation(i, imp, stepsize):
    performance, _ = imp.update_selection(limit=slice(i*stepsize, imp.time + i*stepsize), normalize=True)
    buy_and_hold_performance = imp.buy_and_hold(performance)
    random_swing_performance_analyse, _ = imp.random_swing_trade_ana(performance)
    swing_performance_analyse, _ = imp.swing_trade_ana(performance)

    return performance, buy_and_hold_performance, random_swing_performance_analyse, swing_performance_analyse

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

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(_parallel_sim_computation)(i, self.chartsim) for i in tqdm(range(n)))

            for i in range(n):
                self.performance[i], self.buy_and_hold_performance[i], self.random_swing_performance_analyse[i], self.swing_performance_analyse[i] = results[i]
        else:
            for i in tqdm(range(n)):
                self.performance[i], _ = self.chartsim.simulate_performance(**kwargs)
                self.buy_and_hold_performance[i] = self.chartsim.buy_and_hold(self.performance[i], **kwargs)
                self.random_swing_performance_analyse[i], _ = self.chartsim.random_swing_trade_ana(self.performance[i], **kwargs)
                self.swing_performance_analyse[i], _ = self.chartsim.swing_trade_ana(self.performance[i], **kwargs)

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

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(_parallel_imp_computation)(i, self.chartimp, stepsize) for i in tqdm(range(n)))

            for i in range(n):
                self.performance[i], self.buy_and_hold_performance[i], self.random_swing_performance_analyse[i], self.swing_performance_analyse[i] = results[i]
        
        else:
            for i in tqdm(range(n)):
                self.performance[i], _ = self.chartimp.update_selection(limit=slice(i*stepsize, self.chartimp.time + i*stepsize), normalize=True, **kwargs)
                self.buy_and_hold_performance[i] = self.chartimp.buy_and_hold(self.performance[i], **kwargs)
                self.random_swing_performance_analyse[i], _ = self.chartimp.random_swing_trade_ana(self.performance[i], **kwargs)
                self.swing_performance_analyse[i], _ = self.chartimp.swing_trade_ana(self.performance[i], **kwargs)

        self.index_performance = self.performance[:, -1]
        self.buy_and_hold_profit = self.buy_and_hold_performance[:, -1]
        self.random_swing_profit = self.random_swing_performance_analyse[:, -1]
        self.swing_profit = self.swing_performance_analyse[:, -1]

        return self.performance, self.buy_and_hold_profit, self.random_swing_performance_analyse, self.swing_performance_analyse

    
    def hist_performance(self, bins=50, limits=None, *args, **kwargs):

        if limits is None:
            limits = (min(np.min(self.index_performance), np.min(self.buy_and_hold_profit), np.min(self.random_swing_profit), np.min(self.swing_profit)), 
                      max(np.max(self.index_performance), np.max(self.buy_and_hold_profit), np.max(self.random_swing_profit), np.max(self.swing_profit)))

        plt.hist(self.index_performance, bins=bins, range=limits, alpha=0.5, label="Index Performance")
        plt.hist(self.buy_and_hold_profit, bins=bins, range=limits, alpha=0.5, label="Buy and hold performance")
        plt.hist(self.random_swing_profit, bins=bins, range=limits, alpha=0.5, label="Random swing trade")
        plt.hist(self.swing_profit, bins=bins, range=limits, alpha=0.5, label="Swing trade")

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
            # print(f"Parameters of {self.chartsim.__class__.__name__}:\n")
            # self.chartsim.print_parameters()
            # print("\n")
        if not self.chartimp is None:
            time = self.chartimp.time
            length_of_year = self.chartimp.length_of_year
            # print(f"Parameters of {self.chartimp.__class__.__name__}: \n")
            # self.chartimp.print_parameters()
            # print("\n")

        print(f"Index performance:")
        print(f"  Overall return: {round(self.index_performance.mean(), accuracy)} +/- {round(self.index_performance.std(), accuracy)} (Median: {round(np.median(self.index_performance), accuracy)})")
        print(f"  Yearly return: {round(np.mean(self.index_performance**(length_of_year/time)), accuracy)} +/- {round(np.std(self.index_performance**(length_of_year/time)), accuracy)} (Median: {round(np.median(self.index_performance**(length_of_year/time)), accuracy)}) \n")

        print(f"Buy and hold return:") 
        print(f"  Overall return: {round(self.buy_and_hold_profit.mean(), accuracy)} +/- {round(self.buy_and_hold_profit.std(), accuracy)} (Median: {round(np.median(self.buy_and_hold_profit), accuracy)})")
        print(f"  Yearly return: {round(np.mean(self.buy_and_hold_profit**(length_of_year/time)), accuracy)} +/- {round(np.std(self.buy_and_hold_profit**(length_of_year/time)), accuracy)} (Median: {round(np.median(self.buy_and_hold_profit**(length_of_year/time)), accuracy)}) \n")

        print(f"Random swing trade return analyse:")
        print(f"  Overall return: {round(self.random_swing_profit.mean(), accuracy)} +/- {round(self.random_swing_profit.std(), accuracy)} (Median: {round(np.median(self.random_swing_profit), accuracy)})")
        print(f"  Yearly return: {round(np.mean(self.random_swing_profit**(length_of_year/time)), accuracy)} +/- {round(np.std(self.random_swing_profit**(length_of_year/time)), accuracy)} (Median: {round(np.median(self.random_swing_profit**(length_of_year/time)), accuracy)}) \n")

        print(f"Swing trade return analyse:")
        print(f"  Overall return: {round(self.swing_profit.mean(), accuracy)} +/- {round(self.swing_profit.std(), accuracy)} (Median: {round(np.median(self.swing_profit), accuracy)})")
        print(f"  Yearly return: {round(np.mean(self.swing_profit**(length_of_year/time)), accuracy)} +/- {round(np.std(self.swing_profit**(length_of_year/time)), accuracy)} (Median: {round(np.median(self.swing_profit**(length_of_year/time)), accuracy)}) \n")


if __name__ == "__main__":

    mc = MonteCarloSimulation()
    mc.mc_artificial_chart(n=500, parallel=True)

    mc.hist_performance(bins=50)
    mc.print_results()
