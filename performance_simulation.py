import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import datetime
from tqdm import tqdm

class PerformanceAnalyzer:

    def __init__(self, time=261, dt=15, initial_capital=1, set='simulation', smooth_period=5, max_trades=20,  trades=5, hold_time=14, time_after_reversel=0, trade_coast=0, spread=0, *args, **kwargs):
        self.initial_capital = initial_capital
        self.time = time
        self.dt = dt

        self.set = set
        self.smooth_period = smooth_period
        self.max_trades = max_trades
        self.trades = trades
        self.hold_time = hold_time
        self.time_after_reversel = time_after_reversel
        self.trade_coast = trade_coast
        self.spread = spread



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

    def random_swing_trade_ana(self, data=None, trade_dates=None, set=None, trades=None, trade_coast=None, spread=None, *args, **kwargs):

        if set is None:
            set = self.set
        if trades is None:
            trades = self.trades
        if trade_coast is None:
            trade_coast = self.trade_coast
        if spread is None:
            spread = self.spread
        

        if data is None: # To be corrected
            if set == 'simulation':
                data = self.performance
            elif set == 'data':
                data = self.performance
            else:
                raise ValueError('Set must be either simulation or data')
        
        data_gradient = np.gradient(data)

        if trade_dates is None:

            trade_dates = np.random.choice(np.arange(self.time), size=2*trades, replace=False)
            trade_dates = np.sort(trade_dates)

        self.random_swing_performance_analyse = self._compute_swing_performance(data, trade_dates, trade_coast, spread)
        
        # print('Random Swing Trade:', trade_dates)
        return self.random_swing_performance_analyse, trade_dates


    def swing_trade_ana(self, data=None, trade_dates=None, set=None, smooth_period=None, max_trades=None, hold_time=None, time_after_reversel=None, trade_coast=None, spread=None, *args, **kwargs):

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

            trade_dates = np.array([])

            i = 0
            tr = 0
            while i in range(self.time) and tr < max_trades:

                if data_trend[i] > 0 and i > smooth_period/2 and i < self.time - smooth_period/2:
                    trade_dates = np.append(trade_dates, i + time_after_reversel)
                    trade_dates = np.append(trade_dates, i + time_after_reversel + hold_time)
                    i = i + time_after_reversel + hold_time
                    tr += 1
                else:
                    i += 1
            
        self.swing_performance_analyse = self._compute_swing_performance(data, trade_dates, trade_coast, spread)
        
        # print('Swing Trade:', trade_dates)
        return self.swing_performance_analyse, trade_dates
    
    def _compute_swing_performance(self, data, trade_dates, trade_coast, spread):

        swing_performance = np.array([self.initial_capital])
        data_gradient = np.gradient(data)

        trade_dates = np.sort(trade_dates)
        for i in range(self.time):
            if swing_performance[-1] <= 0:
                swing_performance[-1] = 0
                swing_performance = np.append(swing_performance, np.zeros(self.time - i))
                break
            elif np.sum(trade_dates <= i) % 2 == 1:
                if np.any(trade_dates == i):
                    swing_performance = np.append(swing_performance, swing_performance[-1] * (1-spread) - trade_coast)
                else:
                    swing_performance = np.append(swing_performance, swing_performance[-1] * (1 + data_gradient[i]/data[i]) )
            else:
                if np.any(trade_dates == i):
                    swing_performance = np.append(swing_performance, swing_performance[-1] - trade_coast)
                else:
                    swing_performance = np.append(swing_performance, swing_performance[-1])

        return swing_performance


    def _smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth


class ChartSimulation(PerformanceAnalyzer):

        def __init__(self, yearly_return=1.07, daily_return=1.001, daily_loss=0.999, gain_phase=0.7, loss_phase=0.3, mode="constant_timesteps", *args, **kwargs):
            # Call the parent class's __init__ method to initialize inherited attributes
            super().__init__(*args, **kwargs)
            
            # Initialize additional attributes specific to ChartSimulation
            self.yearly_return = yearly_return
            self.daily_return = daily_return
            self.daily_loss = daily_loss
            self.gain_phase = gain_phase
            self.loss_phase = loss_phase
            self.mode = mode
            # Additional initialization logic for ChartSimulation

        def simulate_performance(self, *args, **kwargs):
            if self.mode == "constant_timesteps":
                self.daily_return = self.yearly_return**(1/self.time/(2*self.gain_phase-1)) 
                self.daily_loss = 1/self.daily_return

            elif self.mode == "constant_gain":
                self.gain_phase = np.log(self.yearly_return**(1/self.time)/self.daily_loss) / np.log(self.daily_return/self.daily_loss)
                self.loss_phase = 1 -  self.gain_phase 

            yearly_return = self.daily_return**(self.gain_phase * self.time) * self.daily_loss**(self.loss_phase * self.time)
            
            performance = np.array([self.initial_capital])
            phase = np.zeros(self.time)

            rnd = np.random.choice([0, 1], p=[self.loss_phase, self.gain_phase], size=self.time//self.dt)

            for i in range(self.time//self.dt):
                phase[i*self.dt:max((i+1)*self.dt, self.time)] = rnd[i]

            for i in range(self.time):
                performance = np.append(performance, performance[-1] * self.daily_return if phase[i] == 1 else performance[-1] * self.daily_loss)

            self.performance = performance
            self.phase = phase

            return performance, phase
        
        def print_parameters(self):

            print("Simulation parameters: \n")
            print("Yearly return: ", self.yearly_return)
            print("Daily return: ", self.daily_return)
            print("Daily loss: ", self.daily_loss)
            print("Gain phase: ", self.gain_phase)
            print("Loss phase: ", self.loss_phase)
            print("\n")

            print("Swing trade parameters: \n")
            print("Max trades: ", self.max_trades)
            print("Trades: ", self.trades)
            print("Hold time: ", self.hold_time)
            print("Time after reversel: ", self.time_after_reversel)
            print("Trade coast: ", self.trade_coast)
            print("Spread: ", self.spread)
            print("\n")
        
        def print_results(self):

            print("Buy and hold return: ", self.performance[-1])
            print("Random swing trade return analyse: ", self.random_swing_performance_analyse[-1])
            print("Swing trade return analyse: ", self.swing_performance_analyse[-1])
            print("Best return: ", self.performance[0] * self.daily_return**(np.sum(self.phase == 1)) )

class ChartImport(PerformanceAnalyzer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, path="990100 - MSCI World Index.csv-2.csv", limit=None, normalize=True, *args, **kwargs):

        if limit is None:
            limit = slice(self.time)

        self.import_data_df = pd.read_csv(path)
        self.import_data_df = self.import_data_df
        self.import_data_df['Date'] = pd.to_datetime(self.import_data_df['Date'])

        if normalize:
            self.import_data_df['Value'] = self.import_data_df['Value'] * self.initial_capital / self.import_data_df['Value'].iloc[0]

        self.performance = self.import_data_df['Value'].to_numpy()[limit]
        self.dates = self.import_data_df['Date'].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_capital / self.performance[0]

        return self.import_data_df
    
    def roll_data(self, shift=1, normalize=True, *args, **kwargs):

        self.performance = np.roll(self.performance, -shift)
        self.dates = np.roll(self.dates, -shift)

        if normalize:
            self.performance = self.performance * self.initial_capital / self.performance[0]

        return self.performance, self.dates
    
    def update_selection(self, limit=None, normalize=True, *args, **kwargs):

        if limit is None:
            limit = slice(self.time)

        self.performance = self.import_data_df['Value'].to_numpy()[limit]
        self.dates = self.import_data_df['Date'].to_numpy()[limit]

        if normalize:
            self.performance = self.performance * self.initial_capital / self.performance[0]

        return self.performance, self.dates

class MonteCarloSimulation:

    def __init__(self, chartsim=None, chartimp=None, *args, **kwargs):
        self.chartsim = chartsim
        self.chartimp = chartimp

    def mc_artificial_chart(self, n=1000, *args, **kwargs):

        if self.chartsim is None:
            self.chartsim = ChartSimulation(**kwargs)
        
        self.performance = np.zeros((n,  self.chartsim.time+1))
        self.random_swing_performance_analyse = np.zeros((n, self.chartsim.time+1))
        self.swing_performance_analyse = np.zeros((n, self.chartsim.time+1))

        for i in tqdm(range(n)):
            self.performance[i], _ = self.chartsim.simulate_performance()
            self.random_swing_performance_analyse[i], _ = self.chartsim.random_swing_trade_ana(self.performance[i], **kwargs)
            self.swing_performance_analyse[i], _ = self.chartsim.swing_trade_ana(self.performance[i], **kwargs)

        self.profit = self.performance[:, -1]
        self.random_swing_profit = self.random_swing_performance_analyse[:, -1]
        self.swing_profit = self.swing_performance_analyse[:, -1]

        return self.performance, self.random_swing_performance_analyse, self.swing_performance_analyse
    
    def mc_import_chart(self, n=1000, time=261, dt=15, path="990100 - MSCI World Index.csv-2.csv", limit=None, normalize=True, *args, **kwargs):

        performance = np.zeros((n, time+1))
        phase = np.zeros((n, time))

        for i in range(n):
            self.chartimp = ChartImport(time=time, dt=dt)
            self.chartimp.load_data(path=path, limit=limit, normalize=normalize)
            performance[i] = sim.performance
            phase[i] = sim.phase

        return performance, phase
    
    def hist_performance(self, bins=50, limits=None, *args, **kwargs):

        if limits is None:
            limits = (min(np.min(self.profit), np.min(self.random_swing_profit), np.min(self.swing_profit)), max(np.max(self.profit), np.max(self.random_swing_profit), np.max(self.swing_profit)))

        plt.hist(self.profit, bins=bins, range=limits, alpha=0.5, label="Buy and hold")
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
            print(f"Parameters of {self.chartsim.__class__.__name__}:\n")
            self.chartsim.print_parameters()
            print("\n")
        if not self.chartimp is None:
            print(f"Parameters of {self.chartimp.__class__.__name__}: \n")
            self.chartimp.print_parameters()
            print("\n")

        print(f"Buy and hold return: {round(self.profit.mean(), accuracy)} +/- {round(self.profit.std(), accuracy)}")
        print(f"Random swing trade return: {round(self.random_swing_profit.mean(), accuracy)} +/- {round(self.random_swing_profit.std(), accuracy)}")
        print(f"Swing trade return: {round(self.swing_profit.mean(), accuracy)} +/- {round(self.swing_profit.std(), accuracy)}")



if __name__ == "__main__":

    sim = ChartSimulation()

    performance, phase = sim.simulate_performance()

    random_swing_performance_analyse, trade_dates_random  = sim.random_swing_trade_ana()
    swing_performance_analyse, trade_dates = sim.swing_trade_ana()

    print("Buy and hold return: ", performance[-1])
    print("Random swing trade return analyse: ", random_swing_performance_analyse[-1])
    print("Swing trade return analyse: ", swing_performance_analyse[-1])
    print("Best return: ", performance[0] * sim.daily_return**(np.sum(phase == 1)) ) 

    plt.plot(performance, label="Buy and hold")
    plt.plot(swing_performance_analyse, label="Swing trade analyse")
    plt.plot(random_swing_performance_analyse, label="Random swing trade analyse")
    plt.axhline(1, color="black", linestyle="--")   

    plt.xlabel("Time")
    plt.ylabel("Performance")

    plt.grid()
    plt.legend()

    plt.show()
