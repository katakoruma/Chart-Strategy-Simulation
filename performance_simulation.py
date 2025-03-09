import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

yearly_return = 1.07

daily_return = 1.001
daily_loss = 1 - 0.01


gain_phase = 0.7
loss_phase = 1 - gain_phase

dt = 15
time = 261

mode = "constant_timesteps"
#mode = "constant_gain"



class PerformanceAnalyzer:

    def __init__(self, time=261, dt=15):
        self.time = time
        self.dt = dt

    def random_swing_trade(self, phase, trades=5, trade_dates=None):
        if trade_dates is None:

            trade_dates = np.random.choice(np.arange(self.time), size=2*trades, replace=False)
            trade_dates = np.sort(trade_dates)

        swing_performance = np.array([1])

        for i in range(self.time):
            if np.sum(trade_dates <= i) % 2 == 1:
                swing_performance = np.append(swing_performance, swing_performance[-1] * self.daily_return if phase[i] == 1 else swing_performance[-1] * self.daily_loss)
            else:
                swing_performance = np.append(swing_performance, swing_performance[-1])
        
        print('Random Swing Trade:', trade_dates)
        return swing_performance, trade_dates

    def swing_trade(self, phase, trades=20, hold_time=14, time_after_reversel=3, trade_dates=None):

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
            
        swing_performance = np.array([1])

        trade_dates = np.sort(trade_dates)
        for i in range(self.time):
            if np.sum(trade_dates <= i) % 2 == 1:
                swing_performance = np.append(swing_performance, swing_performance[-1] * self.daily_return if phase[i] == 1 else swing_performance[-1] * self.daily_loss)
            else:
                swing_performance = np.append(swing_performance, swing_performance[-1])
        
        print('Swing Trade:', trade_dates)
        return swing_performance, trade_dates

    def random_swing_trade_ana(self, data=None, set='simulation', trades=5, trade_dates=None):

        if data is None:
            if set == 'simulation':
                data = self.performance
            elif set == 'data':
                data = self.import_data_np
            else:
                raise ValueError('Set must be either simulation or data')
        
        data_gradient = np.gradient(data)

        if trade_dates is None:

            trade_dates = np.random.choice(np.arange(self.time), size=2*trades, replace=False)
            trade_dates = np.sort(trade_dates)

        swing_performance = np.array([1])

        for i in range(self.time):
            if np.sum(trade_dates <= i) % 2 == 1:
                swing_performance = np.append(swing_performance, swing_performance[-1] * (1+ data_gradient[i] / data[i]) )
            else:
                swing_performance = np.append(swing_performance, swing_performance[-1])
        
        print('Random Swing Trade:', trade_dates)
        return swing_performance, trade_dates


    def swing_trade_ana(self, data=None, set='simulation', smooth_period=5, trades=20, hold_time=14, time_after_reversel=0, trade_dates=None):

        if data is None:
            if set == 'simulation':
                data = self.performance
            elif set == 'data':
                data = self.import_data_np
            else:
                raise ValueError('Set must be either simulation or data')

        data_smooth = self._smooth(data, smooth_period)
        data_trend = np.gradient(data_smooth)
        data_gradient = np.gradient(data)

        if trade_dates is None:

            trade_dates = np.array([])

            i = 0
            tr = 0
            while i in range(self.time) and tr < trades:

                if data_trend[i] > 0:
                    trade_dates = np.append(trade_dates, i + time_after_reversel)
                    trade_dates = np.append(trade_dates, i + time_after_reversel + hold_time)
                    i = i + time_after_reversel + hold_time
                    tr += 1
                else:
                    i += 1
            
        swing_performance = np.array([1])

        trade_dates = np.sort(trade_dates)
        for i in range(self.time):
            if np.sum(trade_dates <= i) % 2 == 1:
                swing_performance = np.append(swing_performance, swing_performance[-1] * (1+ data_gradient[i] / data[i]) )
            else:
                swing_performance = np.append(swing_performance, swing_performance[-1])
        
        print('Swing Trade:', trade_dates)
        return swing_performance, trade_dates


    def _smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth


class ChartSimulation(PerformanceAnalyzer):

        def __init__(self, yearly_return, daily_return, daily_loss, gain_phase, loss_phase, mode, *args, **kwargs):
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

        def simulate_performance(self):
            if self.mode == "constant_timesteps":
                self.daily_return = self.yearly_return**(1/self.time/(2*self.gain_phase-1)) 
                self.daily_loss = 1/self.daily_return
                print("Daily return: ", self.daily_return)

            elif self.mode == "constant_gain":
                self.gain_phase = np.log(self.yearly_return**(1/self.time)/self.daily_loss) / np.log(self.daily_return/self.daily_loss)
                self.loss_phase = 1 -  self.gain_phase 
                print("Gain phase: ",  self.gain_phase )

            yearly_return = self.daily_return**(self.gain_phase * self.time) * self.daily_loss**(self.loss_phase * self.time)
            print("Yearly return: ", yearly_return)

            performance = np.array([1])
            phase = np.zeros(self.time)

            rnd = np.random.choice([0, 1], p=[self.loss_phase, self.gain_phase], size=self.time//self.dt)

            for i in range(self.time//self.dt):
                phase[i*self.dt:max((i+1)*self.dt, self.time)] = rnd[i]

            for i in range(self.time):
                performance = np.append(performance, performance[-1] * self.daily_return if phase[i] == 1 else performance[-1] * self.daily_loss)

            self.performance = performance
            self.phase = phase

            return performance, phase

class ChartImport(PerformanceAnalyzer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, path="990100 - MSCI World Index.csv-2.csv", limit=None, normalize=True):

        if limit is None:
            limit = slice(self.time)

        self.import_data_df = pd.read_csv(path)
        self.import_data_df = self.import_data_df[limit]

        if normalize:
            self.import_data_df['Value'] = self.import_data_df['Value'] / self.import_data_df['Value'].iloc[0]

        self.import_data_np = self.import_data_df['Value'].to_numpy()

        return self.import_data_df




if __name__ == "__main__":

    sim = ChartSimulation(yearly_return, daily_return, daily_loss, gain_phase, loss_phase, mode, time, dt)

    performance, phase = sim.simulate_performance()

    random_swing_performance_analyse, trade_dates_random  = sim.random_swing_trade_ana(performance, trades=5, trade_dates=None)
    swing_performance_analyse, trade_dates = sim.swing_trade_ana(performance, smooth_period=1, trades=20, hold_time=14, time_after_reversel=0, trade_dates=None)

    print("Buy and hold return: ", performance[-1])
    print("Random swing trade return analyse: ", random_swing_performance_analyse[-1])
    print("Swing trade return analyse: ", swing_performance_analyse[-1])
    print("Best return: ", performance[0] * daily_return**(np.sum(phase == 1)) ) 

    chart = ChartImport(time, dt)
    chart.load_data()
    data = chart.import_data_np

    plt.plot(data, label="Buy and hold")
    #plt.plot(swing_performance_analyse, label="Swing trade analyse")
    #plt.plot(random_swing_performance_analyse, label="Random swing trade analyse")
    plt.axhline(1, color="black", linestyle="--")   

    plt.xlabel("Time")
    plt.ylabel("Performance")

    plt.grid()
    plt.legend()

    plt.show()
