# Performance Simulation Project

This project is focused on simulating the performance of trading strategies over a given period. It includes various methods to simulate and analyze trading performance, including constant timesteps, constant gain, and swing trading strategies.

## Key Components

### performance_simulation.py

This script contains the `PerformanceAnalyzer`, `ChartSimulation`, and `ChartImport` classes, which are used to simulate and analyze trading performance. Key methods include:
- `simulate_performance`: Simulates the performance based on the specified mode (e.g., fixed gain phase or fixed return).
- `random_swing_trade`: Simulates random swing trades.
- `swing_trade`: Simulates swing trades based on trends and gradients in the data.
- `buy_and_hold`: Simulates a buy-and-hold strategy for the entire investment period.
- `load_data`: Loads and optionally normalizes historical data from a CSV file or multiple sources.
- `internal_rate_of_return`: Calculates the internal rate of return (IRR) for a given performance.
- `plot_performance`: Visualizes the performance of various strategies over time.
- `print_results`: Summarizes the results of the simulations in a structured format.

### msci_compare.py

This script compares the performance of the simulated trading strategies with historical data from the MSCI World Index. It uses the `PerformanceAnalyzer` class to load data and perform the analysis.

### tests.py

This script contains unit tests for the `PerformanceAnalyzer` class using the `pytest` framework. It includes tests for:
- Simulating performance
- Random swing trades
- Swing trades
- Buy-and-hold strategy
- Loading data
- Calculating internal rate of return
- Plotting performance

### Trading_Rendite.py

This script calculates the trading returns based on a simple trading strategy. It simulates buying and selling an instrument over a specified number of iterations and calculates the final account balance, total costs, and end value of the instrument.

### MonteCarloSimulation

The `MonteCarloSimulation` class enables running multiple simulations in parallel for both artificial and imported chart data. Key methods include:
- `mc_artificial_chart`: Runs Monte Carlo simulations on artificial chart data.
- `mc_import_chart`: Runs Monte Carlo simulations on imported chart data.
- `hist_performance`: Plots a histogram of performance distributions for different strategies.

## Getting Started

To run the simulations and analyses, you will need to have Python and the required libraries installed. You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib tqdm joblib numba pytest
```

You can then run the scripts and tests as needed. For example, to run the performance simulation and compare it with historical data, you can execute:

```bash
python performance_simulation.py
python msci_compare.py
```

To run the tests, you can use:

```bash
pytest tests.py
```

## Features

- **Flexible Simulation**: Simulate various trading strategies, including random swing trades, swing trades, and buy-and-hold.
- **Monte Carlo Simulations**: Perform parallelized Monte Carlo simulations for robust analysis.
- **Data Import**: Load and normalize historical data from single or multiple sources with rebalancing support.
- **Visualization**: Plot performance and distribution of results for better insights.
- **Comprehensive Metrics**: Calculate metrics such as absolute return, relative performance, TTWROR, yearly performance, and internal rate of return.

## Conclusion

This project provides a comprehensive framework for simulating and analyzing trading performance using various strategies. It includes methods for loading and normalizing historical data, simulating performance, and analyzing the results. Τhe Monte Carlo simulations provide robust insights into strategy performance.
