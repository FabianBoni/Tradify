# Streamlit Backtest App

This project is a Streamlit application that allows users to backtest trading strategies using the backtesting.py library. The application provides an interactive interface for users to define their strategies, run backtests, and visualize the results.

## Project Structure

```
streamlit-backtest-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── components            # Contains reusable components for the app
│   │   ├── __init__.py      # Package initializer
│   │   └── charts.py        # Functions for visualizing backtest results
│   ├── strategies            # Contains trading strategies
│   │   ├── __init__.py      # Package initializer
│   │   └── sma_cross.py      # Implementation of the SmaCross strategy
│   └── utils                # Utility functions and classes
│       ├── __init__.py      # Package initializer
│       └── backtest_engine.py # Implementation of the backtest engine
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-backtest-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage

- Once the application is running, you can interact with the interface to define your trading strategies and run backtests.
- The results will be displayed in the application, allowing you to analyze the performance of your strategies.

## Overview

This application is designed to provide a user-friendly way to backtest trading strategies using historical data. It leverages the capabilities of the backtesting.py library to perform the backtesting logic and visualize the results through Streamlit components.