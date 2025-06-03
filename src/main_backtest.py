"""
Main backtesting application with enhanced data fetching
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from src.backtest_engine import BacktestEngine
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy

# Configuration
CRYPTOCOMPARE_API_KEY = "d00fcf138b83f5b992a6af8c73f418bd465c6cc4be3893b260e7399b9fcac6e9"

def main():
    print("🚀 TRADIFY - Advanced Crypto Backtesting Platform")
    print("=" * 60)
    
    # User inputs
    symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
    start_date_str = input("Enter start date (YYYY-MM-DD, default: 2015-01-01): ").strip() or "2015-01-01"
    end_date_str = input("Enter end date (YYYY-MM-DD, default: 2023-12-31): ").strip() or "2023-12-31"
    
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        print("❌ Invalid date format. Using defaults.")
        start_date = datetime(2015, 1, 1)
        end_date = datetime(2023, 12, 31)
    
    print(f"\n📊 Configuration:")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    # Initialize data fetcher
    fetcher = DataFetcher(cryptocompare_api_key=CRYPTOCOMPARE_API_KEY)
    
    # Fetch historical data
    print(f"\n{'='*60}")
    print("DATA FETCHING")
    print(f"{'='*60}")
    
    data_path = f"data/{symbol.lower()}_historical.csv"
    df = fetcher.fetch_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        save_path=data_path
    )
    
    if df.empty:
        print("❌ No data available for backtesting. Exiting.")
        return
    
    # Strategy selection
    print(f"\n{'='*60}")
    print("STRATEGY SELECTION")
    print(f"{'='*60}")
    print("Available strategies:")
    print("1. Moving Average Crossover")
    print("2. RSI Strategy")
    
    strategy_choice = input("Choose strategy (1-2, default: 1): ").strip() or "1"
    
    if strategy_choice == "1":
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        strategy_name = "Moving Average"
    elif strategy_choice == "2":
        strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
        strategy_name = "RSI"
    else:
        print("Invalid choice, using Moving Average")
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        strategy_name = "Moving Average"
    
    # Run backtest
    print(f"\n{'='*60}")
    print(f"BACKTESTING - {strategy_name} Strategy")
    print(f"{'='*60}")
    
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001  # 0.1% commission
    )
    
    results = engine.run_backtest(df, strategy)
    
    # Display results
    print(f"\n📈 BACKTEST RESULTS")
    print(f"{'='*30}")
    
    if results:
        final_value = results['portfolio_value'].iloc[-1]
        total_return = ((final_value - engine.initial_capital) / engine.initial_capital) * 100
        
        print(f"Initial Capital: ${engine.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Calculate additional metrics
        trades = results['position'].diff().abs().sum() / 2  # Number of trades
        print(f"Number of Trades: {trades:.0f}")
        
        if trades > 0:
            # Calculate win rate
            returns = results['portfolio_value'].pct_change().dropna()
            positive_returns = (returns > 0).sum()
            win_rate = (positive_returns / len(returns)) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        # Save results
        results_path = f"results/{symbol.lower()}_{strategy_name.lower().replace(' ', '_')}_results.csv"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to: {results_path}")
        
    else:
        print("❌ Backtest failed to generate results")
    
    print(f"\n🎉 Backtesting completed!")

if __name__ == "__main__":
    main()
