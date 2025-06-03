"""
Production backtest runner
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.main_backtest import main
    main()
