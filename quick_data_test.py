"""
Quick data availability test script
"""
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

def quick_test_symbol(symbol, days_back=365):
    """
    Quick test of data availability for a symbol
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n🔍 TESTING: {symbol}")
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    print("-" * 50)
    
    # Test CryptoCompare
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': min(days_back, 2000)
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data['Response'] == 'Success':
            points = len(data['Data']['Data'])
            latest = datetime.fromtimestamp(data['Data']['Data'][-1]['time'])
            oldest = datetime.fromtimestamp(data['Data']['Data'][0]['time'])
            print(f"✅ CryptoCompare: {points} points ({oldest.date()} to {latest.date()})")
        else:
            print(f"❌ CryptoCompare: {data.get('Message', 'Failed')}")
    except Exception as e:
        print(f"❌ CryptoCompare: Error - {str(e)}")
    
    # Test Yahoo Finance
    for yf_symbol in [symbol, f"{symbol}-USD", f"{symbol}USD"]:
        try:
            data = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                print(f"✅ Yahoo Finance ({yf_symbol}): {len(data)} points ({data.index.min().date()} to {data.index.max().date()})")
                break
        except:
            continue
    else:
        print(f"❌ Yahoo Finance: No data found")

def main():
    """
    Test multiple symbols quickly
    """
    print("🚀 QUICK DATA AVAILABILITY TEST")
    print("=" * 60)
    
    # Test different symbols and time periods
    tests = [
        ('XRP', 365),
        ('BTC', 365), 
        ('ETH', 365),
        ('AAPL', 365),
        ('GOOGL', 365),
        ('XRP', 90),  # Shorter period
        ('BTC', 30),  # Very short period
    ]
    
    for symbol, days in tests:
        quick_test_symbol(symbol, days)
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    print("- If CryptoCompare fails, try Yahoo Finance")
    print("- Use shorter time periods for crypto")
    print("- Traditional stocks (AAPL, GOOGL) have better historical coverage")
    print("- Consider using daily ('1d') interval for longer historical data")

if __name__ == "__main__":
    main()
