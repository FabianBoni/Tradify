"""
Debug script for testing data fetching from multiple sources
"""
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_cryptocompare_api(symbol, start_date, end_date, interval='1h', api_key=None):
    """
    Test CryptoCompare API with detailed debugging
    """
    print(f"\n{'='*60}")
    print(f"TESTING CRYPTOCOMPARE API FOR {symbol}")
    print(f"{'='*60}")
    
    base_url = "https://min-api.cryptocompare.com/data/v2/"
    
    # Convert interval to CryptoCompare format
    if interval in ['1h', 'hour']:
        endpoint = "histohour"
        limit = 2000
    elif interval in ['1d', 'day']:
        endpoint = "histoday"
        limit = 2000
    else:
        endpoint = "histohour"
        limit = 2000
    
    # Calculate timestamps
    if isinstance(start_date, str):
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    else:
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    
    if isinstance(end_date, str):
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    else:
        end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())
    
    print(f"Requested period: {datetime.fromtimestamp(start_timestamp)} to {datetime.fromtimestamp(end_timestamp)}")
    print(f"Endpoint: {endpoint}")
    print(f"Limit: {limit}")
    
    # Test different strategies
    strategies = [
        {
            'name': 'Standard Request with End Timestamp',
            'params': {
                'fsym': symbol.upper().replace('-USD', '').replace('USD', ''),
                'tsym': 'USD',
                'limit': limit,
                'toTs': end_timestamp
            }
        },
        {
            'name': 'Maximum Available Data',
            'params': {
                'fsym': symbol.upper().replace('-USD', '').replace('USD', ''),
                'tsym': 'USD',
                'limit': 2000
            }
        },
        {
            'name': 'Current Time Backwards',
            'params': {
                'fsym': symbol.upper().replace('-USD', '').replace('USD', ''),
                'tsym': 'USD',
                'limit': limit,
                'toTs': int(datetime.now().timestamp())
            }
        }
    ]
    
    if api_key:
        for strategy in strategies:
            strategy['params']['api_key'] = api_key
    
    best_data = None
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n--- Strategy {i}: {strategy['name']} ---")
        try:
            url = f"{base_url}{endpoint}"
            print(f"URL: {url}")
            print(f"Params: {strategy['params']}")
            
            response = requests.get(url, params=strategy['params'], timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data['Response'] == 'Error':
                print(f"❌ API Error: {data.get('Message', 'Unknown error')}")
                continue
            
            # Process data
            df_data = []
            for item in data['Data']['Data']:
                df_data.append({
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'Open': item['open'],
                    'High': item['high'],
                    'Low': item['low'],
                    'Close': item['close'],
                    'Volume': item['volumefrom']
                })
            
            df = pd.DataFrame(df_data)
            if len(df) == 0:
                print("❌ No data returned")
                continue
                
            df.set_index('timestamp', inplace=True)
            
            # Remove invalid data
            valid_df = df[(df['Volume'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
            
            print(f"✅ Raw data points: {len(df)}")
            print(f"✅ Valid data points: {len(valid_df)}")
            print(f"✅ Date range: {valid_df.index.min()} to {valid_df.index.max()}")
            
            # Check coverage of requested period
            requested_start = datetime.fromtimestamp(start_timestamp)
            requested_end = datetime.fromtimestamp(end_timestamp)
            
            coverage_start = max(valid_df.index.min(), requested_start)
            coverage_end = min(valid_df.index.max(), requested_end)
            
            if coverage_start <= coverage_end:
                covered_data = valid_df[(valid_df.index >= coverage_start) & (valid_df.index <= coverage_end)]
                print(f"📊 Coverage for requested period: {len(covered_data)} points")
                print(f"📊 Coverage dates: {coverage_start} to {coverage_end}")
            else:
                print(f"❌ No coverage for requested period")
            
            if best_data is None or len(valid_df) > len(best_data):
                best_data = valid_df
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return best_data

def test_yahoo_finance(symbol, start_date, end_date, interval='1h'):
    """
    Test Yahoo Finance API
    """
    print(f"\n{'='*60}")
    print(f"TESTING YAHOO FINANCE FOR {symbol}")
    print(f"{'='*60}")
    
    # Try different symbol formats
    yf_symbols = [symbol, f"{symbol}-USD", f"{symbol}USD"]
    
    best_data = None
    
    for yf_symbol in yf_symbols:
        print(f"\n--- Testing symbol: {yf_symbol} ---")
        try:
            # Try different intervals
            intervals_to_try = [interval, '1d']
            
            for test_interval in intervals_to_try:
                print(f"Trying interval: {test_interval}")
                
                data = yf.download(
                    yf_symbol, 
                    start=start_date, 
                    end=end_date,
                    interval=test_interval,
                    auto_adjust=False, 
                    progress=False
                )
                
                if data.empty:
                    print(f"❌ No data for {yf_symbol} with interval {test_interval}")
                    continue
                
                # Handle MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]
                
                data = data.dropna()
                
                print(f"✅ Data points: {len(data)}")
                print(f"✅ Date range: {data.index.min()} to {data.index.max()}")
                print(f"✅ Columns: {list(data.columns)}")
                
                if best_data is None or len(data) > len(best_data):
                    best_data = data
                    print(f"🏆 Best result so far: {len(data)} points")
                
                break  # If successful, no need to try other intervals
                
        except Exception as e:
            print(f"❌ Error with {yf_symbol}: {str(e)}")
    
    return best_data

def compare_data_sources(symbol, start_date, end_date, interval='1h', api_key=None):
    """
    Compare data from different sources
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE DATA SOURCE COMPARISON")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print(f"{'='*80}")
    
    results = {}
    
    # Test CryptoCompare
    try:
        cc_data = test_cryptocompare_api(symbol, start_date, end_date, interval, api_key)
        if cc_data is not None and len(cc_data) > 0:
            results['CryptoCompare'] = {
                'data': cc_data,
                'points': len(cc_data),
                'start': cc_data.index.min(),
                'end': cc_data.index.max(),
                'success': True
            }
        else:
            results['CryptoCompare'] = {'success': False, 'error': 'No data returned'}
    except Exception as e:
        results['CryptoCompare'] = {'success': False, 'error': str(e)}
    
    # Test Yahoo Finance
    try:
        yf_data = test_yahoo_finance(symbol, start_date, end_date, interval)
        if yf_data is not None and len(yf_data) > 0:
            results['Yahoo Finance'] = {
                'data': yf_data,
                'points': len(yf_data),
                'start': yf_data.index.min(),
                'end': yf_data.index.max(),
                'success': True
            }
        else:
            results['Yahoo Finance'] = {'success': False, 'error': 'No data returned'}
    except Exception as e:
        results['Yahoo Finance'] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for source, result in results.items():
        print(f"\n{source}:")
        if result['success']:
            print(f"  ✅ Success: {result['points']} data points")
            print(f"  📅 Range: {result['start']} to {result['end']}")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    # Recommend best source
    successful_sources = {k: v for k, v in results.items() if v['success']}
    
    if successful_sources:
        best_source = max(successful_sources.items(), key=lambda x: x[1]['points'])
        print(f"\n🏆 RECOMMENDED SOURCE: {best_source[0]}")
        print(f"   Reason: Most data points ({best_source[1]['points']})")
        return best_source[1]['data']
    else:
        print(f"\n❌ NO SUCCESSFUL DATA SOURCES FOUND")
        return None

def main():
    """
    Main debug function
    """
    # Test parameters from your error
    test_configs = [
        {
            'symbol': 'XRP',
            'start_date': '2014-01-01',
            'end_date': '2023-02-28',
            'interval': '1h',
            'description': 'Original failing request (XRP historical)'
        },
        {
            'symbol': 'XRP',
            'start_date': '2025-03-01',
            'end_date': '2025-06-03',
            'interval': '1h',
            'description': 'Recent XRP data (should work)'
        },
        {
            'symbol': 'BTC',
            'start_date': '2020-01-01',
            'end_date': '2023-01-01',
            'interval': '1d',
            'description': 'BTC daily data (good coverage)'
        },
        {
            'symbol': 'AAPL',
            'start_date': '2020-01-01',
            'end_date': '2023-01-01',
            'interval': '1d',
            'description': 'Traditional stock (Yahoo Finance)'
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n\n{'#'*100}")
        print(f"TEST {i}: {config['description']}")
        print(f"{'#'*100}")
        
        best_data = compare_data_sources(
            symbol=config['symbol'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            interval=config['interval'],
            api_key=None  # Add your API key here if you have one
        )
        
        if best_data is not None:
            print(f"\n📈 SAMPLE DATA:")
            print(best_data.head())
            print(f"\n📈 DATA SUMMARY:")
            print(best_data.describe())

if __name__ == "__main__":
    main()
