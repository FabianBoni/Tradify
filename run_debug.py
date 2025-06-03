"""
Enhanced debug runner for data fetch issues based on working patterns
"""
import sys
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# XRP Launch Date - XRP was created in 2012 and launched publicly in 2013
XRP_LAUNCH_DATE = datetime(2013, 1, 1)  # XRP public launch
XRP_LAUNCH_TIMESTAMP = int(XRP_LAUNCH_DATE.timestamp())

def test_cryptocompare_detailed(symbol="XRP", api_key=None, test_historical=True):
    """
    Detailed CryptoCompare test based on working patterns
    """
    print(f"\n🔍 DETAILED CRYPTOCOMPARE TEST FOR {symbol}")
    print("=" * 60)
    
    # Test basic connection first
    test_url = "https://min-api.cryptocompare.com/data/price"
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    params = {"fsym": symbol, "tsyms": "USD"}
    
    print("1. Testing basic API connection...")
    try:
        response = requests.get(test_url, headers=headers, params=params)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if "USD" in data and data["USD"] > 0:
                print(f"   ✅ Basic API working! Current {symbol} price: ${data['USD']}")
            else:
                print(f"   ❌ API returned invalid price: {data}")
                return False
        else:
            print(f"   ❌ API request failed")
            return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False
    
    # Test historical data endpoints with XRP-specific dates
    print("\n2. Testing historical data endpoints...")
    
    if symbol.upper() == "XRP" and test_historical:
        print(f"   📅 Using XRP launch date: {XRP_LAUNCH_DATE}")
        
        # Test different strategies to get historical data
        historical_tests = [
            ("histoday", "Daily data from launch (with fromTs)", {
                "limit": 2000, 
                "toTs": int(datetime.now().timestamp()),
                "fromTs": XRP_LAUNCH_TIMESTAMP
            }),
            ("histoday", "Daily data from launch (no fromTs)", {
                "limit": 2000, 
                "toTs": int(datetime.now().timestamp())
            }),
            ("histoday", "Daily data to 2020 (specific period)", {
                "limit": 2000,
                "toTs": int(datetime(2020, 12, 31).timestamp()),
                "fromTs": XRP_LAUNCH_TIMESTAMP
            }),
            ("histoday", "Daily data to 2018 (earlier period)", {
                "limit": 2000,
                "toTs": int(datetime(2018, 12, 31).timestamp()),
                "fromTs": XRP_LAUNCH_TIMESTAMP
            }),
            ("histoday", "Daily data (last 365 days)", {"limit": 365}),
            ("histohour", "Hourly data (last 30 days)", {"limit": 720}),
        ]
    else:
        # Default tests for other symbols
        historical_tests = [
            ("histoday", "Daily data", {"limit": 30}),
            ("histohour", "Hourly data", {"limit": 24}),
            ("histominute", "Minute data", {"limit": 60})
        ]
    
    base_url = "https://min-api.cryptocompare.com/data/v2"
    
    for endpoint, description, extra_params in historical_tests:
        print(f"\n   Testing {endpoint} ({description})...")
        
        url = f"{base_url}/{endpoint}"
        params = {
            "fsym": symbol,
            "tsym": "USD",
            **extra_params
        }
        
        if api_key:
            headers = {"authorization": f"Apikey {api_key}"}
        else:
            headers = {}
        
        print(f"     URL: {url}")
        print(f"     Params: {params}")
        
        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"     Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"     Response type: {data.get('Response', 'Unknown')}")
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    print(f"     ✅ Got {len(hist_data)} data points")
                    
                    if hist_data:
                        first_point = hist_data[0]
                        last_point = hist_data[-1]
                        
                        first_date = datetime.fromtimestamp(first_point['time'])
                        last_date = datetime.fromtimestamp(last_point['time'])
                        
                        print(f"     Date range: {first_date} to {last_date}")
                        
                        # Check if fromTs parameter was respected
                        if 'fromTs' in params:
                            requested_start = datetime.fromtimestamp(params['fromTs'])
                            actual_start_diff = (first_date - requested_start).days
                            print(f"     Requested start: {requested_start}")
                            print(f"     Actual start diff: {actual_start_diff} days later")
                            
                            if actual_start_diff > 365:
                                print(f"     ⚠️ API LIMITATION: fromTs parameter ignored! Got data from {actual_start_diff} days later than requested")
                                print(f"     💡 This indicates free tier API limitations")
                            else:
                                print(f"     ✅ fromTs parameter respected")
                        
                        # Calculate how far back the data goes
                        if symbol.upper() == "XRP":
                            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
                            total_days_since_launch = (datetime.now() - XRP_LAUNCH_DATE).days
                            coverage_percentage = (len(hist_data) / total_days_since_launch) * 100
                            
                            print(f"     Days from XRP launch: {days_from_launch}")
                            print(f"     Historical coverage: {coverage_percentage:.1f}% of total history")
                            
                            if days_from_launch < 365:
                                print(f"     ❌ INSUFFICIENT: Only {days_from_launch} days from launch")
                            elif days_from_launch < 2000:
                                print(f"     ⚠️ LIMITED: {days_from_launch} days from launch (missing early years)")
                            else:
                                print(f"     ✅ GOOD: {days_from_launch} days from launch")
                        
                        print(f"     First point: Open={first_point.get('open', 0)}, Close={first_point.get('close', 0)}")
                        print(f"     Last point: Open={last_point.get('open', 0)}, Close={last_point.get('close', 0)}")
                        
                        # Check for zero values
                        zero_count = sum(1 for d in hist_data if d.get('close', 0) == 0)
                        if zero_count > 0:
                            print(f"     ⚠️ Found {zero_count} points with zero close price")
                        else:
                            print(f"     ✅ All points have valid prices")
                            
                        # Check price range
                        prices = [d.get('close', 0) for d in hist_data if d.get('close', 0) > 0]
                        if prices:
                            min_price = min(prices)
                            max_price = max(prices)
                            print(f"     Price range: ${min_price:.6f} - ${max_price:.6f}")
                            
                else:
                    print(f"     ❌ API Error: {data.get('Message', 'Unknown')}")
                    
                    # Check for specific error messages
                    error_msg = data.get('Message', '').lower()
                    if 'rate limit' in error_msg:
                        print(f"     💡 Rate limit hit - try with API key or wait")
                    elif 'invalid' in error_msg:
                        print(f"     💡 Invalid symbol or parameters")
                    elif 'no data' in error_msg:
                        print(f"     💡 No historical data available for this period")
                        
            else:
                print(f"     ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Request error: {e}")
        
        time.sleep(0.5)  # Rate limiting between requests
    
    return True

def test_xrp_historical_coverage(api_key=None):
    """
    Specific test for XRP historical data coverage
    """
    print(f"\n📅 XRP HISTORICAL DATA COVERAGE TEST")
    print("=" * 60)
    print(f"XRP Launch Date: {XRP_LAUNCH_DATE}")
    print(f"Days since launch: {(datetime.now() - XRP_LAUNCH_DATE).days}")
    
    # Test different time periods
    test_periods = [
        ("Last 30 days", datetime.now() - pd.Timedelta(days=30), datetime.now()),
        ("Last 1 year", datetime.now() - pd.Timedelta(days=365), datetime.now()),
        ("Last 2 years", datetime.now() - pd.Timedelta(days=730), datetime.now()),
        ("2020-2023", datetime(2020, 1, 1), datetime(2023, 12, 31)),
        ("2015-2020", datetime(2015, 1, 1), datetime(2020, 12, 31)),
        ("From launch (2013-2015)", XRP_LAUNCH_DATE, datetime(2015, 12, 31)),
    ]
    
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    
    for period_name, start_date, end_date in test_periods:
        print(f"\n🔍 Testing period: {period_name}")
        print(f"   From: {start_date}")
        print(f"   To: {end_date}")
        
        params = {
            "fsym": "XRP",
            "tsym": "USD",
            "limit": 2000,
            "toTs": int(end_date.timestamp()),
            "fromTs": int(start_date.timestamp())
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    
                    if hist_data:
                        # Filter data within requested range
                        filtered_data = [
                            d for d in hist_data 
                            if start_date.timestamp() <= d['time'] <= end_date.timestamp()
                            and d.get('close', 0) > 0
                        ]
                        
                        if filtered_data:
                            first_date = datetime.fromtimestamp(filtered_data[0]['time'])
                            last_date = datetime.fromtimestamp(filtered_data[-1]['time'])
                            
                            print(f"   ✅ Found {len(filtered_data)} valid data points")
                            print(f"   Actual range: {first_date} to {last_date}")
                            
                            # Calculate coverage
                            requested_days = (end_date - start_date).days
                            actual_days = (last_date - first_date).days
                            coverage = (actual_days / requested_days) * 100 if requested_days > 0 else 0
                            
                            print(f"   Coverage: {coverage:.1f}% ({actual_days}/{requested_days} days)")
                            
                            # Price info
                            prices = [d['close'] for d in filtered_data]
                            print(f"   Price range: ${min(prices):.6f} - ${max(prices):.6f}")
                            

                        else:
                            print(f"   ❌ No valid data in requested range")
                    else:
                        print(f"   ❌ No data returned")
                else:
                    print(f"   ❌ API Error: {data.get('Message', 'Unknown')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request error: {e}")
        
        time.sleep(1)  # Rate limiting

def test_yahoo_finance_detailed(symbol="XRP"):
    """
    Detailed Yahoo Finance test
    """
    print(f"\n📊 DETAILED YAHOO FINANCE TEST FOR {symbol}")
    print("=" * 60)
    
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Install with: pip install yfinance")
        return False
    
    # Try different symbol formats
    symbol_variants = [
        symbol,
        f"{symbol}-USD", 
        f"{symbol}USD",
        f"{symbol}USDT"
    ]
    
    intervals = ["1d", "1h", "5m"]
    periods = ["1mo", "3mo", "1y", "2y"]
    
    for variant in symbol_variants:
        print(f"\n   Testing symbol: {variant}")
        
        for period in periods:
            for interval in intervals:
                try:
                    print(f"     Trying {interval} data for {period}...")
                    
                    data = yf.download(
                        variant, 
                        period=period,
                        interval=interval,
                        auto_adjust=False, 
                        progress=False
                    )
                    
                    if not data.empty:
                        # Handle MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = [col[0] for col in data.columns]
                        
                        print(f"       ✅ Success: {len(data)} points from {data.index.min()} to {data.index.max()}")
                        print(f"       Columns: {list(data.columns)}")
                        
                        if 'Close' in data.columns:
                            price_range = f"${data['Close'].min():.6f} - ${data['Close'].max():.6f}"
                            print(f"       Price range: {price_range}")
                        
                        return True  # Found working data
                    else:
                        print(f"       ❌ No data")
                        
                except Exception as e:
                    print(f"       ❌ Error: {e}")
    
    print("   ❌ No working Yahoo Finance data found")
    return False

def test_alternative_apis(symbol="XRP"):
    """
    Test alternative free APIs
    """
    print(f"\n🔄 TESTING ALTERNATIVE APIs FOR {symbol}")
    print("=" * 60)
    
    # Test CoinGecko
    print("1. Testing CoinGecko...")
    try:
        # Map common symbols to CoinGecko IDs
        coingecko_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "ADA": "cardano",
            "DOT": "polkadot"
        }
        
        coin_id = coingecko_ids.get(symbol.upper(), symbol.lower())
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "30",
            "interval": "daily"
        }
        
        response = requests.get(url, params=params)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "prices" in data:
                prices = data["prices"]
                print(f"   ✅ Got {len(prices)} price points")
                
                if prices:
                    first_date = datetime.fromtimestamp(prices[0][0] / 1000)
                    last_date = datetime.fromtimestamp(prices[-1][0] / 1000)
                    print(f"   Date range: {first_date} to {last_date}")
                    print(f"   Price range: ${min(p[1] for p in prices):.6f} - ${max(p[1] for p in prices):.6f}")
                return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test Binance
    print("\n2. Testing Binance...")
    try:
        symbol_binance = f"{symbol}USDT"
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol_binance,
            "interval": "1d",
            "limit": 30
        }
        
        response = requests.get(url, params=params)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                print(f"   ✅ Got {len(data)} candles")
                
                first_candle = data[0]
                last_candle = data[-1]
                
                first_date = datetime.fromtimestamp(first_candle[0] / 1000)
                last_date = datetime.fromtimestamp(last_candle[0] / 1000)
                
                print(f"   Date range: {first_date} to {last_date}")
                print(f"   Price range: ${min(float(d[4]) for d in data):.6f} - ${max(float(d[4]) for d in data):.6f}")
                return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

def comprehensive_data_test():
    """
    Run comprehensive data availability test
    """
    print("🚀 COMPREHENSIVE DATA AVAILABILITY TEST")
    print("=" * 80)
    
    # Test symbols with their launch dates
    test_cases = [
        ("XRP", None, "XRP (Ripple) - launched 2013"),
        ("BTC", None, "Bitcoin - launched 2009"),
        ("ETH", None, "Ethereum - launched 2015"),
        ("DOGE", "d00fcf138b83f5b992a6af8c73f418bd465c6cc4be3893b260e7399b9fcac6e9", "Dogecoin - launched 2013"),
    ]
    
    results = {}
    
    for symbol, api_key, description in test_cases:
        print(f"\n{'#' * 60}")
        print(f"TESTING: {description}")
        print(f"{'#' * 60}")
        
        results[symbol] = {
            "cryptocompare": False,
            "yahoo_finance": False,
            "alternatives": False
        }
        
        # Test CryptoCompare with historical focus
        try:
            cc_result = test_cryptocompare_detailed(symbol, api_key, test_historical=True)
            results[symbol]["cryptocompare"] = cc_result
            
            # Additional XRP-specific test
            if symbol.upper() == "XRP":
                print(f"\n📊 Running XRP-specific historical coverage test...")
                test_xrp_historical_coverage(api_key)
                
        except Exception as e:
            print(f"❌ CryptoCompare test failed: {e}")
        
        # Test Yahoo Finance
        try:
            yf_result = test_yahoo_finance_detailed(symbol)
            results[symbol]["yahoo_finance"] = yf_result
        except Exception as e:
            print(f"❌ Yahoo Finance test failed: {e}")
        
        # Test alternatives
        try:
            alt_result = test_alternative_apis(symbol)
            results[symbol]["alternatives"] = alt_result
        except Exception as e:
            print(f"❌ Alternative APIs test failed: {e}")
        
        time.sleep(2)  # Rate limiting between symbols
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY OF RESULTS")
    print(f"{'=' * 80}")
    
    for symbol, tests in results.items():
        print(f"\n{symbol}:")
        for source, success in tests.items():
            status = "✅ WORKING" if success else "❌ FAILED"
            print(f"  {source.replace('_', ' ').title()}: {status}")
    
    # XRP-specific recommendations
    print(f"\n{'=' * 80}")
    print("XRP-SPECIFIC RECOMMENDATIONS")
    print(f"{'=' * 80}")
    
    print(f"🔍 XRP Historical Data Analysis:")
    print(f"- XRP launched: {XRP_LAUNCH_DATE}")
    print(f"- Days since launch: {(datetime.now() - XRP_LAUNCH_DATE).days}")
    print(f"- Your backtest requested: 2014-01-01 to 2023-02-28")
    print(f"- This should be available (1+ year after launch)")
    
    print(f"\n💡 Recommendations for XRP backtesting:")
    print("1. Use CryptoCompare with API key for best historical coverage")
    print("2. Try Yahoo Finance with 'XRP-USD' symbol")
    print("3. Use daily ('1d') interval for longest historical data")
    print("4. Consider starting from 2014 instead of 2013 for better data quality")
    print("5. Split long periods into smaller chunks if API limits hit")

def test_api_limitation_workarounds(symbol="XRP", api_key=None):
    """
    Test workarounds for API limitations
    """
    print(f"\n🔧 TESTING API LIMITATION WORKAROUNDS FOR {symbol}")
    print("=" * 60)
    
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    
    # Strategy 1: Multiple overlapping requests to get older data
    print("\n1. Testing multiple overlapping requests...")
    
    # Start from current time and go backwards in chunks
    end_time = datetime.now()
    chunk_size_days = 1000  # Request 1000 days at a time
    
    for i in range(3):  # Test 3 chunks
        chunk_end = end_time - pd.Timedelta(days=i * chunk_size_days)
        chunk_start = chunk_end - pd.Timedelta(days=chunk_size_days)
        
        print(f"\n   Chunk {i+1}: {chunk_start.date()} to {chunk_end.date()}")
        
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": chunk_size_days,
            "toTs": int(chunk_end.timestamp())
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    
                    if hist_data:
                        first_date = datetime.fromtimestamp(hist_data[0]['time'])
                        last_date = datetime.fromtimestamp(hist_data[-1]['time'])
                        
                        print(f"     ✅ Got {len(hist_data)} points: {first_date.date()} to {last_date.date()}")
                        
                        # Check how far back this chunk goes
                        days_back = (datetime.now() - first_date).days
                        print(f"     Reaches back {days_back} days from now")
                        
                        if symbol.upper() == "XRP":
                            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
                            if days_from_launch < 0:
                                print(f"     🎉 SUCCESS: Reaches {abs(days_from_launch)} days BEFORE XRP launch!")
                            else:
                                print(f"     Still {days_from_launch} days after XRP launch")
                        
                    else:
                        print(f"     ❌ No data in chunk")
                else:
                    print(f"     ❌ API Error: {data.get('Message', 'Unknown')}")
            else:
                print(f"     ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Request error: {e}")
        
        time.sleep(1)  # Rate limiting
    
    # Strategy 2: Test different endpoints and aggregations
    print(f"\n2. Testing different endpoints for older data...")
    
    test_configs = [
        ("histoday", {"aggregate": 1}, "Daily data"),
        ("histoday", {"aggregate": 7}, "Weekly data (7-day aggregate)"),
        ("histoday", {"aggregate": 30}, "Monthly data (30-day aggregate)"),
    ]
    
    for endpoint, extra_params, description in test_configs:
        print(f"\n   Testing {description}...")
        
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 2000,
            "toTs": int(datetime(2020, 1, 1).timestamp()),  # Try to get data up to 2020
            **extra_params
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    
                    if hist_data:
                        first_date = datetime.fromtimestamp(hist_data[0]['time'])
                        last_date = datetime.fromtimestamp(hist_data[-1]['time'])
                        
                        print(f"     ✅ Got {len(hist_data)} points: {first_date.date()} to {last_date.date()}")
                        
                        if symbol.upper() == "XRP":
                            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
                            print(f"     Days from XRP launch: {days_from_launch}")
                        
                    else:
                        print(f"     ❌ No data returned")
                else:
                    print(f"     ❌ API Error: {data.get('Message', 'Unknown')}")
            else:
                print(f"     ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Request error: {e}")
        
        time.sleep(1)

def test_coindesk_api(symbol="XRP"):
    """
    Test CoinDesk API for historical data
    """
    print(f"\n💰 TESTING COINDESK API FOR {symbol}")
    print("=" * 60)
    
    # CoinDesk API endpoints
    base_url = "https://api.coindesk.com/v2"
    
    # Test current price endpoint
    print("1. Testing current price endpoint...")
    try:
        url = f"{base_url}/bpi/currentprice.json"
        response = requests.get(url)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'bpi' in data and 'USD' in data['bpi']:
                price = data['bpi']['USD']['rate_float']
                print(f"   ✅ Bitcoin price: ${price:,.2f}")
            else:
                print(f"   ❌ Unexpected response format: {data}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test historical data endpoint
    print("\n2. Testing historical data endpoint...")
    try:
        # CoinDesk historical endpoint (Bitcoin only)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"{base_url}/bpi/historical/close.json"
        params = {
            "start": start_date,
            "end": end_date
        }
        
        response = requests.get(url, params=params)
        print(f"   Status: {response.status_code}")
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        
        if response.status_code == 200:
            data = response.json()
            if 'bpi' in data:
                historical_data = data['bpi']
                print(f"   ✅ Got {len(historical_data)} historical data points")
                
                dates = list(historical_data.keys())
                if dates:
                    print(f"   Date range: {min(dates)} to {max(dates)}")
                    prices = list(historical_data.values())
                    print(f"   Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
                
                return True
            else:
                print(f"   ❌ No historical data in response: {data}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("   💡 Note: CoinDesk API only supports Bitcoin (BTC) data")
    return False

def test_cryptocompare_with_proper_params(symbol="XRP", api_key=None):
    """
    Test CryptoCompare with proper parameter usage based on documentation
    """
    print(f"\n🔍 CRYPTOCOMPARE API WITH PROPER PARAMETERS FOR {symbol}")
    print("=" * 60)
    
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    
    # Strategy 1: Use toTs to go backwards in time (proper way)
    print("\n1. Testing backwards pagination with toTs...")
    
    # Start from different points in time and see how far back we can go
    test_dates = [
        ("Current time", datetime.now()),
        ("2023-01-01", datetime(2023, 1, 1)),
        ("2020-01-01", datetime(2020, 1, 1)),
        ("2018-01-01", datetime(2018, 1, 1)),
        ("2015-01-01", datetime(2015, 1, 1)),
        ("XRP Launch", XRP_LAUNCH_DATE),
    ]
    
    for test_name, end_date in test_dates:
        print(f"\n   Testing from {test_name} ({end_date.date()})...")
        
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 2000,  # Maximum allowed
            "toTs": int(end_date.timestamp())
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    
                    if hist_data:
                        first_date = datetime.fromtimestamp(hist_data[0]['time'])
                        last_date = datetime.fromtimestamp(hist_data[-1]['time'])
                        
                        print(f"     ✅ Got {len(hist_data)} points: {first_date.date()} to {last_date.date()}")
                        
                        # Check how far back we got
                        days_back = (end_date - first_date).days
                        print(f"     Reached {days_back} days back from requested end date")
                        
                        if symbol.upper() == "XRP":
                            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
                            if days_from_launch <= 0:
                                print(f"     🎉 SUCCESS: Reached XRP launch date! ({abs(days_from_launch)} days before/after)")
                            else:
                                print(f"     Still {days_from_launch} days after XRP launch")
                        
                        # Check data quality
                        valid_prices = [d for d in hist_data if d.get('close', 0) > 0]
                        print(f"     Valid price points: {len(valid_prices)}/{len(hist_data)}")
                        
                        if valid_prices:
                            prices = [d['close'] for d in valid_prices]
                            print(f"     Price range: ${min(prices):.6f} - ${max(prices):.6f}")
                        
                    else:
                        print(f"     ❌ No data returned")
                else:
                    print(f"     ❌ API Error: {data.get('Message', 'Unknown')}")
                    
                    # Check for rate limiting
                    if 'rate limit' in data.get('Message', '').lower():
                        print(f"     💡 Rate limited - need API key or wait")
                        break
                        
            else:
                print(f"     ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Request error: {e}")
        
        time.sleep(1.5)  # Rate limiting
    
    # Strategy 2: Test aggregate parameters for longer historical coverage
    print(f"\n2. Testing aggregate parameters for longer coverage...")
    
    aggregate_tests = [
        (1, "Daily data"),
        (7, "Weekly aggregated data"),
        (30, "Monthly aggregated data"),
    ]
    
    for aggregate, description in aggregate_tests:
        print(f"\n   Testing {description} (aggregate={aggregate})...")
        
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 2000,
            "aggregate": aggregate,
            "toTs": int(datetime(2020, 1, 1).timestamp())  # Try to get data up to 2020
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    
                    if hist_data:
                        first_date = datetime.fromtimestamp(hist_data[0]['time'])
                        last_date = datetime.fromtimestamp(hist_data[-1]['time'])
                        
                        print(f"     ✅ Got {len(hist_data)} points: {first_date.date()} to {last_date.date()}")
                        
                        # Calculate effective coverage with aggregation
                        total_days = (last_date - first_date).days
                        print(f"     Total days covered: {total_days}")
                        
                        if symbol.upper() == "XRP":
                            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
                            print(f"     Days from XRP launch: {days_from_launch}")
                        
                    else:
                        print(f"     ❌ No data returned")
                else:
                    print(f"     ❌ API Error: {data.get('Message', 'Unknown')}")
            else:
                print(f"     ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Request error: {e}")
        
        time.sleep(1)

def test_alternative_crypto_apis(symbol="XRP"):
    """
    Test multiple alternative crypto APIs for historical data
    """
    print(f"\n🌐 TESTING ALTERNATIVE CRYPTO APIs FOR {symbol}")
    print("=" * 60)
    
    # Test CoinGecko with longer historical data
    print("1. Testing CoinGecko (extended)...")
    try:
        coingecko_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "ADA": "cardano",
            "DOT": "polkadot"
        }
        
        coin_id = coingecko_ids.get(symbol.upper(), symbol.lower())
        
        # Test different time ranges
        test_ranges = [
            ("max", "Maximum available data"),
            ("1095", "3 years"),
            ("730", "2 years"),
            ("365", "1 year")
        ]
        
        for days, description in test_ranges:
            print(f"\n   Testing {description}...")
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            
            response = requests.get(url, params=params)
            print(f"     Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "prices" in data:
                    prices = data["prices"]
                    print(f"     ✅ Got {len(prices)} price points")
                    
                    if prices:
                        first_date = datetime.fromtimestamp(prices[0][0] / 1000)
                        last_date = datetime.fromtimestamp(prices[-1][0] / 1000)
                        print(f"     Date range: {first_date.date()} to {last_date.date()}")
                        
                        if symbol.upper() == "XRP":
                            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
                            print(f"     Days from XRP launch: {days_from_launch}")
                            
                            if days_from_launch <= 30:  # Close to launch
                                print(f"     🎉 EXCELLENT: Very close to XRP launch date!")
                                return True
                        
                        price_values = [p[1] for p in prices]
                        print(f"     Price range: ${min(price_values):.6f} - ${max(price_values):.6f}")
                        
                        if days == "max":  # This is our best shot at full history
                            return len(prices) > 1000  # Good amount of historical data
                            
            else:
                print(f"     ❌ Failed: {response.status_code}")
                
            time.sleep(0.5)  # Rate limiting
            
    except Exception as e:
        print(f"   ❌ CoinGecko error: {e}")
    
    # Test CoinDesk (Bitcoin only)
    if symbol.upper() == "BTC":
        print(f"\n2. Testing CoinDesk for Bitcoin...")
        return test_coindesk_api(symbol)
    else:
        print(f"\n2. Skipping CoinDesk (Bitcoin only, testing {symbol})")
    
    return False

def test_cryptocompare_pagination(symbol="XRP", api_key=None, target_start_date=None):
    """
    Test CryptoCompare pagination to get complete historical data
    """
    print(f"\n📋 CRYPTOCOMPARE PAGINATION TEST FOR {symbol}")
    print("=" * 60)
    
    if target_start_date is None:
        target_start_date = XRP_LAUNCH_DATE if symbol.upper() == "XRP" else datetime(2015, 1, 1)
    
    print(f"Target start date: {target_start_date}")
    print(f"Target: Get all data from {target_start_date} to now")
    
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    
    all_data = []
    current_to_ts = int(datetime.now().timestamp())
    batch_count = 0
    max_batches = 20  # Safety limit
    
    print(f"\nStarting pagination from {datetime.fromtimestamp(current_to_ts)}...")
    
    while current_to_ts > target_start_date.timestamp() and batch_count < max_batches:
        batch_count += 1
        print(f"\n--- Batch {batch_count} ---")
        print(f"Requesting data up to: {datetime.fromtimestamp(current_to_ts)}")
        
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 2000,
            "toTs": current_to_ts
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hist_data = data.get('Data', {}).get('Data', [])
                    
                    if hist_data and len(hist_data) > 0:
                        # Filter out zero/invalid data
                        valid_data = [d for d in hist_data if d.get('close', 0) > 0]
                        
                        if valid_data:
                            first_date = datetime.fromtimestamp(valid_data[0]['time'])
                            last_date = datetime.fromtimestamp(valid_data[-1]['time'])
                            
                            print(f"✅ Got {len(valid_data)} valid points: {first_date.date()} to {last_date.date()}")
                            
                            # Add to collection (avoid duplicates)
                            new_data = []
                            existing_timestamps = {d['time'] for d in all_data}
                            
                            for point in valid_data:
                                if point['time'] not in existing_timestamps:
                                    new_data.append(point)
                            
                            all_data.extend(new_data)
                            print(f"Added {len(new_data)} new data points (total: {len(all_data)})")
                            
                            # Check if we've reached our target
                            if first_date <= target_start_date:
                                print(f"🎉 SUCCESS: Reached target start date!")
                                
                                # Filter to exact target range
                                filtered_data = [
                                    d for d in all_data 
                                    if d['time'] >= target_start_date.timestamp()
                                ]
                                
                                print(f"Final dataset: {len(filtered_data)} points from target date")
                                break
                            
                            # Update toTs for next batch (use earliest timestamp - 1)
                            current_to_ts = valid_data[0]['time'] - 86400  # Go back 1 day
                            
                            # Calculate progress
                            days_from_target = (first_date - target_start_date).days
                            print(f"Still {days_from_target} days from target date")
                            

                        else:
                            print("❌ No valid data in batch")
                            break
                    else:
                        print("❌ No data returned in batch")
                        break
                else:
                    error_msg = data.get('Message', 'Unknown error')
                    print(f"❌ API Error: {error_msg}")
                    
                    if 'rate limit' in error_msg.lower():
                        print("💡 Rate limited - waiting longer...")
                        time.sleep(10)
                        continue
                    else:
                        break
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                break
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            break
        
        # Rate limiting between requests
        time.sleep(2)
    
    # Final summary
    print(f"\n{'='*50}")
    print("PAGINATION SUMMARY")
    print(f"{'='*50}")
    
    if all_data:
        # Sort by timestamp
        all_data.sort(key=lambda x: x['time'])
        
        first_date = datetime.fromtimestamp(all_data[0]['time'])
        last_date = datetime.fromtimestamp(all_data[-1]['time'])
        
        print(f"Total data points collected: {len(all_data)}")
        print(f"Date range: {first_date.date()} to {last_date.date()}")
        print(f"Total days covered: {(last_date - first_date).days}")
        
        if symbol.upper() == "XRP":
            days_from_launch = (first_date - XRP_LAUNCH_DATE).days
            if days_from_launch <= 30:
                print(f"🎉 EXCELLENT: Got within {abs(days_from_launch)} days of XRP launch!")
            else:
                print(f"⚠️ Missing early data: {days_from_launch} days after XRP launch")
        
        # Price analysis
        prices = [d['close'] for d in all_data]
        print(f"Price range: ${min(prices):.6f} - ${max(prices):.6f}")
        
        # Data quality check
        zero_count = len([d for d in all_data if d.get('close', 0) == 0])
        print(f"Data quality: {len(all_data) - zero_count}/{len(all_data)} valid prices")
        
        return all_data
    else:
        print("❌ No data collected")
        return []

def test_complete_historical_retrieval(symbol="XRP", api_key=None):
    """
    Test complete historical data retrieval using proper pagination
    """
    print(f"\n🏆 COMPLETE HISTORICAL DATA RETRIEVAL FOR {symbol}")
    print("=" * 70)
    
    # Set target based on symbol
    if symbol.upper() == "XRP":
        target_date = XRP_LAUNCH_DATE
        print(f"Target: Get all XRP data from launch ({target_date.date()}) to now")
    else:
        target_date = datetime(2015, 1, 1)  # General crypto start
        print(f"Target: Get all {symbol} data from {target_date.date()} to now")
    
    # Run pagination test
    historical_data = test_cryptocompare_pagination(symbol, api_key, target_date)
    
    if historical_data:
        print(f"\n📊 DATA ANALYSIS")
        print("="*30)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('date').sort_index()
        
        print(f"DataFrame shape: {df.shape}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Check for gaps
        date_diffs = df.index.to_series().diff()
        normal_gaps = (date_diffs == pd.Timedelta(days=1)).sum()
        total_gaps = len(date_diffs) - 1
        
        print(f"Data continuity: {normal_gaps}/{total_gaps} normal 1-day gaps")
        
        if total_gaps > 0:
            continuity_pct = (normal_gaps / total_gaps) * 100
            print(f"Continuity percentage: {continuity_pct:.1f}%")
        
        # Year-by-year breakdown
        print(f"\nYear-by-year data availability:")
        yearly_counts = df.groupby(df.index.year).size()
        for year, count in yearly_counts.items():
            print(f"  {year}: {count} days")
        
        # Price statistics
        print(f"\nPrice statistics:")
        print(f"  All-time low: ${df['close'].min():.6f}")
        print(f"  All-time high: ${df['close'].max():.6f}")
        print(f"  Current price: ${df['close'].iloc[-1]:.6f}")
        
        # Check if this covers the backtesting period (2014-2023)
        backtest_start = datetime(2014, 1, 1)
        backtest_end = datetime(2023, 2, 28)
        
        backtest_data = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
        
        print(f"\n🎯 BACKTESTING COVERAGE ANALYSIS")
        print("="*35)
        print(f"Requested period: {backtest_start.date()} to {backtest_end.date()}")
        print(f"Available data: {len(backtest_data)} days")
        
        if len(backtest_data) > 0:
            actual_start = backtest_data.index.min().date()
            actual_end = backtest_data.index.max().date()
            print(f"Actual coverage: {actual_start} to {actual_end}")
            
            # Calculate coverage percentage
            requested_days = (backtest_end - backtest_start).days
            actual_days = len(backtest_data)
            coverage_pct = (actual_days / requested_days) * 100
            
            print(f"Coverage: {coverage_pct:.1f}% ({actual_days}/{requested_days} days)")
            
            if coverage_pct >= 90:
                print("✅ EXCELLENT: Sufficient data for backtesting!")
            elif coverage_pct >= 70:
                print("⚠️ GOOD: Reasonable data for backtesting")
            else:
                print("❌ INSUFFICIENT: Limited data for backtesting")
        else:
            print("❌ NO DATA: No coverage for backtesting period")
        
        return df
    
    return None

def main():
    print("🔧 ENHANCED DATA FETCH DEBUG TOOLS")
    print("=" * 50)
    print("1. Comprehensive debug (all symbols and sources)")
    print("2. XRP historical coverage test")
    print("3. CryptoCompare detailed test")
    print("4. CryptoCompare with proper parameters")
    print("5. CryptoCompare pagination test")
    print("6. Complete historical retrieval test")
    print("7. Yahoo Finance detailed test")
    print("8. Alternative crypto APIs test")
    print("9. CoinDesk API test")
    
    choice = input("\nChoose option (1-9): ").strip()
    
    if choice == "1":
        print("\n🔄 Running comprehensive debug...")
        comprehensive_data_test()
    elif choice == "2":
        api_key = input("Enter CryptoCompare API key (optional): ").strip() or None
        test_xrp_historical_coverage(api_key)
    elif choice == "3":
        symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
        api_key = input("Enter CryptoCompare API key (optional): ").strip() or None
        test_cryptocompare_detailed(symbol, api_key)
    elif choice == "4":
        symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
        api_key = input("Enter CryptoCompare API key (optional): ").strip() or None
        test_cryptocompare_with_proper_params(symbol, api_key)
    elif choice == "5":
        symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
        api_key = input("Enter CryptoCompare API key (optional): ").strip() or None
        target_date_str = input("Enter target start date (YYYY-MM-DD, default: XRP launch): ").strip()
        
        if target_date_str:
            try:
                target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format, using default")
                target_date = None
        else:
            target_date = None
            
        test_cryptocompare_pagination(symbol, api_key, target_date)
    elif choice == "6":
        symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
        api_key = input("Enter CryptoCompare API key (optional): ").strip() or None
        test_complete_historical_retrieval(symbol, api_key)
    elif choice == "7":
        symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
        test_yahoo_finance_detailed(symbol)
    elif choice == "8":
        symbol = input("Enter symbol (default: XRP): ").strip() or "XRP"
        test_alternative_crypto_apis(symbol)
    elif choice == "9":
        symbol = input("Enter symbol (default: BTC): ").strip() or "BTC"
        test_coindesk_api(symbol)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
