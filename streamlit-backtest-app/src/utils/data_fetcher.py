"""
Production data fetcher with proper pagination support
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf


class DataFetcher:
    """Enhanced data fetcher with pagination support for complete historical data"""
    
    def __init__(self, cryptocompare_api_key=None):
        self.cryptocompare_api_key = cryptocompare_api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2"
        
    def fetch_cryptocompare_paginated(self, symbol, start_date, end_date=None, interval='1d'):
        """
        Fetch complete historical data using pagination
        """
        if end_date is None:
            end_date = datetime.now()
            
        print(f"🔄 Fetching {symbol} data from {start_date.date()} to {end_date.date()} using pagination...")
        
        # Map intervals
        interval_map = {
            '1d': 'histoday',
            '1h': 'histohour', 
            '4h': 'histohour',  # Use hourly with aggregate
            '5m': 'histominute'
        }
        
        endpoint = interval_map.get(interval, 'histoday')
        url = f"{self.base_url}/{endpoint}"
        
        headers = {"authorization": f"Apikey {self.cryptocompare_api_key}"} if self.cryptocompare_api_key else {}
        
        all_data = []
        current_to_ts = int(end_date.timestamp())
        batch_count = 0
        max_batches = 50  # Allow more batches for complete historical data
        
        # Set aggregate for 4h interval
        aggregate = 4 if interval == '4h' else 1
        
        while current_to_ts > start_date.timestamp() and batch_count < max_batches:
            batch_count += 1
            
            params = {
                "fsym": symbol,
                "tsym": "USD",
                "limit": 2000,
                "toTs": current_to_ts,
                "aggregate": aggregate
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('Response') == 'Success':
                        hist_data = data.get('Data', {}).get('Data', [])
                        
                        if hist_data:
                            # Filter valid data
                            valid_data = [d for d in hist_data if d.get('close', 0) > 0]
                            
                            if valid_data:
                                first_date = datetime.fromtimestamp(valid_data[0]['time'])
                                last_date = datetime.fromtimestamp(valid_data[-1]['time'])
                                
                                print(f"  Batch {batch_count}: {len(valid_data)} points ({first_date.date()} to {last_date.date()})")
                                
                                # Add new data (avoid duplicates)
                                existing_timestamps = {d['time'] for d in all_data}
                                new_data = [d for d in valid_data if d['time'] not in existing_timestamps]
                                all_data.extend(new_data)
                                
                                # Check if we've reached target
                                if first_date <= start_date:
                                    print(f"  ✅ Reached target start date!")
                                    break
                                
                                # Update for next batch
                                time_delta = 86400 if interval == '1d' else 3600  # 1 day or 1 hour
                                current_to_ts = valid_data[0]['time'] - time_delta
                            else:
                                print(f"  ❌ No valid data in batch {batch_count}")
                                break
                        else:
                            print(f"  ❌ No data in batch {batch_count}")
                            break
                    else:
                        error_msg = data.get('Message', 'Unknown error')
                        print(f"  ❌ API Error: {error_msg}")
                        
                        if 'rate limit' in error_msg.lower():
                            print("  ⏳ Rate limited, waiting...")
                            time.sleep(10)
                            continue
                        else:
                            break
                else:
                    print(f"  ❌ HTTP Error: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"  ❌ Request error: {e}")
                break
            
            # Rate limiting
            time.sleep(1)
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            
            # Filter to exact date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            # Prepare OHLCV format
            df = df.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volumeto': 'Volume'
            })
            
            # Ensure Volume column exists
            if 'Volume' not in df.columns:
                df['Volume'] = 0
                
            # Set index and select required columns
            df = df.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.sort_index()
            
            print(f"✅ Retrieved {len(df)} data points from {df.index.min().date()} to {df.index.max().date()}")
            return df
        else:
            print(f"❌ No data retrieved")
            return pd.DataFrame()
    
    def fetch_yahoo_finance(self, symbol, start_date, end_date=None, interval='1d'):
        """
        Fetch data from Yahoo Finance
        """
        if end_date is None:
            end_date = datetime.now()
            
        print(f"🔄 Fetching {symbol} from Yahoo Finance...")
        
        # Try different symbol formats
        symbol_variants = [f"{symbol}-USD", symbol, f"{symbol}USD"]
        
        for variant in symbol_variants:
            try:
                data = yf.download(
                    variant,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    progress=False
                )
                
                if not data.empty:
                    # Handle MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] for col in data.columns]
                    
                    # Ensure we have required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_cols:
                        if col not in data.columns:
                            data[col] = 0
                    
                    data = data[required_cols]
                    
                    print(f"✅ Yahoo Finance: {len(data)} points from {data.index.min().date()} to {data.index.max().date()}")
                    return data
                    
            except Exception as e:
                print(f"  Failed {variant}: {e}")
                continue
        
        print(f"❌ Yahoo Finance failed for all symbol variants")
        return pd.DataFrame()
    
    def fetch_historical_data(self, symbol, start_date, end_date=None, interval='1d'):
        """
        Main method to fetch historical data with fallback strategy
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        print(f"\n📊 Fetching {symbol} historical data")
        print(f"Period: {start_date.date()} to {(end_date or datetime.now()).date()}")
        print(f"Interval: {interval}")
        
        # Detect if crypto symbol
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'DOGE', 'LTC']
        is_crypto = any(symbol.upper().startswith(crypto) for crypto in crypto_symbols)
        
        df = pd.DataFrame()
        
        if is_crypto:
            # Strategy 1: Try CryptoCompare with pagination for crypto
            df = self.fetch_cryptocompare_paginated(symbol, start_date, end_date, interval)
        
        # Strategy 2: Fallback to Yahoo Finance if insufficient data or not crypto
        if df.empty or len(df) < 100:
            print("🔄 Trying Yahoo Finance...")
            df = self.fetch_yahoo_finance(symbol, start_date, end_date, interval)
        
        # Final validation
        if not df.empty:
            # Remove any remaining invalid data
            df = df[(df['Close'] > 0) & (df['Open'] > 0)].copy()
            df = df.sort_index()
            
            # Coverage analysis
            requested_days = (end_date - start_date).days if end_date else (datetime.now() - start_date).days
            actual_days = len(df)
            coverage = (actual_days / requested_days) * 100 if requested_days > 0 else 0
            
            print(f"📈 Final result: {len(df)} data points")
            print(f"📅 Coverage: {coverage:.1f}% ({actual_days}/{requested_days} days)")
            
            return df
        else:
            print("❌ No data retrieved from any source")
            return pd.DataFrame()


# Convenience function for backward compatibility
def fetch_cryptocompare_data(symbol, start_date, end_date=None, interval="1d", api_key=None):
    """
    Convenience function that maintains compatibility with existing code
    """
    fetcher = DataFetcher(cryptocompare_api_key=api_key)
    return fetcher.fetch_historical_data(symbol, start_date, end_date, interval)
