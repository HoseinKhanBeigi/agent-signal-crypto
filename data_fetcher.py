"""
CCXT-based data fetcher for cryptocurrency price data
Optimized for 5-15 minute timeframes
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
from config import EXCHANGE_NAME, TIMEFRAME, SYMBOL, DATA_LIMIT


class CCXTDataFetcher:
    """Fetches cryptocurrency data using CCXT library"""
    
    def __init__(self, exchange_name=EXCHANGE_NAME):
        """
        Initialize CCXT exchange
        
        Args:
            exchange_name: Name of the exchange (binance, coinbase, etc.)
        """
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange()
        
    def _init_exchange(self):
        """Initialize exchange connection"""
        exchange_class = getattr(ccxt, self.exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        return exchange
    
    def fetch_ohlcv(self, symbol=SYMBOL, timeframe=TIMEFRAME, limit=DATA_LIMIT):
        """
        Fetch OHLCV data
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe ('5m', '15m', '1h', etc.)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV columns: timestamp, open, high, low, close, volume
        """
        try:
            print(f"Fetching {limit} candles of {symbol} on {timeframe} timeframe...")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validate data
            df = self._clean_data(df)
            
            print(f"✓ Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None
    
    def _clean_data(self, df):
        """Clean and validate data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove zero volume candles
        df = df[df['volume'] > 0]
        
        # Forward fill missing values
        df = df.ffill()
        
        # Drop any remaining NaN
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_index()
        
        return df
    
    def fetch_latest(self, symbol=SYMBOL, timeframe=TIMEFRAME, limit=100):
        """Fetch latest candles for real-time prediction"""
        return self.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    def get_multiple_pairs(self, symbols, timeframe=TIMEFRAME, limit=DATA_LIMIT):
        """
        Fetch data for multiple trading pairs
        
        Args:
            symbols: List of trading pairs
            timeframe: Timeframe
            limit: Number of candles per pair
            
        Returns:
            Dictionary of DataFrames
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.fetch_ohlcv(symbol, timeframe, limit)
            time.sleep(0.5)  # Rate limiting
        return data


if __name__ == "__main__":
    # Test data fetcher
    fetcher = CCXTDataFetcher()
    data = fetcher.fetch_ohlcv()
    
    if data is not None:
        print(f"\nData shape: {data.shape}")
        print(f"\nFirst 5 rows:")
        print(data.head())
        print(f"\nLast 5 rows:")
        print(data.tail())
        print(f"\nData statistics:")
        print(data.describe())

