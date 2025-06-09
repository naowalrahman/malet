import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataFetcher:
    """
    Handles fetching and preprocessing stock data using yfinance
    """
    
    def __init__(self):
        self.cache = {}
    
    def fetch_daily_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """
        Fetch daily stock data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Period to fetch data for ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean the data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Add symbol column
            data['Symbol'] = symbol
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical daily data for a custom date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            data['Symbol'] = symbol
            
            return data
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
        
    # Add a convenience method for backwards compatibility
    # def fetch_daily_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
    #     """Backwards compatibility - redirects to fetch_daily_data"""
    #     return self.fetch_daily_data(symbol, period)
    
    def fetch_multiple_symbols(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Period to fetch data for
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict = {}
        
        for symbol in symbols:
            data = self.fetch_daily_data(symbol, period)
            if not data.empty:
                data_dict[symbol] = data
        
        return data_dict
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get basic information about a stock
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'name': symbol}
    
    def prepare_data_for_ml(self, data: pd.DataFrame, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning by creating sequences
        
        Args:
            data: Stock data DataFrame
            look_back: Number of time steps to look back
        
        Returns:
            Tuple of (X, y) arrays for training
        """
        # Use closing prices for prediction
        prices = data['Close'].values
        
        # Normalize the data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        X, y = [], []
        
        for i in range(look_back, len(prices_scaled)):
            X.append(prices_scaled[i-look_back:i])
            y.append(prices_scaled[i])
        
        return np.array(X), np.array(y), scaler
    
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and features to the data
        
        Args:
            data: Stock data DataFrame
        
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Price position in range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Gap features
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        return df.dropna()

# Utility functions
def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol exists and has data
    
    Args:
        symbol: Stock symbol to validate
    
    Returns:
        True if symbol is valid, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return not data.empty
    except:
        return False

def get_popular_symbols() -> List[str]:
    """
    Get a list of popular stock symbols for testing
    
    Returns:
        List of popular stock symbols
    """
    return [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'META', 'NFLX', 'NVDA', 'AMD', 'INTC',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI'
    ]
