import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator
    """
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available technical indicators
        
        Args:
            data: Stock data DataFrame with OHLCV columns
        
        Returns:
            DataFrame with all technical indicators added
        """
        df = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Moving Averages
        df = TechnicalIndicators.add_moving_averages(df)
        
        # Momentum Indicators
        df = TechnicalIndicators.add_momentum_indicators(df)
        
        # Volatility Indicators
        df = TechnicalIndicators.add_volatility_indicators(df)
        
        # Volume Indicators
        df = TechnicalIndicators.add_volume_indicators(df)
        
        # Trend Indicators
        df = TechnicalIndicators.add_trend_indicators(df)
        
        # Support/Resistance
        df = TechnicalIndicators.add_support_resistance(df)
        
        # Pattern Recognition
        df = TechnicalIndicators.add_patterns(df)
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Hull Moving Average
        df['HMA_20'] = TechnicalIndicators.hull_moving_average(df['Close'], 20)
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_30'] = ta.momentum.rsi(df['Close'], window=30)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # ROC (Rate of Change)
        df['ROC'] = ta.momentum.roc(df['Close'], window=12)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Ultimate Oscillator
        df['Ultimate_Osc'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['KC_Upper'] = kc.keltner_channel_hband()
        df['KC_Middle'] = kc.keltner_channel_mband()
        df['KC_Lower'] = kc.keltner_channel_lband()
        
        # Average True Range
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Donchian Channels
        dc = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['DC_Upper'] = dc.donchian_channel_hband()
        df['DC_Lower'] = dc.donchian_channel_lband()
        df['DC_Middle'] = dc.donchian_channel_mband()
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        # Volume Moving Averages
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Accumulation/Distribution Line
        df['ADL'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Price Trend
        df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        
        # Force Index
        df['Force_Index'] = ta.volume.force_index(df['Close'], df['Volume'])
        
        # Money Flow Index
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        # ADX (Average Directional Index)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['ADX_Pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
        df['ADX_Neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
        
        # Aroon
        aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
        df['Aroon_Up'] = aroon.aroon_up()
        df['Aroon_Down'] = aroon.aroon_down()
        df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.psar_up_indicator(df['High'], df['Low'], df['Close'])
        
        # Ichimoku
        ich = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['Ichimoku_A'] = ich.ichimoku_a()
        df['Ichimoku_B'] = ich.ichimoku_b()
        df['Ichimoku_Base'] = ich.ichimoku_base_line()
        df['Ichimoku_Conversion'] = ich.ichimoku_conversion_line()
        
        # Trix
        df['TRIX'] = ta.trend.trix(df['Close'])
        
        return df
    
    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels"""
        # Pivot Points
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        
        # Local highs and lows
        df['Local_High'] = df['High'].rolling(window=window, center=True).max()
        df['Local_Low'] = df['Low'].rolling(window=window, center=True).min()
        
        # Distance from support/resistance
        df['Dist_From_High'] = (df['Local_High'] - df['Close']) / df['Close']
        df['Dist_From_Low'] = (df['Close'] - df['Local_Low']) / df['Close']
        
        return df
    
    @staticmethod
    def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition indicators"""
        # Doji patterns
        body = abs(df['Close'] - df['Open'])
        shadow_upper = df['High'] - df[['Close', 'Open']].max(axis=1)
        shadow_lower = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        df['Doji'] = (body <= (df['High'] - df['Low']) * 0.1).astype(int)
        
        # Hammer and hanging man
        df['Hammer'] = ((shadow_lower > 2 * body) & 
                       (shadow_upper < 0.1 * body) & 
                       (df['Close'] < df['Open'])).astype(int)
        
        # Engulfing patterns
        df['Bullish_Engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift(1) < df['Open'].shift(1)) &
                                  (df['Open'] < df['Close'].shift(1)) &
                                  (df['Close'] > df['Open'].shift(1))).astype(int)
        
        df['Bearish_Engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift(1) > df['Open'].shift(1)) &
                                  (df['Open'] > df['Close'].shift(1)) &
                                  (df['Close'] < df['Open'].shift(1))).astype(int)
        
        # Gap detection
        df['Gap_Up'] = (df['Open'] > df['Close'].shift(1) * 1.01).astype(int)
        df['Gap_Down'] = (df['Open'] < df['Close'].shift(1) * 0.99).astype(int)
        
        return df
    
    @staticmethod
    def hull_moving_average(series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = series.rolling(window=half_period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        )
        wma_full = series.rolling(window=period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        )
        
        hull_ma = (2 * wma_half - wma_full).rolling(window=sqrt_period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        )
        
        return hull_ma
    
    @staticmethod
    def get_signal_strength(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall signal strength"""
        signals = df.copy()
        
        # Momentum signals
        signals['RSI_Signal'] = np.where(signals['RSI'] < 30, 1, 
                                np.where(signals['RSI'] > 70, -1, 0))
        
        signals['MACD_Signal_Ind'] = np.where(signals['MACD'] > signals['MACD_Signal'], 1, -1)
        
        # Trend signals
        signals['MA_Signal'] = np.where(signals['Close'] > signals['SMA_50'], 1, -1)
        
        # Volatility signals
        signals['BB_Signal'] = np.where(signals['Close'] < signals['BB_Lower'], 1,
                              np.where(signals['Close'] > signals['BB_Upper'], -1, 0))
        
        # Combine signals
        signal_columns = ['RSI_Signal', 'MACD_Signal_Ind', 'MA_Signal', 'BB_Signal']
        signals['Combined_Signal'] = signals[signal_columns].sum(axis=1)
        
        return signals
