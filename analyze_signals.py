import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Any, Optional

class MarketCondition(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"

@dataclass
class SignalConfig:
    use_dynamic_weights: bool = True
    enable_logging: bool = True
    confidence_boost_threshold: float = 0.8
    max_histogram_weight: float = 2.0

class AdaptiveTechnicalSignalAnalyzer:
    """
    High-performance technical signal analyzer with vectorized operations.
    Optimized for speed, accuracy, and financial logic robustness.
    """
    
    # Class-level constants for maximum performance
    _SIGNAL_COLUMNS = frozenset(['MACD_Trade_Signal', 'OBV_Signal', 'BB_Signal', 
                                  'Stoch_Signal', 'ADX_Signal'])
    _SCORE_SUFFIXES = ('_buy_score', '_sell_score')
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        if self.config.enable_logging:
            self._setup_logging()
        
        # Single source of truth for weights - no duplication
        self._weights = {
            'Position_vs_SMA20': 1.2,
            'Position_vs_SMA50': 1.5,
            'EMA_signal': 1.8,
            'MACD_Trade_Signal': 2.2,
            'MACD_Histogram': 1.0,
            'Stoch_Signal': 1.3,
            'RSI_Signal': 1.5,
            'OBV_Signal': 1.4,
            'BB_Signal': 1.6,
            'ADX_Signal': 1.7,
            'SR_Zone': 1.3,
        }
        
        self.performance_log = []

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger('SignalAnalyzer')

    def vectorized_signal_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ultra-optimized vectorized signal calculation.
        
        Performance improvements:
        - Single-pass column operations
        - Minimal DataFrame copies
        - Pre-allocated arrays
        - Batch string normalization
        - Efficient boolean indexing
        """
        # Input validation
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        required_cols = {'Close', 'High', 'Low'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
        
        # Extract numpy arrays once for speed (avoids pandas overhead)
        close_vals = df['Close'].values
        n_rows = len(close_vals)
        
        # Normalize signal columns in batch (vectorized string operations)
        signal_cols_present = self._SIGNAL_COLUMNS & set(df.columns)
        if signal_cols_present:
            for col in signal_cols_present:
                # Single vectorized operation instead of method chaining
                df[col] = df[col].astype(str).str.strip().str.capitalize()

        # ===== MOVING AVERAGES =====
        # Use numpy for rolling calculations where possible
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

        sma_20_vals = df['SMA_20'].values
        sma_50_vals = df['SMA_50'].values
        
        # Vectorized position calculation (single operation)
        df['Position_vs_SMA20'] = np.where(close_vals > sma_20_vals, 'Above', 'Below')
        df['Position_vs_SMA50'] = np.where(close_vals > sma_50_vals, 'Above', 'Below')

        # ===== PRE-ALLOCATE SCORE ARRAYS =====
        # Allocate all score columns as numpy arrays first (faster than DataFrame operations)
        score_arrays = {
            'trend_buy_score': np.zeros(n_rows, dtype=np.float32),
            'trend_sell_score': np.zeros(n_rows, dtype=np.float32),
            'momentum_buy_score': np.zeros(n_rows, dtype=np.float32),
            'momentum_sell_score': np.zeros(n_rows, dtype=np.float32),
            'volume_buy_score': np.zeros(n_rows, dtype=np.float32),
            'volume_sell_score': np.zeros(n_rows, dtype=np.float32),
            'volatility_buy_score': np.zeros(n_rows, dtype=np.float32),
            'volatility_sell_score': np.zeros(n_rows, dtype=np.float32),
            'strength_buy_score': np.zeros(n_rows, dtype=np.float32),
            'strength_sell_score': np.zeros(n_rows, dtype=np.float32),
            'sr_buy_score': np.zeros(n_rows, dtype=np.float32),
            'sr_sell_score': np.zeros(n_rows, dtype=np.float32),
        }

        # ===== SUPPORT/RESISTANCE ZONES =====
        if {'S1', 'R1'}.issubset(df.columns):
            r1_vals = df['R1'].values
            s1_vals = df['S1'].values
            
            # Vectorized zone detection with tolerance
            at_resistance = close_vals >= (r1_vals * 0.995)
            at_support = close_vals <= (s1_vals * 1.005)
            
            # Initialize as array, then assign (faster than df operations)
            sr_zone = np.full(n_rows, 'None', dtype=object)
            sr_zone[at_resistance] = 'Resistance'
            sr_zone[at_support] = 'Support'
            df['SR_Zone'] = sr_zone
            
            # Vectorized scoring
            w_sr = self._weights['SR_Zone']
            score_arrays['sr_buy_score'][at_support] = w_sr
            score_arrays['sr_sell_score'][at_resistance] = w_sr
        else:
            df['SR_Zone'] = 'None'

        # ===== TREND INDICATORS =====
        w_sma20 = self._weights['Position_vs_SMA20']
        w_sma50 = self._weights['Position_vs_SMA50']
        
        # Create masks once (boolean arrays)
        above_sma20 = df['Position_vs_SMA20'].values == 'Above'
        below_sma20 = ~above_sma20
        above_sma50 = df['Position_vs_SMA50'].values == 'Above'
        below_sma50 = ~above_sma50
        
        # Vectorized scoring (in-place operations on numpy arrays)
        score_arrays['trend_buy_score'][above_sma20] += w_sma20
        score_arrays['trend_sell_score'][below_sma20] += w_sma20
        score_arrays['trend_buy_score'][above_sma50] += w_sma50
        score_arrays['trend_sell_score'][below_sma50] += w_sma50
        
        if 'EMA_signal' in df.columns:
            w_ema = self._weights['EMA_signal']
            ema_vals = df['EMA_signal'].values
            
            golden_cross = ema_vals == 'Golden Cross'
            death_cross = ema_vals == 'Death Cross'
            
            score_arrays['trend_buy_score'][golden_cross] += w_ema
            score_arrays['trend_sell_score'][death_cross] += w_ema

        # ===== MOMENTUM INDICATORS =====
        # RSI Signal - combined signal generation and scoring
        if 'RSI' in df.columns:
            rsi_vals = df['RSI'].values
            w_rsi = self._weights['RSI_Signal']
            
            # Vectorized RSI signal generation
            rsi_signal = np.full(n_rows, 'Hold', dtype=object)
            rsi_buy = rsi_vals < 30
            rsi_sell = rsi_vals > 70
            rsi_signal[rsi_buy] = 'Buy'
            rsi_signal[rsi_sell] = 'Sell'
            df['RSI_Signal'] = rsi_signal
            
            score_arrays['momentum_buy_score'][rsi_buy] += w_rsi
            score_arrays['momentum_sell_score'][rsi_sell] += w_rsi
        
        # MACD Histogram - optimized with direct array operations
        if 'MACD_Histogram' in df.columns:
            w_hist = self._weights['MACD_Histogram']
            hist_vals = df['MACD_Histogram'].values
            
            hist_positive = hist_vals > 0
            hist_negative = hist_vals < 0
            
            score_arrays['momentum_buy_score'][hist_positive] += w_hist
            score_arrays['momentum_sell_score'][hist_negative] += w_hist
        
        # MACD Trade Signal
        if 'MACD_Trade_Signal' in df.columns:
            w_macd = self._weights['MACD_Trade_Signal']
            macd_vals = df['MACD_Trade_Signal'].values
            
            macd_buy = macd_vals == 'Buy'
            macd_sell = macd_vals == 'Sell'
            
            score_arrays['momentum_buy_score'][macd_buy] += w_macd
            score_arrays['momentum_sell_score'][macd_sell] += w_macd
        
        # Stochastic Signal
        if 'Stoch_Signal' in df.columns:
            w_stoch = self._weights['Stoch_Signal']
            stoch_vals = df['Stoch_Signal'].values
            
            stoch_buy = stoch_vals == 'Buy'
            stoch_sell = stoch_vals == 'Sell'
            
            score_arrays['momentum_buy_score'][stoch_buy] += w_stoch
            score_arrays['momentum_sell_score'][stoch_sell] += w_stoch

        # ===== VOLUME INDICATORS =====
        if 'OBV_Signal' in df.columns:
            w_obv = self._weights['OBV_Signal']
            obv_vals = df['OBV_Signal'].values
            
            obv_buy = obv_vals == 'Buy'
            obv_sell = obv_vals == 'Sell'
            
            score_arrays['volume_buy_score'][obv_buy] += w_obv
            score_arrays['volume_sell_score'][obv_sell] += w_obv

        # ===== VOLATILITY INDICATORS =====
        if 'BB_Signal' in df.columns:
            w_bb = self._weights['BB_Signal']
            bb_vals = df['BB_Signal'].values
            
            bb_buy = bb_vals == 'Buy'
            bb_sell = bb_vals == 'Sell'
            
            score_arrays['volatility_buy_score'][bb_buy] += w_bb
            score_arrays['volatility_sell_score'][bb_sell] += w_bb

        # ===== STRENGTH INDICATORS =====
        if 'ADX_Signal' in df.columns:
            w_adx = self._weights['ADX_Signal']
            adx_vals = df['ADX_Signal'].values
            
            adx_buy = adx_vals == 'Buy'
            adx_sell = adx_vals == 'Sell'
            
            score_arrays['strength_buy_score'][adx_buy] += w_adx
            score_arrays['strength_sell_score'][adx_sell] += w_adx

        # ===== ASSIGN ARRAYS BACK TO DATAFRAME =====
        # Batch assignment (much faster than individual column assignments)
        for col_name, arr in score_arrays.items():
            df[col_name] = arr

        # ===== AGGREGATE SCORES =====
        # Use numpy arrays for summation (faster than pandas)
        buy_scores = np.column_stack([
            score_arrays['trend_buy_score'],
            score_arrays['momentum_buy_score'],
            score_arrays['volume_buy_score'],
            score_arrays['volatility_buy_score'],
            score_arrays['strength_buy_score'],
            score_arrays['sr_buy_score']
        ])
        
        sell_scores = np.column_stack([
            score_arrays['trend_sell_score'],
            score_arrays['momentum_sell_score'],
            score_arrays['volume_sell_score'],
            score_arrays['volatility_sell_score'],
            score_arrays['strength_sell_score'],
            score_arrays['sr_sell_score']
        ])
        
        df['Buy_Score'] = buy_scores.sum(axis=1)
        df['Sell_Score'] = sell_scores.sum(axis=1)
        df['Net_Score'] = df['Buy_Score'] - df['Sell_Score']

        # ===== IMPORTANT SCORES (excluding volatility) =====
        important_buy = np.column_stack([
            score_arrays['trend_buy_score'],
            score_arrays['momentum_buy_score'],
            score_arrays['volume_buy_score'],
            score_arrays['strength_buy_score'],
            score_arrays['sr_buy_score']
        ])
        
        important_sell = np.column_stack([
            score_arrays['trend_sell_score'],
            score_arrays['momentum_sell_score'],
            score_arrays['volume_sell_score'],
            score_arrays['strength_sell_score'],
            score_arrays['sr_sell_score']
        ])
        
        df['Important_Buy_Score'] = important_buy.sum(axis=1)
        df['Important_Sell_Score'] = important_sell.sum(axis=1)
        
        # Calculate net score and signal (vectorized)
        important_net = df['Important_Buy_Score'].values - df['Important_Sell_Score'].values
        df['Important_Net_Score'] = important_net
        
        # Vectorized signal classification (single operation)
        df['Important_Signal'] = np.where(
            important_net > 0, 'Buy',
            np.where(important_net < 0, 'Sell', 'Hold')
        )

        return df


def analyze_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function for technical signal analysis.
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        
    Returns:
        DataFrame with calculated signals and scores
    """
    try:
        analyzer = AdaptiveTechnicalSignalAnalyzer(
            SignalConfig(enable_logging=False)
        )
        return analyzer.vectorized_signal_calculation(df)
    except Exception as e:
        # Log error but don't crash - return original DataFrame
        logging.error(f"Signal analysis failed: {str(e)}")
        return df