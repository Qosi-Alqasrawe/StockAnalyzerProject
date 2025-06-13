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
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        if self.config.enable_logging:
            self._setup_logging()
        self.base_weights = {
            'trend_indicators': {
                'Position_vs_SMA20': 1.2,
                'Position_vs_SMA50': 1.5,
                'EMA_signal': 1.8,
            },
            'momentum_indicators': {
                'MACD_Trade_Signal': 2.2,
                'MACD_Histogram': 1.0,
                'Stoch_Signal': 1.3,
                'RSI_Signal': 1.5,
            },
            'volume_indicators': {
                'OBV_Signal': 1.4,
            },
            'volatility_indicators': {
                'BB_Signal': 1.6,
            },
            'strength_indicators': {
                'ADX_Signal': 1.7,
            },
            'support_resistance': {
                'SR_Zone': 1.3,
            }
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
        # Normalize signal columns
        for col in ['MACD_Trade_Signal', 'OBV_Signal', 'BB_Signal', 'Stoch_Signal', 'ADX_Signal']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.capitalize()

        # Compute SMAs
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Position vs SMAs
        df['Position_vs_SMA20'] = np.where(df['Close'] > df['SMA_20'], 'Above', 'Below')
        df['Position_vs_SMA50'] = np.where(df['Close'] > df['SMA_50'], 'Above', 'Below')

        # Initialize score columns
        sc_cols = ['trend_buy_score', 'trend_sell_score',
                   'momentum_buy_score', 'momentum_sell_score',
                   'volume_buy_score', 'volume_sell_score',
                   'volatility_buy_score', 'volatility_sell_score',
                   'strength_buy_score', 'strength_sell_score',
                   'sr_buy_score', 'sr_sell_score']
        for sc in sc_cols:
            df[sc] = 0.0

        # Support/Resistance zone
        if all(c in df.columns for c in ['Close','S1','R1']):
            def sr_zone(r):
                price, s1, r1 = r['Close'], r['S1'], r['R1']
                if price >= r1 * 0.995: return 'Resistance'
                if price <= s1 * 1.005: return 'Support'
                return 'None'
            df['SR_Zone'] = df.apply(sr_zone, axis=1)
        else:
            df['SR_Zone'] = 'None'
        df.loc[df['SR_Zone']=='Support','sr_buy_score'] += self.base_weights['support_resistance']['SR_Zone']
        df.loc[df['SR_Zone']=='Resistance','sr_sell_score'] += self.base_weights['support_resistance']['SR_Zone']

        # Trend scoring (position and EMA)
        df.loc[df['Position_vs_SMA20']=='Above','trend_buy_score'] += self.base_weights['trend_indicators']['Position_vs_SMA20']
        df.loc[df['Position_vs_SMA20']=='Below','trend_sell_score'] += self.base_weights['trend_indicators']['Position_vs_SMA20']
        df.loc[df['Position_vs_SMA50']=='Above','trend_buy_score'] += self.base_weights['trend_indicators']['Position_vs_SMA50']
        df.loc[df['Position_vs_SMA50']=='Below','trend_sell_score'] += self.base_weights['trend_indicators']['Position_vs_SMA50']
        if 'EMA_signal' in df.columns:
            df.loc[df['EMA_signal']=='Golden Cross','trend_buy_score'] += self.base_weights['trend_indicators']['EMA_signal']
            df.loc[df['EMA_signal']=='Death Cross','trend_sell_score'] += self.base_weights['trend_indicators']['EMA_signal']

        # Momentum scoring
        if 'RSI' in df.columns:
            df['RSI_Signal'] = np.where(df['RSI']<30,'Buy',np.where(df['RSI']>70,'Sell','Hold'))
            df.loc[df['RSI_Signal']=='Buy','momentum_buy_score'] += self.base_weights['momentum_indicators']['RSI_Signal']
            df.loc[df['RSI_Signal']=='Sell','momentum_sell_score'] += self.base_weights['momentum_indicators']['RSI_Signal']
        df.loc[df['MACD_Histogram']>0,'momentum_buy_score'] += self.base_weights['momentum_indicators']['MACD_Histogram']
        df.loc[df['MACD_Histogram']<0,'momentum_sell_score'] += self.base_weights['momentum_indicators']['MACD_Histogram']
        if 'Stoch_Signal' in df.columns:
            df.loc[df['Stoch_Signal']=='Buy','momentum_buy_score'] += self.base_weights['momentum_indicators']['Stoch_Signal']
            df.loc[df['Stoch_Signal']=='Sell','momentum_sell_score'] += self.base_weights['momentum_indicators']['Stoch_Signal']

        # Volume scoring
        if 'OBV_Signal' in df.columns:
            df.loc[df['OBV_Signal']=='Buy','volume_buy_score'] += self.base_weights['volume_indicators']['OBV_Signal']
            df.loc[df['OBV_Signal']=='Sell','volume_sell_score'] += self.base_weights['volume_indicators']['OBV_Signal']

        # Volatility scoring
        if 'BB_Signal' in df.columns:
            df.loc[df['BB_Signal']=='Buy','volatility_buy_score'] += self.base_weights['volatility_indicators']['BB_Signal']
            df.loc[df['BB_Signal']=='Sell','volatility_sell_score'] += self.base_weights['volatility_indicators']['BB_Signal']

        # Strength scoring
        if 'ADX_Signal' in df.columns:
            df.loc[df['ADX_Signal']=='Buy','strength_buy_score'] += self.base_weights['strength_indicators']['ADX_Signal']
            df.loc[df['ADX_Signal']=='Sell','strength_sell_score'] += self.base_weights['strength_indicators']['ADX_Signal']

        # Aggregate scores
        df['Buy_Score'] = df.filter(like='_buy_score').sum(axis=1)
        df['Sell_Score'] = df.filter(like='_sell_score').sum(axis=1)
        df['Net_Score'] = df['Buy_Score'] - df['Sell_Score']

        # Important aggregate
        buy_cols = ['trend_buy_score','momentum_buy_score','volume_buy_score','strength_buy_score','sr_buy_score']
        sell_cols= ['trend_sell_score','momentum_sell_score','volume_sell_score','strength_sell_score','sr_sell_score']
        for col in buy_cols+sell_cols:
            if col not in df.columns:
                df[col] = 0.0
        df['Important_Buy_Score'] = df[buy_cols].sum(axis=1)
        df['Important_Sell_Score']= df[sell_cols].sum(axis=1)
        df['Important_Net_Score'] = df['Important_Buy_Score'] - df['Important_Sell_Score']
        df['Important_Signal']    = np.where(df['Important_Net_Score']>0,'Buy',np.where(df['Important_Net_Score']<0,'Sell','Hold'))

        return df

# Wrapper

def analyze_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    return AdaptiveTechnicalSignalAnalyzer().vectorized_signal_calculation(df)
