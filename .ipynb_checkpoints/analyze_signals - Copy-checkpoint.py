import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Any, Optional


class MarketCondition(Enum):
    """حالات السوق المختلفة"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class SignalConfig:
    """إعدادات قابلة للتخصيص"""
    use_dynamic_weights: bool = True
    enable_logging: bool = True
    confidence_boost_threshold: float = 0.8
    max_histogram_weight: float = 2.0


class AdaptiveTechnicalSignalAnalyzer:
    """
    محلل إشارات تقنية متكيف مع معايرة ديناميكية وتحسينات الأداء
    """
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        if self.config.enable_logging:
            self._setup_logging()
        self.base_weights = {
            'trend_indicators': {
                'Position_vs_SMA20': 1.2,
                'Position_vs_SMA50': 1.5,
                'Trend_Signal': 2.0,
            },
            'momentum_indicators': {
                'Market_Condition': 1.8,
                'MACD_Trade_Signal': 2.2,
                'MACD_Histogram': 1.0,
                'Stoch_Signal': 1.3,
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
                'Pivot_Signal': 1.1,
                'Fib_Signal': 1.2,
                'SR_Zone': 1.3,
            },
            'combined': {
                'Combined_Signal': 2.5,
            }
        }
        self.decision_thresholds = {
            'strong_buy': 0.75,
            'buy': 0.6,
            'weak_hold': 0.55,
            'moderate_hold': 0.45,
            'sell': 0.4,
            'strong_sell': 0.25
        }
        self.performance_log = []

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger('SignalAnalyzer')

    def detect_market_condition(self, row: pd.Series) -> MarketCondition:
        try:
            adx = float(row.get('ADX', np.nan))
            atr = float(row.get('ATR_14', np.nan))
            price = float(row.get('Close', np.nan))
            atr_ratio = atr / price if price else np.nan
            bb_signal = str(row.get('BB_Signal', ''))

            if not np.isnan(atr_ratio):
                if atr_ratio > 0.03:
                    return MarketCondition.VOLATILE
                if atr_ratio < 0.01 and 'Squeeze' in bb_signal:
                    return MarketCondition.CALM
            if adx > 25:
                return MarketCondition.TRENDING
            elif adx > 20:
                return MarketCondition.RANGING
            return MarketCondition.RANGING
        except:
            return MarketCondition.VOLATILE

    def get_adaptive_weights(self, row: pd.Series, market_condition: MarketCondition) -> Dict:
        weights = {cat: inds.copy() for cat, inds in self.base_weights.items()}
        if market_condition == MarketCondition.TRENDING:
            for k in weights['trend_indicators']:
                weights['trend_indicators'][k] *= 1.3
            for k in weights['support_resistance']:
                weights['support_resistance'][k] *= 0.8
        elif market_condition == MarketCondition.RANGING:
            for k in weights['support_resistance']:
                weights['support_resistance'][k] *= 1.4
        elif market_condition == MarketCondition.VOLATILE:
            for k in weights['volatility_indicators']:
                weights['volatility_indicators'][k] *= 1.2
        return weights

    def vectorized_signal_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) حساب مواقع السعر مقارنة بالـ SMA
        df['Position_vs_SMA20'] = np.where(df['Close'] > df['Close'].rolling(20).mean(), 'Above', 'Below')
        df['Position_vs_SMA50'] = np.where(df['Close'] > df['Close'].rolling(50).mean(), 'Above', 'Below')

        # 2) تهيئة درجات الأسعار
        df['trend_buy_score'] = 0.0
        df['trend_sell_score'] = 0.0
        df['sr_buy_score']    = 0.0
        df['sr_sell_score']   = 0.0
        # ... (تهيئة بقية فئات الدرجات كما في الكود الأصلي)

        # 3) تعيين SR_Zone بناءً على sr_zones المخزنة في attrs
        zones = df.attrs.get('sr_zones', [])
        def map_zone(price):
            for low, high in zones:
                if low <= price <= high:
                    # نختار Support أو Resistance بناءً على قرب السعر
                    return ('Support' if price - low < high - price else 'Resistance')
            return 'None'
        df['SR_Zone'] = df['Close'].apply(map_zone)

        # 4) إضافة وزن SR_Zone
        df.loc[df['SR_Zone']=='Support',  'sr_buy_score']  += self.base_weights['support_resistance']['SR_Zone']
        df.loc[df['SR_Zone']=='Resistance','sr_sell_score'] += self.base_weights['support_resistance']['SR_Zone']

        # 5) إضافة وزن Position_vs_SMA
        df.loc[df['Position_vs_SMA20']=='Above',  'trend_buy_score']  += self.base_weights['trend_indicators']['Position_vs_SMA20']
        df.loc[df['Position_vs_SMA20']=='Below',  'trend_sell_score'] += self.base_weights['trend_indicators']['Position_vs_SMA20']
        df.loc[df['Position_vs_SMA50']=='Above',  'trend_buy_score']  += self.base_weights['trend_indicators']['Position_vs_SMA50']
        df.loc[df['Position_vs_SMA50']=='Below',  'trend_sell_score'] += self.base_weights['trend_indicators']['Position_vs_SMA50']

        # 6) بقية الحسابات لبقية المؤشرات... (MACD, Stoch, OBV, BB, ADX, Pivot, Fib, Combined)

        # 7) جمع الدرجات النهائية
        df['Buy_Score']  = df.filter(like='_buy_score').sum(axis=1)
        df['Sell_Score'] = df.filter(like='_sell_score').sum(axis=1)

        return df

    # بقية طرق _analyze_* تُبقى كما هي


# دالة بسيطة لاستخدام الإشارات
config = SignalConfig(enable_logging=False)
_analyzer = AdaptiveTechnicalSignalAnalyzer(config)

def analyze_technical_signals(row: pd.Series) -> Tuple[float, float]:
    df_single = pd.DataFrame([row])
    df_signals = _analyzer.vectorized_signal_calculation(df_single)
    if df_signals.empty or 0 not in df_signals.index or 'Buy_Score' not in df_signals.columns or 'Sell_Score' not in df_signals.columns:
        return 0.0, 0.0
    return df_signals.at[0, 'Buy_Score'], df_signals.at[0, 'Sell_Score']