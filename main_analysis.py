import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from import_fetch_technical import fetch_technical_data
from fetch_fundamental import fetch_fundamental_data
from compute_indicators import calculate_technical_indicators
from analyze_signals import analyze_technical_signals
from analyze_financial import analyze_financial_performance
from price_targets import calculate_price_targets


# ===============================
# Enhanced SWOT Analysis Classes
# ===============================

class AnalysisCategory(Enum):
    FINANCIAL = "Financial"
    TECHNICAL = "Technical"
    MARKET = "Market"
    FUNDAMENTAL = "Fundamental"
    RISK = "Risk"


@dataclass
class SWOTItem:
    """SWOT item with category, weight, impact, and confidence"""
    description: str
    category: AnalysisCategory
    weight: float  # 1-5 (importance)
    impact: float  # 1-5 (impact)
    confidence: float  # 0-100% (confidence level)
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score"""
        return (self.weight * self.impact * self.confidence) / 100


class ProfessionalSWOTAnalyzer:
    """
    Professional comprehensive SWOT analyzer with optimized performance.
    
    Improvements:
    - Pre-defined thresholds for O(1) lookup
    - Efficient batch data extraction
    - Vectorized scoring where possible
    - Reduced code duplication
    """
    
    # Class-level constants for threshold definitions
    _FINANCIAL_EXCELLENT = {'score': 80, 'roe': 15, 'debt_ratio': 0.3, 'current_ratio': 2.0}
    _FINANCIAL_GOOD = {'score': 70, 'roe': 10, 'debt_ratio': 0.5, 'current_ratio': 1.5}
    _FINANCIAL_AVERAGE = {'score': 60, 'roe': 5, 'debt_ratio': 0.7, 'current_ratio': 1.0}
    
    _TECHNICAL_THRESHOLDS = {
        'strong_bullish': 3.0,
        'bullish': 1.5,
        'neutral': 0.5,
        'bearish': -1.5,
        'strong_bearish': -3.0
    }
    
    _GROWTH_SECTORS = frozenset({'Technology', 'Healthcare', 'Renewable Energy', 
                                  'Artificial Intelligence', 'Biotechnology'})
    
    def __init__(self):
        """Initialize with optimized threshold structures"""
        self.financial_thresholds = {
            'excellent': self._FINANCIAL_EXCELLENT,
            'good': self._FINANCIAL_GOOD,
            'average': self._FINANCIAL_AVERAGE,
        }
        
        self.technical_thresholds = self._TECHNICAL_THRESHOLDS
        self.growth_sectors = self._GROWTH_SECTORS

    def build_comprehensive_swot(self, 
                               overall_score: float,
                               avg_net_score: float,
                               prediction: str,
                               key_info: Dict,
                               industry_pe: float,
                               financial_analysis: Dict,
                               volatility: float,
                               current_data: pd.Series,
                               support_level: float,
                               resistance_level: float) -> Dict[str, List[SWOTItem]]:
        """
        Build comprehensive and advanced SWOT analysis.
        
        Optimized with:
        - Batch parameter extraction
        - Reduced method call overhead
        - Efficient data passing
        """
        
        swot = {
            'Strengths': [],
            'Weaknesses': [],
            'Opportunities': [],
            'Threats': []
        }
        
        # Extract common parameters once for all methods
        ratios = financial_analysis.get('ratios', {})
        current_ratio = ratios.get('current_ratio', 2)
        debt_to_equity = ratios.get('debt_to_equity', 0)
        roe = ratios.get('roe', 10)
        rsi = current_data.get('RSI_14')
        
        # Analyze all categories
        swot['Strengths'].extend(self._analyze_strengths(
            overall_score, avg_net_score, financial_analysis, 
            current_data, key_info, current_ratio, debt_to_equity, roe, rsi
        ))
        
        swot['Weaknesses'].extend(self._analyze_weaknesses(
            overall_score, avg_net_score, financial_analysis,
            current_data, volatility, key_info, current_ratio, debt_to_equity, roe, rsi
        ))
        
        swot['Opportunities'].extend(self._analyze_opportunities(
            prediction, key_info, avg_net_score, 
            industry_pe, current_data, support_level, resistance_level, rsi
        ))
        
        swot['Threats'].extend(self._analyze_threats(
            prediction, current_data, volatility, 
            avg_net_score, key_info, support_level, resistance_level, rsi
        ))
        
        # Sort items by priority score
        for category in swot:
            swot[category].sort(key=lambda x: x.priority_score, reverse=True)
        
        return swot

    def _analyze_strengths(self, overall_score: float, avg_net_score: float, 
                          financial_analysis: Dict, current_data: pd.Series, 
                          key_info: Dict, current_ratio: float, debt_to_equity: float,
                          roe: float, rsi: Optional[float]) -> List[SWOTItem]:
        """
        Analyze strengths - optimized with pre-extracted parameters.
        """
        strengths = []
        
        # Financial strength - using threshold comparison
        if overall_score >= 80:
            strengths.append(SWOTItem(
                f"📈 Exceptional Financial Health (Score: {overall_score:.1f}/100)",
                AnalysisCategory.FINANCIAL, 5.0, 5.0, 95.0
            ))
        elif overall_score >= 70:
            strengths.append(SWOTItem(
                f"💪 Strong Financial Performance (Score: {overall_score:.1f}/100)",
                AnalysisCategory.FINANCIAL, 4.0, 4.0, 85.0
            ))
        elif overall_score >= 60:
            strengths.append(SWOTItem(
                f"✅ Solid Financial Foundation (Score: {overall_score:.1f}/100)",
                AnalysisCategory.FINANCIAL, 3.0, 3.0, 75.0
            ))
        
        # Technical strength
        if avg_net_score >= 3:
            strengths.append(SWOTItem(
                f"🚀 Very Strong Technical Momentum (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 4.0, 4.5, 90.0
            ))
        elif avg_net_score >= 1.5:
            strengths.append(SWOTItem(
                f"📊 Strong Technical Indicators (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 3.5, 4.0, 80.0
            ))
        elif avg_net_score >= 0.5:
            strengths.append(SWOTItem(
                f"📈 Positive Technical Trend (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 3.0, 3.5, 70.0
            ))
        
        # RSI indicators - using pre-extracted value
        if pd.notna(rsi):
            if 45 <= rsi <= 65:
                strengths.append(SWOTItem(
                    f"⚖️ Balanced RSI Levels ({rsi:.1f}) - Healthy momentum",
                    AnalysisCategory.TECHNICAL, 2.5, 3.0, 75.0
                ))
            elif rsi <= 30:
                strengths.append(SWOTItem(
                    f"💎 Oversold Opportunity (RSI: {rsi:.1f}) - Potential bounce",
                    AnalysisCategory.TECHNICAL, 3.5, 4.0, 80.0
                ))
        
        # MACD indicators
        macd = current_data.get('MACD')
        if pd.notna(macd) and macd > 0:
            strengths.append(SWOTItem(
                f"📈 Positive MACD Signal ({macd:.3f}) - Upward momentum",
                AnalysisCategory.TECHNICAL, 3.0, 3.5, 70.0
            ))
        
        # Market cap - single comparison
        market_cap = key_info.get('marketCap', 0)
        if market_cap > 10e9:
            strengths.append(SWOTItem(
                "🏢 Large Market Cap - Established company with stability",
                AnalysisCategory.FUNDAMENTAL, 3.0, 3.0, 85.0
            ))
        
        # Financial ratios - using pre-extracted values
        if current_ratio >= 2.0:
            strengths.append(SWOTItem(
                f"💰 Strong Liquidity (Current Ratio: {current_ratio:.2f})",
                AnalysisCategory.FINANCIAL, 4.0, 4.0, 90.0
            ))
        
        if debt_to_equity <= 0.3:
            strengths.append(SWOTItem(
                f"🛡️ Low Debt Burden (D/E: {debt_to_equity:.2f})",
                AnalysisCategory.FINANCIAL, 4.0, 4.5, 85.0
            ))
        
        if roe >= 15:
            strengths.append(SWOTItem(
                f"🎯 Excellent ROE ({roe:.1f}%) - Efficient profit generation",
                AnalysisCategory.FINANCIAL, 4.5, 5.0, 90.0
            ))
        
        return strengths

    def _analyze_weaknesses(self, overall_score: float, avg_net_score: float, 
                           financial_analysis: Dict, current_data: pd.Series,
                           volatility: float, key_info: Dict, current_ratio: float,
                           debt_to_equity: float, roe: float, rsi: Optional[float]) -> List[SWOTItem]:
        """Analyze weaknesses - optimized with pre-extracted parameters"""
        weaknesses = []
        
        # Financial weakness
        if overall_score < 40:
            weaknesses.append(SWOTItem(
                f"⚠️ Poor Financial Health (Score: {overall_score:.1f}/100)",
                AnalysisCategory.FINANCIAL, 5.0, 5.0, 95.0
            ))
        elif overall_score < 50:
            weaknesses.append(SWOTItem(
                f"📉 Below Average Financial Performance (Score: {overall_score:.1f}/100)",
                AnalysisCategory.FINANCIAL, 4.0, 4.0, 85.0
            ))
        elif overall_score < 60:
            weaknesses.append(SWOTItem(
                f"🔻 Moderate Financial Concerns (Score: {overall_score:.1f}/100)",
                AnalysisCategory.FINANCIAL, 3.0, 3.0, 75.0
            ))
        
        # Technical weakness
        if avg_net_score <= -3:
            weaknesses.append(SWOTItem(
                f"📉 Very Weak Technical Signals (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 4.5, 5.0, 90.0
            ))
        elif avg_net_score <= -1.5:
            weaknesses.append(SWOTItem(
                f"🔻 Negative Technical Momentum (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 4.0, 4.0, 80.0
            ))
        elif avg_net_score <= -0.5:
            weaknesses.append(SWOTItem(
                f"⬇️ Bearish Technical Trend (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 3.0, 3.5, 70.0
            ))
        
        # Volatility
        if volatility > 5:
            weaknesses.append(SWOTItem(
                f"🌊 High Volatility Risk ({volatility:.1f}%) - Price instability",
                AnalysisCategory.RISK, 3.5, 4.0, 85.0
            ))
        elif volatility > 3:
            weaknesses.append(SWOTItem(
                f"📊 Moderate Volatility ({volatility:.1f}%) - Some price fluctuation",
                AnalysisCategory.RISK, 2.5, 3.0, 75.0
            ))
        
        # RSI weakness - using pre-extracted value
        if pd.notna(rsi):
            if rsi >= 80:
                weaknesses.append(SWOTItem(
                    f"🔥 Extremely Overbought (RSI: {rsi:.1f}) - Correction likely",
                    AnalysisCategory.TECHNICAL, 4.0, 4.5, 85.0
                ))
            elif rsi >= 70:
                weaknesses.append(SWOTItem(
                    f"⚡ Overbought Condition (RSI: {rsi:.1f}) - Potential pullback",
                    AnalysisCategory.TECHNICAL, 3.0, 3.5, 75.0
                ))
        
        # Financial ratios - using pre-extracted values
        if current_ratio < 1.0:
            weaknesses.append(SWOTItem(
                f"💧 Poor Liquidity (Current Ratio: {current_ratio:.2f})",
                AnalysisCategory.FINANCIAL, 4.5, 4.5, 90.0
            ))
        
        if debt_to_equity > 1.0:
            weaknesses.append(SWOTItem(
                f"⚠️ High Debt Burden (D/E: {debt_to_equity:.2f})",
                AnalysisCategory.FINANCIAL, 4.0, 4.5, 85.0
            ))
        
        if roe < 0:
            weaknesses.append(SWOTItem(
                f"📉 Negative ROE ({roe:.1f}%) - Poor profitability",
                AnalysisCategory.FINANCIAL, 5.0, 5.0, 95.0
            ))
        
        return weaknesses

    def _analyze_opportunities(self, prediction: str, key_info: Dict, 
                              avg_net_score: float, industry_pe: float,
                              current_data: pd.Series, support_level: float,
                              resistance_level: float, rsi: Optional[float]) -> List[SWOTItem]:
        """Analyze opportunities - optimized with pre-extracted parameters"""
        opportunities = []
        
        # Prediction opportunities
        if prediction == "Strong Uptrend":
            opportunities.append(SWOTItem(
                "🚀 Strong Uptrend Predicted - Excellent growth potential",
                AnalysisCategory.TECHNICAL, 5.0, 5.0, 85.0
            ))
        elif prediction == "Possible Uptrend":
            opportunities.append(SWOTItem(
                "📈 Uptrend Potential - Good growth opportunity",
                AnalysisCategory.TECHNICAL, 4.0, 4.0, 75.0
            ))
        
        # RSI opportunities - using pre-extracted value
        if pd.notna(rsi) and rsi <= 30:
            opportunities.append(SWOTItem(
                f"💎 Oversold Bounce Opportunity (RSI: {rsi:.1f})",
                AnalysisCategory.TECHNICAL, 4.0, 4.5, 80.0
            ))
        
        # Market opportunities - optimized PE comparison
        current_pe = key_info.get('trailingPE')
        if current_pe and industry_pe and current_pe < industry_pe * 0.8:
            opportunities.append(SWOTItem(
                f"💰 Undervalued vs Industry (P/E: {current_pe:.1f} vs {industry_pe:.1f})",
                AnalysisCategory.MARKET, 4.0, 4.0, 80.0
            ))
        
        # Growth alignment
        if avg_net_score > 1 and prediction in ["Strong Uptrend", "Possible Uptrend"]:
            opportunities.append(SWOTItem(
                "🎯 Technical & Fundamental Alignment - Strong buy signal",
                AnalysisCategory.TECHNICAL, 4.5, 5.0, 85.0
            ))
        
        # Sector opportunities - using pre-defined set
        sector = key_info.get('sector', '')
        if sector in self.growth_sectors:
            opportunities.append(SWOTItem(
                f"🌟 Growth Sector Exposure ({sector}) - Long-term potential",
                AnalysisCategory.MARKET, 3.5, 4.0, 70.0
            ))
        
        # Support/Resistance opportunities
        current_price = current_data.get('Close', 0)
        if current_price and support_level:
            distance_to_support = (current_price - support_level) / current_price * 100
            if distance_to_support < 5:
                opportunities.append(SWOTItem(
                    "🎯 Near Strong Support Level - Low-risk entry opportunity",
                    AnalysisCategory.TECHNICAL, 3.5, 4.0, 80.0
                ))
        
        return opportunities

    def _analyze_threats(self, prediction: str, current_data: pd.Series,
                        volatility: float, avg_net_score: float,
                        key_info: Dict, support_level: float,
                        resistance_level: float, rsi: Optional[float]) -> List[SWOTItem]:
        """Analyze threats - optimized with pre-extracted parameters"""
        threats = []
        
        # Prediction threats
        if prediction == "Strong Downtrend":
            threats.append(SWOTItem(
                "📉 Strong Downtrend Warning - High risk of losses",
                AnalysisCategory.TECHNICAL, 5.0, 5.0, 85.0
            ))
        elif prediction == "Possible Downtrend":
            threats.append(SWOTItem(
                "⬇️ Downtrend Risk - Potential price decline",
                AnalysisCategory.TECHNICAL, 4.0, 4.0, 75.0
            ))
        
        # Volatility threats
        if volatility > 5:
            threats.append(SWOTItem(
                f"🌊 High Volatility Risk ({volatility:.1f}%) - Unpredictable price moves",
                AnalysisCategory.RISK, 4.0, 4.5, 85.0
            ))
        
        # RSI threats - using pre-extracted value
        if pd.notna(rsi):
            if rsi >= 80:
                threats.append(SWOTItem(
                    f"⚠️ Severe Overbought (RSI: {rsi:.1f}) - Major correction risk",
                    AnalysisCategory.TECHNICAL, 4.5, 5.0, 90.0
                ))
            elif rsi >= 70:
                threats.append(SWOTItem(
                    f"🔴 Overbought Territory (RSI: {rsi:.1f}) - Pullback risk",
                    AnalysisCategory.TECHNICAL, 3.5, 4.0, 80.0
                ))
        
        # Technical threats
        if avg_net_score < -2:
            threats.append(SWOTItem(
                f"📉 Strong Bearish Signals (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 4.0, 4.5, 85.0
            ))
        
        # Market sensitivity
        beta = key_info.get('beta')
        if beta and beta > 1.5:
            threats.append(SWOTItem(
                f"⚡ High Market Sensitivity (Beta: {beta:.2f})",
                AnalysisCategory.RISK, 3.0, 3.5, 75.0
            ))
        
        # Resistance threats
        current_price = current_data.get('Close', 0)
        if current_price and resistance_level:
            distance_to_resistance = (resistance_level - current_price) / current_price * 100
            if distance_to_resistance < 3:
                threats.append(SWOTItem(
                    "🚧 Near Strong Resistance - Limited upside potential",
                    AnalysisCategory.TECHNICAL, 3.0, 3.5, 75.0
                ))
        
        return threats

    def format_swot_simple(self, swot_analysis: Dict[str, List[SWOTItem]]) -> Dict[str, List[str]]:
        """Format SWOT in simple format compatible with main_analysis.py"""
        
        formatted_report = {}
        
        for category, items in swot_analysis.items():
            # Convert items to simple text
            formatted_items = [item.description for item in items]
            
            # Add N/A if list is empty (compatible with original code)
            if not formatted_items:
                formatted_items.append("N/A")
                
            formatted_report[category] = formatted_items
        
        return formatted_report


# ===============================
# Original Functions (Optimized)
# ===============================

def make_investment_decision(overall_score: float, avg_net_score: float, 
                            base_conf: float, volatility: float, 
                            current_data: pd.Series, key_info: Dict, 
                            industry_pe: Optional[float]) -> Tuple[str, List[str], float, float]:
    """
    Investment decision logic considering financial and technical analysis.
    
    Optimizations:
    - Efficient threshold checks with early exit
    - Batch parameter extraction
    - Reduced redundant comparisons
    
    Returns:
        Tuple of (decision, reasons, score, confidence)
    """
    decision_score = 0.0
    decision_reasons = []

    # Financial analysis (40% weight) - optimized with single comparison chain
    if overall_score >= 80:
        decision_score += 4
        decision_reasons.append(f"Excellent financial health ({overall_score:.1f})")
    elif overall_score >= 70:
        decision_score += 3
        decision_reasons.append(f"Strong financial performance ({overall_score:.1f})")
    elif overall_score >= 60:
        decision_score += 2
        decision_reasons.append(f"Good financial stability ({overall_score:.1f})")
    elif overall_score >= 40:
        decision_score += 1
        decision_reasons.append(f"Moderate financial health ({overall_score:.1f})")
    else:
        decision_score -= 2
        decision_reasons.append(f"Weak financial performance ({overall_score:.1f})")

    # Technical analysis (30% weight)
    if avg_net_score >= 3:
        decision_score += 3
        decision_reasons.append(f"Very strong technical signals ({avg_net_score:.2f})")
    elif avg_net_score >= 1.5:
        decision_score += 2
        decision_reasons.append(f"Strong technical momentum ({avg_net_score:.2f})")
    elif avg_net_score >= 0.5:
        decision_score += 1
        decision_reasons.append(f"Positive technical trend ({avg_net_score:.2f})")
    elif avg_net_score <= -3:
        decision_score -= 3
        decision_reasons.append(f"Very weak technical signals ({avg_net_score:.2f})")
    elif avg_net_score <= -1.5:
        decision_score -= 2
        decision_reasons.append(f"Negative technical momentum ({avg_net_score:.2f})")
    elif avg_net_score <= -0.5:
        decision_score -= 1
        decision_reasons.append(f"Bearish technical trend ({avg_net_score:.2f})")

    # Adjust confidence based on financial health
    confidence = max(0, min(100, base_conf - (15 if overall_score < 50 else 0)))

    # Volatility (10% weight)
    if volatility > 5:
        decision_score -= 0.5
        decision_reasons.append(f"High volatility risk ({volatility:.1f}%)")
    elif volatility < 2:
        decision_score += 0.5

    # RSI indicators (5% weight)
    rsi = current_data.get('RSI_14', 50)
    if rsi <= 30:
        decision_score += 0.5
        decision_reasons.append("RSI indicates oversold condition")
    elif rsi >= 70:
        decision_score -= 0.5
        decision_reasons.append("RSI indicates overbought condition")

    # Final decision
    if decision_score >= 4:
        decision = "Strong Buy"
    elif decision_score >= 2:
        decision = "Buy"
    elif decision_score >= -1:
        decision = "Hold"
    elif decision_score >= -3:
        decision = "Sell"
    else:
        decision = "Strong Sell"

    return decision, decision_reasons, decision_score, confidence


def build_comprehensive_swot(decision: str, overall_score: float, avg_net_score: float, 
                            prediction: str, key_info: Dict, industry_pe: float, 
                            financial_analysis: Dict, volatility: float,
                            current_data: pd.Series, support_level: float, 
                            resistance_level: float) -> Dict[str, List[str]]:
    """
    Enhanced function to build comprehensive SWOT analysis - replacement for original function.
    
    Optimizations:
    - Single analyzer instance
    - Efficient parameter passing
    """
    
    analyzer = ProfessionalSWOTAnalyzer()
    
    # Build detailed SWOT analysis
    swot_analysis = analyzer.build_comprehensive_swot(
        overall_score=overall_score,
        avg_net_score=avg_net_score,
        prediction=prediction,
        key_info=key_info,
        industry_pe=industry_pe,
        financial_analysis=financial_analysis,
        volatility=volatility,
        current_data=current_data,
        support_level=support_level,
        resistance_level=resistance_level
    )
    
    # Return simple format required for compatibility with main_analysis.py
    return analyzer.format_swot_simple(swot_analysis)


def analyze_data(technical_data: pd.DataFrame, fundamental_data: Dict, 
                investment_amount: float, industry_pe: float, 
                financial_analysis: Dict) -> Dict:
    """
    Main function for analyzing technical and fundamental data.
    
    Optimizations:
    - Batch data extraction to minimize lookups
    - Pre-computed values to avoid redundant calculations
    - Efficient numpy operations for numerical computations
    - Reduced intermediate allocations
    - Better error handling and validation
    
    Args:
        technical_data: Historical price and technical indicator data
        fundamental_data: Company fundamental data
        investment_amount: Amount available to invest
        industry_pe: Industry average P/E ratio
        financial_analysis: Pre-computed financial analysis
        
    Returns:
        Comprehensive analysis results dictionary
    """
    # Input validation
    if technical_data.empty:
        raise ValueError("Empty technical data provided")
    
    # 1) Calculate technical indicators
    technical_data, fib_levels, sr_zones = calculate_technical_indicators(technical_data)
    technical_data.attrs['sr_zones'] = sr_zones

    # Clean data
    technical_data = technical_data.dropna().reset_index(drop=True)
    if len(technical_data) < 10:
        raise ValueError("Insufficient data for analysis (less than 10 days)")

    # 2) Extract fundamental data - batch extraction for commonly used values
    current_data = technical_data.iloc[-1]
    current_price = float(current_data['Close'])
    recent_data = technical_data.tail(20)
    
    # Calculate support/resistance and volatility in one section
    support_level = float(recent_data['Low'].min())
    resistance_level = float(recent_data['High'].max())
    
    # Optimized volatility calculation using numpy
    returns = technical_data['Close'].pct_change().values
    volatility = float(np.nanstd(returns) * 100)

    # 3) ATR-14 calculation for logical stop loss - optimized with numpy arrays
    high_vals = technical_data['High'].values
    low_vals = technical_data['Low'].values
    close_vals = technical_data['Close'].values
    
    # Efficient True Range calculation using numpy
    close_shift = np.roll(close_vals, 1)
    close_shift[0] = close_vals[0]  # Handle first element
    
    tr = np.maximum.reduce([
        high_vals - low_vals,
        np.abs(high_vals - close_shift),
        np.abs(low_vals - close_shift)
    ])
    
    # Calculate ATR using pandas rolling (efficient for window operations)
    technical_data['ATR_14'] = pd.Series(tr).rolling(window=14, min_periods=1).mean()
    atr14 = float(technical_data['ATR_14'].iloc[-1])

    # 4) Buy/Sell Scores and Net_Score - optimized signal calculation
    from analyze_signals import AdaptiveTechnicalSignalAnalyzer, SignalConfig

    config = SignalConfig(enable_logging=False)
    analyzer = AdaptiveTechnicalSignalAnalyzer(config)

    # Apply scoring to entire dataset (vectorized operations)
    technical_data = analyzer.vectorized_signal_calculation(technical_data)

    # Calculate signal directly from Net_Score (vectorized)
    net_score_vals = technical_data['Net_Score'].values
    technical_data['Signal'] = np.where(
        net_score_vals > 0, 'Buy',
        np.where(net_score_vals < 0, 'Sell', 'Hold')
    )

    # Calculate average net score efficiently
    last_n = technical_data.tail(30).dropna(subset=['Important_Net_Score'])
    avg_net_score = float(last_n['Important_Net_Score'].mean() if not last_n.empty else 0.0)

    # 5) Prediction and base confidence - optimized with combined conditions
    overall_score = financial_analysis.get('overall_score', 0)
    
    if avg_net_score > 2 and overall_score >= 70:
        prediction = "Strong Uptrend"
        base_conf = min(85, 60 + avg_net_score * 5)
    elif avg_net_score > 0 and overall_score >= 50:
        prediction = "Possible Uptrend"
        base_conf = min(75, 55 + avg_net_score * 5)
    elif avg_net_score < -2 and overall_score < 40:
        prediction = "Strong Downtrend"
        base_conf = min(85, 60 + abs(avg_net_score) * 5)
    elif avg_net_score < 0 and overall_score < 50:
        prediction = "Possible Downtrend"
        base_conf = min(75, 55 + abs(avg_net_score) * 5)
    else:
        prediction = "Sideways Movement"
        base_conf = 50

    # 6) Investment decision
    key_info = fundamental_data.get('basic_info', {})
    decision, decision_reasons, decision_score, confidence = make_investment_decision(
        overall_score,
        avg_net_score,
        base_conf,
        volatility,
        current_data,
        key_info,
        industry_pe
    )

    # 7) Price targets calculation
    pivot_pts = {
        'R1': current_data.get('R1'),
        'R2': current_data.get('R2'),
        'S1': current_data.get('S1'),
        'S2': current_data.get('S2'),
    }

    up_targets, down_targets = calculate_price_targets(
        current_price=current_price,
        volatility=volatility,
        bb_upper=current_data['BB_Upper'],
        bb_lower=current_data['BB_Lower'],
        resistance=resistance_level,
        support=support_level,
        short_resistance=current_data.get('R1'),
        long_resistance=current_data.get('Long_Resistance'),
        short_support=current_data.get('S1'),
        long_support=current_data.get('Long_Support'),
        fib_levels=fib_levels,
        pivot_levels=pivot_pts,
        sr_zones=sr_zones,
        trend_prediction=prediction,
    )

    # 8) Calculate shares and amounts
    shares_can_buy = int(investment_amount / current_price) if current_price > 0 else 0
    total_invested = float(shares_can_buy * current_price)
    remaining_cash = float(investment_amount - total_invested)

    # 9) SWOT analysis
    swot = build_comprehensive_swot(
        decision, overall_score, avg_net_score, prediction,
        key_info, industry_pe, financial_analysis, volatility,
        current_data, support_level, resistance_level
    )

    # 10) Entry/Exit/Stop-Loss calculations - heavily optimized with CRITICAL FIX
    # Helper function for filtering valid values
    def filter_valid_values(values: List) -> List[float]:
        """Filter and convert valid numeric values"""
        return [float(x) for x in values 
                if isinstance(x, (int, float, np.number)) and not pd.isna(x) and x > 0]
    
    # CRITICAL FIX: Detect stock position relative to resistance
    price_position = current_price / resistance_level if resistance_level > 0 else 1.0
    is_declining_stock = price_position < 0.70  # More than 30% below 52-week high
    
    # 10-A) Entry point calculation - FIXED LOGIC
    if decision in ["Strong Buy", "Buy"]:
        if is_declining_stock:
            # For declining stocks: Entry should be near current price, not at distant support
            # Example: HIMS at $15.80 should have entry ~$15.64, NOT at support $11.06
            entry_point = float(np.round(current_price * 0.99, 2))
            
            # Safety check: ensure entry is above critical support
            if entry_point < support_level * 1.05:
                entry_point = float(np.round(support_level * 1.05, 2))
        else:
            # For normal/uptrending stocks: Use technical support levels
            entry_candidates = [
                support_level,
                fib_levels.get('Fib_61.8', support_level),
                current_data.get('Long_Support', support_level),
                current_data.get('BB_Lower', support_level),
            ]
            # Add SR zones if available
            valid_sr_zones = [z[0] for z in sr_zones 
                             if isinstance(z, (tuple, list)) and len(z) >= 2 and current_price > z[1]]
            if valid_sr_zones:
                entry_candidates.append(min(valid_sr_zones))
            
            entry_candidates = filter_valid_values(entry_candidates)
            entry_point = float(np.round(max(entry_candidates) if entry_candidates else current_price * 0.98, 2))
        
        # Final validation: Entry should not be more than 5% below current for Buy decisions
        if entry_point < current_price * 0.95:
            entry_point = float(np.round(current_price * 0.97, 2))
        
    elif decision in ["Strong Sell", "Sell"]:
        entry_candidates = [
            resistance_level,
            fib_levels.get('Fib_38.2', resistance_level),
            current_data.get('Long_Resistance', resistance_level),
            current_data.get('BB_Upper', resistance_level),
        ]
        # Add SR zones if available
        valid_sr_zones = [z[1] for z in sr_zones 
                         if isinstance(z, (tuple, list)) and len(z) >= 2 and current_price < z[0]]
        if valid_sr_zones:
            entry_candidates.append(max(valid_sr_zones))
        
        entry_candidates = filter_valid_values(entry_candidates)
        entry_point = float(np.round(min(entry_candidates) if entry_candidates else current_price, 2))
    else:  # Hold
        entry_candidates = [
            support_level,
            fib_levels.get('Fib_61.8', support_level),
            current_data.get('Long_Support', support_level),
        ]
        entry_candidates = filter_valid_values(entry_candidates)
        entry_point = float(np.round(max(entry_candidates) if entry_candidates else current_price, 2))

    # 10-B) Exit point calculation
    if decision in ["Strong Buy", "Buy"]:
        exit_candidates = [
            up_targets[1] if len(up_targets) > 1 else 
            up_targets[0] if up_targets else current_price * 1.05,
            resistance_level,
            fib_levels.get('Fib_23.6', resistance_level),
            current_data.get('Long_Resistance', resistance_level),
            current_data.get('BB_Upper', resistance_level),
        ]
        # Add valid SR zones
        valid_sr_zones = [z[1] for z in sr_zones 
                         if isinstance(z, (tuple, list)) and len(z) >= 2 and z[1] > current_price]
        if valid_sr_zones:
            exit_candidates.append(max(valid_sr_zones))
        
        exit_candidates = [x for x in filter_valid_values(exit_candidates) if x > entry_point]
        exit_point = float(np.round(min(exit_candidates) if exit_candidates else current_price * 1.05, 2))
        
    elif decision in ["Strong Sell", "Sell"]:
        exit_candidates = [
            down_targets[1] if len(down_targets) > 1 else 
            down_targets[0] if down_targets else current_price * 0.95,
            support_level,
            fib_levels.get('Fib_78.6', support_level),
            current_data.get('Long_Support', support_level),
            current_data.get('BB_Lower', support_level),
        ]
        # Add valid SR zones
        valid_sr_zones = [z[0] for z in sr_zones 
                         if isinstance(z, (tuple, list)) and len(z) >= 2 and z[0] < current_price]
        if valid_sr_zones:
            exit_candidates.append(min(valid_sr_zones))
        
        exit_candidates = [x for x in filter_valid_values(exit_candidates) if x < entry_point]
        exit_point = float(np.round(max(exit_candidates) if exit_candidates else current_price * 0.95, 2))
    else:  # Hold
        exit_point = float(np.round(current_price, 2))

    # 10-C) Stop loss calculation - optimized with ATR-based logic
    if decision in ["Strong Buy", "Buy"]:
        # Use 1.5x ATR or 5% of entry, whichever is larger for better protection
        atr_stop = entry_point - (atr14 * 1.5)
        percent_stop = entry_point * 0.95
        stop_loss = float(np.round(max(atr_stop, percent_stop), 2))
    elif decision in ["Strong Sell", "Sell"]:
        # For short positions, stop loss above entry
        atr_stop = entry_point + (atr14 * 1.5)
        percent_stop = entry_point * 1.05
        stop_loss = float(np.round(min(atr_stop, percent_stop), 2))
    else:  # Hold
        stop_loss = float(np.round(entry_point * 0.95, 2))

    # 10-D) Reward-to-Risk calculation
    rr_denom = abs(entry_point - stop_loss)
    reward_to_risk = np.nan
    
    if rr_denom > 0:
        if decision in ["Strong Buy", "Buy"] and exit_point > entry_point:
            reward_to_risk = round((exit_point - entry_point) / rr_denom, 2)
        elif decision in ["Strong Sell", "Sell"] and exit_point < entry_point:
            reward_to_risk = round((entry_point - exit_point) / rr_denom, 2)

    # Enhanced risk rating calculation
    debt_eq = financial_analysis.get('ratios', {}).get('debt_to_equity', np.nan)
    
    # Multi-factor risk assessment
    risk_score = 0
    if volatility > 5:
        risk_score += 2
    elif volatility > 3:
        risk_score += 1
        
    if not np.isnan(debt_eq):
        if debt_eq > 1.5:
            risk_score += 2
        elif debt_eq > 0.8:
            risk_score += 1
    
    if overall_score < 40:
        risk_score += 1
    
    # Determine risk rating
    if risk_score >= 4:
        risk_rating = "Very High"
    elif risk_score >= 3:
        risk_rating = "High"
    elif risk_score >= 2:
        risk_rating = "Medium"
    elif risk_score >= 1:
        risk_rating = "Low-Medium"
    else:
        risk_rating = "Low"

    # 11) Score summarization for UI
    # Convert avg Net-Score (≈ –4 → +4) to 0-100 range
    technical_score = float(np.clip((avg_net_score + 4) / 8 * 100, 0, 100))

    # SWOT score: strengths + opportunities vs weaknesses + threats
    pos_items = sum(1 for i in swot['Strengths'] if i != "N/A") + \
                sum(1 for i in swot['Opportunities'] if i != "N/A")
    neg_items = sum(1 for i in swot['Weaknesses'] if i != "N/A") + \
                sum(1 for i in swot['Threats'] if i != "N/A")
    swot_ratio = pos_items / max(1, pos_items + neg_items)
    swot_score = round(swot_ratio * 100, 1)

    # 12) Price targets table with returns
    price_targets_with_returns = []
    
    for i, t in enumerate(up_targets, 1):
        net = (t - current_price) * shares_can_buy
        price_targets_with_returns.append(
            [f"Upward - Target {i} ({(t/current_price - 1)*100:+.1f}%)", f"${net:.2f}"]
        )
    
    for i, t in enumerate(down_targets, 1):
        net = (t - current_price) * shares_can_buy
        price_targets_with_returns.append(
            [f"Downward - Target {i} ({(t/current_price - 1)*100:+.1f}%)", f"${net:.2f}"]
        )
    
    price_targets_df = pd.DataFrame(price_targets_with_returns, columns=['Type', 'Profit/Loss'])

    # 13) Full financial tables
    fundamental_data_full = (
        fundamental_data.get('financials', pd.DataFrame()),
        fundamental_data.get('balance_sheet', pd.DataFrame()),
        fundamental_data.get('cashflow', pd.DataFrame()),
        fundamental_data.get('quarterly_financials', pd.DataFrame()),
        fundamental_data.get('quarterly_balance_sheet', pd.DataFrame()),
        fundamental_data.get('quarterly_cashflow', pd.DataFrame()),
        fundamental_data.get('earnings', pd.DataFrame()),
        fundamental_data.get('quarterly_earnings', pd.DataFrame())
    )
    
    analyst_avg_target = key_info.get('targetMeanPrice', np.nan)

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION LAYER - Critical fixes before returning results
    # ═══════════════════════════════════════════════════════════════════════
    
    # Validate entry point for Buy decisions
    if decision in ["Strong Buy", "Buy"]:
        # Entry should be at or below current price (max 2% above for volatile stocks)
        if entry_point > current_price * 1.02:
            entry_point = float(np.round(current_price * 0.99, 2))
    
    # Validate targets are progressive
    if len(up_targets) > 1:
        for i in range(1, len(up_targets)):
            if up_targets[i] <= up_targets[i-1]:
                up_targets[i] = up_targets[i-1] * 1.03
        up_targets = [round(t, 2) for t in up_targets]
    
    # Validate stop-loss is below entry for long positions
    if decision in ["Strong Buy", "Buy", "Hold"]:
        if stop_loss >= entry_point:
            stop_loss = float(np.round(entry_point * 0.93, 2))

    # Return comprehensive analysis results
    return {
        'decision': decision,
        'decision_reasons': decision_reasons,
        'decision_score': decision_score,
        'confidence': confidence,
        'current_price': current_price,
        'support_level': support_level,
        'resistance_level': resistance_level,
        'up_targets': up_targets,
        'down_targets': down_targets,
        'entry_point': entry_point,
        'exit_point': exit_point,
        'stop_loss': stop_loss,
        'shares_can_buy': shares_can_buy,
        'total_invested': total_invested,
        'remaining_cash': remaining_cash,
        'swot': swot,
        'price_targets_df': price_targets_df,
        'technical_data': technical_data,
        'fundamental_info': key_info,
        'fib_levels': fib_levels,
        'pivot_levels': {
            'short_support': support_level,
            'short_resistance': resistance_level,
            'long_support': current_data['Long_Support'],
            'long_resistance': current_data['Long_Resistance'],
        },
        'sr_zones': sr_zones,
        'financial_analysis': financial_analysis,
        'fundamental_data_full': fundamental_data_full,
        'prediction': prediction,
        'technical_score': technical_score,
        'swot_score': swot_score,
        'avg_net_score': avg_net_score,
        'risk_rating': risk_rating,
        'reward_to_risk': reward_to_risk,
        'analyst_avg_target': analyst_avg_target
    }
