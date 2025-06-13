
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
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
    """عنصر SWOT مع التصنيف والوزن"""
    description: str
    category: AnalysisCategory
    weight: float  # 1-5 (الأهمية)
    impact: float  # 1-5 (التأثير)
    confidence: float  # 0-100% (مستوى الثقة)
    
    @property
    def priority_score(self) -> float:
        """حساب درجة الأولوية"""
        return (self.weight * self.impact * self.confidence) / 100

class ProfessionalSWOTAnalyzer:
    """محلل SWOT احترافي شامل"""
    
    def __init__(self):
        self.financial_thresholds = {
            'excellent': {'score': 80, 'roe': 15, 'debt_ratio': 0.3, 'current_ratio': 2.0},
            'good': {'score': 60, 'roe': 10, 'debt_ratio': 0.5, 'current_ratio': 1.5},
            'average': {'score': 40, 'roe': 5, 'debt_ratio': 0.7, 'current_ratio': 1.0},
        }
        
        self.technical_thresholds = {
            'strong_bullish': 3.0,
            'bullish': 1.5,
            'neutral': 0.5,
            'bearish': -1.5,
            'strong_bearish': -3.0
        }

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
        بناء تحليل SWOT شامل ومتقدم
        """
        
        swot = {
            'Strengths': [],
            'Weaknesses': [],
            'Opportunities': [],
            'Threats': []
        }
        
        # تحليل نقاط القوة
        strengths = self._analyze_strengths(
            overall_score, avg_net_score, financial_analysis, 
            current_data, key_info
        )
        swot['Strengths'].extend(strengths)
        
        # تحليل نقاط الضعف
        weaknesses = self._analyze_weaknesses(
            overall_score, avg_net_score, financial_analysis,
            current_data, volatility, key_info
        )
        swot['Weaknesses'].extend(weaknesses)
        
        # تحليل الفرص
        opportunities = self._analyze_opportunities(
            prediction, key_info, avg_net_score, 
            industry_pe, current_data, support_level, resistance_level
        )
        swot['Opportunities'].extend(opportunities)
        
        # تحليل التهديدات
        threats = self._analyze_threats(
            prediction, current_data, volatility, 
            avg_net_score, key_info, support_level, resistance_level
        )
        swot['Threats'].extend(threats)
        
        # ترتيب العناصر حسب الأولوية
        for category in swot:
            swot[category] = sorted(swot[category], 
                                  key=lambda x: x.priority_score, 
                                  reverse=True)
        
        return swot

    def _analyze_strengths(self, overall_score: float, avg_net_score: float, 
                          financial_analysis: Dict, current_data: pd.Series, 
                          key_info: Dict) -> List[SWOTItem]:
        """تحليل نقاط القوة"""
        strengths = []
        
        # القوة المالية
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
        
        # القوة الفنية
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
        
        # مؤشرات RSI
        if hasattr(current_data, 'RSI_14') and not pd.isna(current_data.get('RSI_14')):
            rsi = current_data['RSI_14']
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
        
        # مؤشرات MACD
        if hasattr(current_data, 'MACD') and not pd.isna(current_data.get('MACD')):
            macd = current_data['MACD']
            if macd > 0:
                strengths.append(SWOTItem(
                    f"📈 Positive MACD Signal ({macd:.3f}) - Upward momentum",
                    AnalysisCategory.TECHNICAL, 3.0, 3.5, 70.0
                ))
        
        # التحليل الأساسي
        market_cap = key_info.get('marketCap', 0)
        if market_cap and market_cap > 10e9:  # أكثر من 10 مليار
            strengths.append(SWOTItem(
                f"🏢 Large Market Cap - Established company with stability",
                AnalysisCategory.FUNDAMENTAL, 3.0, 3.0, 85.0
            ))
        
        # نسب مالية قوية
        ratios = financial_analysis.get('ratios', {})
        if ratios.get('current_ratio', 0) >= 2.0:
            strengths.append(SWOTItem(
                f"💰 Strong Liquidity (Current Ratio: {ratios['current_ratio']:.2f})",
                AnalysisCategory.FINANCIAL, 4.0, 4.0, 90.0
            ))
        
        if ratios.get('debt_to_equity', 1) <= 0.3:
            strengths.append(SWOTItem(
                f"🛡️ Low Debt Burden (D/E: {ratios['debt_to_equity']:.2f})",
                AnalysisCategory.FINANCIAL, 4.0, 4.5, 85.0
            ))
        
        if ratios.get('roe', 0) >= 15:
            strengths.append(SWOTItem(
                f"🎯 Excellent ROE ({ratios['roe']:.1f}%) - Efficient profit generation",
                AnalysisCategory.FINANCIAL, 4.5, 5.0, 90.0
            ))
        
        return strengths

    def _analyze_weaknesses(self, overall_score: float, avg_net_score: float, 
                           financial_analysis: Dict, current_data: pd.Series,
                           volatility: float, key_info: Dict) -> List[SWOTItem]:
        """تحليل نقاط الضعف"""
        weaknesses = []
        
        # الضعف المالي
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
        
        # الضعف الفني
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
        
        # التقلبات العالية
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
        
        # مؤشرات RSI ضعيفة
        if hasattr(current_data, 'RSI_14') and not pd.isna(current_data.get('RSI_14')):
            rsi = current_data['RSI_14']
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
        
        # نسب مالية ضعيفة
        ratios = financial_analysis.get('ratios', {})
        if ratios.get('current_ratio', 2) < 1.0:
            weaknesses.append(SWOTItem(
                f"💧 Poor Liquidity (Current Ratio: {ratios['current_ratio']:.2f})",
                AnalysisCategory.FINANCIAL, 4.5, 4.5, 90.0
            ))
        
        if ratios.get('debt_to_equity', 0) > 1.0:
            weaknesses.append(SWOTItem(
                f"⚠️ High Debt Burden (D/E: {ratios['debt_to_equity']:.2f})",
                AnalysisCategory.FINANCIAL, 4.0, 4.5, 85.0
            ))
        
        if ratios.get('roe', 10) < 0:
            weaknesses.append(SWOTItem(
                f"📉 Negative ROE ({ratios['roe']:.1f}%) - Poor profitability",
                AnalysisCategory.FINANCIAL, 5.0, 5.0, 95.0
            ))
        
        return weaknesses

    def _analyze_opportunities(self, prediction: str, key_info: Dict, 
                              avg_net_score: float, industry_pe: float,
                              current_data: pd.Series, support_level: float,
                              resistance_level: float) -> List[SWOTItem]:
        """تحليل الفرص"""
        opportunities = []
        
        # فرص التنبؤ
        if prediction == "Strong Uptrend":
            opportunities.append(SWOTItem(
                f"🚀 Strong Uptrend Predicted - Excellent growth potential",
                AnalysisCategory.TECHNICAL, 5.0, 5.0, 85.0
            ))
        elif prediction == "Possible Uptrend":
            opportunities.append(SWOTItem(
                f"📈 Uptrend Potential - Good growth opportunity",
                AnalysisCategory.TECHNICAL, 4.0, 4.0, 75.0
            ))
        
        # فرص RSI
        if hasattr(current_data, 'RSI_14') and not pd.isna(current_data.get('RSI_14')):
            rsi = current_data['RSI_14']
            if rsi <= 30:
                opportunities.append(SWOTItem(
                    f"💎 Oversold Bounce Opportunity (RSI: {rsi:.1f})",
                    AnalysisCategory.TECHNICAL, 4.0, 4.5, 80.0
                ))
        
        # فرص السوق
        current_pe = key_info.get('trailingPE')
        if current_pe and industry_pe and current_pe < industry_pe * 0.8:
            opportunities.append(SWOTItem(
                f"💰 Undervalued vs Industry (P/E: {current_pe:.1f} vs {industry_pe:.1f})",
                AnalysisCategory.MARKET, 4.0, 4.0, 80.0
            ))
        
        # فرص النمو
        if avg_net_score > 1 and prediction in ["Strong Uptrend", "Possible Uptrend"]:
            opportunities.append(SWOTItem(
                f"🎯 Technical & Fundamental Alignment - Strong buy signal",
                AnalysisCategory.TECHNICAL, 4.5, 5.0, 85.0
            ))
        
        # فرص القطاع
        sector = key_info.get('sector', '')
        if sector in ['Technology', 'Healthcare', 'Renewable Energy']:
            opportunities.append(SWOTItem(
                f"🌟 Growth Sector Exposure ({sector}) - Long-term potential",
                AnalysisCategory.MARKET, 3.5, 4.0, 70.0
            ))
        
        # فرص الدعم والمقاومة
        current_price = current_data.get('Close', 0)
        if current_price and support_level:
            distance_to_support = (current_price - support_level) / current_price * 100
            if distance_to_support < 5:  # أقل من 5% من الدعم
                opportunities.append(SWOTItem(
                    f"🎯 Near Strong Support Level - Low-risk entry opportunity",
                    AnalysisCategory.TECHNICAL, 3.5, 4.0, 80.0
                ))
        
        return opportunities

    def _analyze_threats(self, prediction: str, current_data: pd.Series,
                        volatility: float, avg_net_score: float,
                        key_info: Dict, support_level: float,
                        resistance_level: float) -> List[SWOTItem]:
        """تحليل التهديدات"""
        threats = []
        
        # تهديدات التنبؤ
        if prediction == "Strong Downtrend":
            threats.append(SWOTItem(
                f"📉 Strong Downtrend Warning - High risk of losses",
                AnalysisCategory.TECHNICAL, 5.0, 5.0, 85.0
            ))
        elif prediction == "Possible Downtrend":
            threats.append(SWOTItem(
                f"⬇️ Downtrend Risk - Potential price decline",
                AnalysisCategory.TECHNICAL, 4.0, 4.0, 75.0
            ))
        
        # تهديدات التقلبات
        if volatility > 5:
            threats.append(SWOTItem(
                f"🌊 High Volatility Risk ({volatility:.1f}%) - Unpredictable price moves",
                AnalysisCategory.RISK, 4.0, 4.5, 85.0
            ))
        
        # تهديدات RSI
        if hasattr(current_data, 'RSI_14') and not pd.isna(current_data.get('RSI_14')):
            rsi = current_data['RSI_14']
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
        
        # تهديدات فنية
        if avg_net_score < -2:
            threats.append(SWOTItem(
                f"📉 Strong Bearish Signals (Net Score: {avg_net_score:.2f})",
                AnalysisCategory.TECHNICAL, 4.0, 4.5, 85.0
            ))
        
        # تهديدات السوق العامة
        beta = key_info.get('beta')
        if beta and beta > 1.5:
            threats.append(SWOTItem(
                f"⚡ High Market Sensitivity (Beta: {beta:.2f})",
                AnalysisCategory.RISK, 3.0, 3.5, 75.0
            ))
        
        # تهديدات المقاومة
        current_price = current_data.get('Close', 0)
        if current_price and resistance_level:
            distance_to_resistance = (resistance_level - current_price) / current_price * 100
            if distance_to_resistance < 3:  # أقل من 3% من المقاومة
                threats.append(SWOTItem(
                    f"🚧 Near Strong Resistance - Limited upside potential",
                    AnalysisCategory.TECHNICAL, 3.0, 3.5, 75.0
                ))
        
        return threats

    def format_swot_simple(self, swot_analysis: Dict[str, List[SWOTItem]]) -> Dict[str, List[str]]:
        """تنسيق SWOT بصيغة بسيطة متوافقة مع main_analysis.py"""
        
        formatted_report = {}
        
        for category, items in swot_analysis.items():
            # تحويل العناصر إلى نصوص بسيطة
            formatted_items = []
            for item in items:
                formatted_items.append(item.description)
            
            # إضافة N/A إذا كانت القائمة فارغة (متوافق مع الكود الأصلي)
            if not formatted_items:
                formatted_items.append("N/A")
                
            formatted_report[category] = formatted_items
        
        return formatted_report


# ===============================
# Original Functions (Updated)
# ===============================

def make_investment_decision(overall_score, avg_net_score, base_conf, volatility, current_data, key_info, industry_pe):
    """
    منطق قرار استثماري يأخذ بعين الاعتبار التحليل المالي والفني.
    Returns: (decision, reasons, score, confidence)
    """
    decision_score = 0
    decision_reasons = []

    # التحليل المالي (وزن 40%)
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

    # التحليل الفني (وزن 30%)
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

    # تعديل الثقة بناءً على الصحة المالية
    confidence = max(0, min(100, base_conf - (15 if overall_score < 50 else 0)))

    # التقلبات (وزن 10%)
    if volatility > 5:
        decision_score -= 0.5
        decision_reasons.append(f"High volatility risk ({volatility:.1f}%)")
    elif volatility < 2:
        decision_score += 0.5

    # مؤشرات إضافية (RSI) (وزن 5%)
    rsi = current_data.get('RSI_14', 50)
    if rsi <= 30:
        decision_score += 0.5
        decision_reasons.append("RSI indicates oversold condition")
    elif rsi >= 70:
        decision_score -= 0.5
        decision_reasons.append("RSI indicates overbought condition")

    # القرار النهائي
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


def build_comprehensive_swot(decision, overall_score, avg_net_score, prediction,
                             key_info, industry_pe, financial_analysis, volatility,
                             current_data, support_level, resistance_level):
    """
    دالة محسنة لبناء تحليل SWOT شامل - استبدال للدالة الأصلية
    """
    
    analyzer = ProfessionalSWOTAnalyzer()
    
    # بناء تحليل SWOT المفصل
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
    
    # إرجاع التنسيق البسيط المطلوب للتوافق مع main_analysis.py
    return analyzer.format_swot_simple(swot_analysis)


def analyze_data(technical_data, fundamental_data, investment_amount, industry_pe, financial_analysis):
    """
    الدالة الرئيسية لتحليل البيانات الفنية والأساسية.
    """
    # 1) حساب المؤشرات الفنية
    technical_data, fib_levels, sr_zones = calculate_technical_indicators(technical_data)
    # تخزين مناطق الدعم/المقاومة في attrs للـ DataFrame
    technical_data.attrs['sr_zones'] = sr_zones

    # ملاحظة: لتعمل الدالة vectorized_signal_calculation بشكل صحيح،
    # يجب قبل استدعائها أن تُخزن sr_zones بهذه الطريقة في attrs
    # بحيث يقرأها المحلل ويُنشئ عمود SR_Zone بناءً عليها.

    technical_data = technical_data.dropna().reset_index(drop=True)
    if len(technical_data) < 10:
        raise ValueError("Insufficient data for analysis (less than 10 days)")


    # 2) الحصول على البيانات الأساسية
    current_data     = technical_data.iloc[-1]
    current_price    = current_data['Close']
    recent_data      = technical_data.tail(20)
    support_level    = recent_data['Low'].min()
    resistance_level = recent_data['High'].max()
    volatility       = technical_data['Close'].pct_change().std() * 100

    # === ATR-14 لحساب وقف الخسارة المنطقي ===
    high  = technical_data['High']
    low   = technical_data['Low']
    close = technical_data['Close']

    true_ranges = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1)

    technical_data['ATR_14'] = (
        true_ranges.max(axis=1)
        .rolling(window=14, min_periods=1)
        .mean()
    )
    atr14 = technical_data['ATR_14'].iloc[-1]


    # 3) Buy/Sell Scores و Net_Score
    # توزيع الدرجات مرة واحدة لكل الداتا
    from analyze_signals import AdaptiveTechnicalSignalAnalyzer, SignalConfig

    config = SignalConfig(enable_logging=False)
    analyzer = AdaptiveTechnicalSignalAnalyzer(config)

    # طبق توزيع النقاط على الداتا كاملة
    technical_data = analyzer.vectorized_signal_calculation(technical_data)

    # احسب الإشارة (Buy/Sell/Hold) مباشرة من Net_Score
    technical_data['Signal'] = technical_data['Net_Score'].apply(
        lambda x: 'Buy' if x > 0 else ('Sell' if x < 0 else 'Hold')
    )

    # بعد توزيع الدرجات وإضافة Important columns:
    # بعد توزيع الدرجات وإضافة Important columns:
    last_n = technical_data.tail(30).dropna(subset=['Important_Net_Score'])
    avg_net_score = (
        last_n['Important_Net_Score'].mean()
        if not last_n.empty else np.nan
    )




    # 4) التنبؤ (prediction) و base_conf
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

    # 5) قرار الاستثمار
    decision, decision_reasons, decision_score, confidence = make_investment_decision(
        overall_score,
        avg_net_score,
        base_conf,
        volatility,
        current_data,
        fundamental_data.get('basic_info', {}),
        industry_pe
    )

    # 6) حساب أهداف السعر
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

    # 7) حساب الأسهم والمبالغ
    shares_can_buy  = int(investment_amount / current_price) if current_price > 0 else 0
    total_invested  = shares_can_buy * current_price
    remaining_cash  = investment_amount - total_invested

    # 8) تحليل SWOT
    key_info = fundamental_data.get('basic_info', {})
    swot     = build_comprehensive_swot(
        decision, overall_score, avg_net_score, prediction,
        key_info, industry_pe, financial_analysis, volatility,
        current_data, support_level, resistance_level
    )

# 9) نقاط الدخول/الخروج/وقف الخسارة (محسَّنة جداً)
    # -------- 9-A) نقطة الدخول --------
    entry_candidates = [
        support_level,
        fib_levels.get('Fib_61.8', support_level),
        min(
            [z[0] for z in sr_zones
             if isinstance(z, (tuple, list)) and len(z) >= 2 and current_price > z[1]],
            default=support_level
        ),
        current_data.get('Long_Support', support_level),
        current_data.get('BB_Lower', support_level),
    ]
    entry_candidates = [
        x for x in entry_candidates
        if isinstance(x, (int, float)) and not pd.isna(x) and x > 0
    ]

    if decision in ["Strong Sell", "Sell"]:
        entry_candidates = [
            resistance_level,
            fib_levels.get('Fib_38.2', resistance_level),
            max(
                [z[1] for z in sr_zones
                 if isinstance(z, (tuple, list)) and len(z) >= 2 and current_price < z[0]],
                default=resistance_level
            ),
            current_data.get('Long_Resistance', resistance_level),
            current_data.get('BB_Upper', resistance_level),
        ]
        entry_candidates = [
            x for x in entry_candidates
            if isinstance(x, (int, float)) and not pd.isna(x) and x > 0
        ]

    if decision in ["Strong Buy", "Buy"]:
        entry_point = float(np.round(max(entry_candidates) if entry_candidates else current_price, 2))
    elif decision in ["Strong Sell", "Sell"]:
        entry_point = float(np.round(min(entry_candidates) if entry_candidates else current_price, 2))
    else:  # Hold
        # نعتمد أقرب مستوى دعم/مقاومة بدلاً من السعر الحالي
        entry_point = float(np.round(max(entry_candidates) if entry_candidates else current_price, 2))

    # -------- 9-B) نقطة الخروج --------
    if decision in ["Strong Buy", "Buy"]:
        exit_candidates = [
            up_targets[1] if len(up_targets) > 1 else
            up_targets[0] if up_targets else current_price * 1.05,
            resistance_level,
            fib_levels.get('Fib_23.6', resistance_level),
            current_data.get('Long_Resistance', resistance_level),
            current_data.get('BB_Upper', resistance_level),
            max(
                [z[1] for z in sr_zones
                 if isinstance(z, (tuple, list)) and len(z) >= 2 and z[1] > current_price],
                default=resistance_level
            )
        ]
        exit_candidates = [
            x for x in exit_candidates
            if isinstance(x, (int, float)) and not pd.isna(x) and x > entry_point
        ]
        exit_point = float(np.round(min(exit_candidates) if exit_candidates else current_price * 1.05, 2))

    elif decision in ["Strong Sell", "Sell"]:
        exit_candidates = [
            down_targets[1] if len(down_targets) > 1 else
            down_targets[0] if down_targets else current_price * 0.95,
            support_level,
            fib_levels.get('Fib_78.6', support_level),
            current_data.get('Long_Support', support_level),
            current_data.get('BB_Lower', support_level),
            min(
                [z[0] for z in sr_zones
                 if isinstance(z, (tuple, list)) and len(z) >= 2 and z[0] < current_price],
                default=support_level
            )
        ]
        exit_candidates = [
            x for x in exit_candidates
            if isinstance(x, (int, float)) and not pd.isna(x) and x < entry_point
        ]
        exit_point = float(np.round(max(exit_candidates) if exit_candidates else current_price * 0.95, 2))
    else:  # Hold
        exit_point = float(np.round(current_price, 2))

    # -------- 9-C) وقف الخسارة (يُحسَب فقط من نقطة الدخول) --------
    if decision in ["Strong Buy", "Buy"]:
        # وقف الخسارة = ‎5 ٪‎ تحت *نقطة الدخول* (دائمًا < Entry)
        stop_loss = float(np.round(entry_point * 0.95, 2))

    elif decision in ["Strong Sell", "Sell"]:
        # وقف الخسارة = ‎5 ٪‎ فوق *نقطة الدخول* (دائمًا > Entry)
        stop_loss = float(np.round(entry_point * 1.05, 2))

    else:  # Hold أو أي قرار آخر
        # اجعل SL على بُعد 5 ٪ من *Entry* أيضًا (أدنى للقيمة الوقائية)
        stop_loss = float(np.round(entry_point * 0.95, 2))


    # -------- 9-D) Reward-to-Risk & التصنيف --------
    rr_denom = abs(entry_point - stop_loss)
    reward_to_risk = np.nan
    if rr_denom > 0:
        if decision in ["Strong Buy", "Buy"] and exit_point > entry_point:
            reward_to_risk = round((exit_point - entry_point) / rr_denom, 2)
        elif decision in ["Strong Sell", "Sell"] and exit_point < entry_point:
            reward_to_risk = round((entry_point - exit_point) / rr_denom, 2)

    debt_eq = financial_analysis.get('ratios', {}).get('debt_to_equity', np.nan)
    if volatility > 5 or (not np.isnan(debt_eq) and debt_eq > 1):
        risk_rating = "High"
    elif volatility > 3 or (not np.isnan(debt_eq) and debt_eq > 0.5):
        risk_rating = "Medium"
    else:
        risk_rating = "Low"

    # === 9.5) تلخيص الدرجات للواجهة ===
    #  تحويل متوسط الـ Net-Score (≈ –4 → +4) إلى نطاق 0-100
    technical_score = np.clip((avg_net_score + 4) / 8 * 100, 0, 100)

    #  درجة SWOT مبسّطة: نقاط القوة + الفرص مقابل الضعف + التهديدات
    pos_items = len([i for i in swot['Strengths'] if i != "N/A"]) + \
                len([i for i in swot['Opportunities'] if i != "N/A"])
    neg_items = len([i for i in swot['Weaknesses'] if i != "N/A"]) + \
                len([i for i in swot['Threats'] if i != "N/A"])
    swot_ratio  = (pos_items / max(1, pos_items + neg_items))
    swot_score  = round(swot_ratio * 100, 1)


    # 10) جدول العوائد
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

    # 11) الجداول المالية الكاملة
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
	"technical_score": technical_score,   # 0-100
        "swot_score":      swot_score,
        'avg_net_score': avg_net_score,
        'risk_rating':      risk_rating,
        'reward_to_risk':   reward_to_risk,
        'analyst_avg_target': analyst_avg_target
    }

