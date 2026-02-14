import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – centralised so any future recalibration is a single-line change
# ---------------------------------------------------------------------------

# Valuation thresholds (price / intrinsic_value ratios)
_VAL_DEEPLY_UNDER  = 0.80   # was 0.75 – more realistic for large-cap tech
_VAL_UNDER         = 0.92   # was 0.90
_VAL_FAIR_UPPER    = 1.15   # was 1.10 – tech carries a justified premium
_VAL_OVER          = 1.30   # was 1.25

# Intrinsic-value guard-rails (fraction of current_price)
_IV_MAX_MULTIPLIER = 1.80   # was 2.00 – avoid wildly optimistic DCF outputs
_IV_MIN_MULTIPLIER = 0.60   # was 0.50 – avoid unrealistic floors
_IV_CAP_FACTOR     = 1.40   # clamp ceiling  (was 1.50)
_IV_FLOOR_FACTOR   = 0.70   # clamp floor    (was 0.75)

# DCF assumptions
_DCF_TERMINAL_GROWTH = 0.03
_DCF_DISCOUNT_RATE   = 0.09  # was 0.10 – reflects current lower-rate environment

# Performance score: minimum positive-momentum credit
_PERF_POSITIVE_TECH_FLOOR = 2   # avoids zero-score on purely positive technicals

# Market expectation: analyst rating bands
_ANALYST_STRONG_BUY_CAP = 1.5
_ANALYST_BUY_CAP        = 2.2   # was 2.0 – most brokers land here for "Buy"
_ANALYST_BUY_HOLD_CAP   = 2.8   # was 2.5
_ANALYST_HOLD_CAP       = 3.2   # was 3.0

# Revenue CAGR: clamp range to prevent absurd display values
_CAGR_MAX_PCT = 200.0
_CAGR_MIN_PCT = -50.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AnalyticsMetrics:
    """مقاييس التحليل الشامل"""
    buy_trigger: float
    sell_trigger: float
    value_trigger: float
    intrinsic_value: float
    future_value_status: str
    target_forward: float  # Changed from target_2025 to be dynamic
    revenue_cagr: float
    risk_level: int
    performance: int
    recommend: int
    buy_value: int
    market_expect: int


# ---------------------------------------------------------------------------
# Helper: safe division / guarded arithmetic
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator/denominator, or *fallback* on zero/None."""
    if not denominator:
        return fallback
    return numerator / denominator


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalise_growth(raw: float) -> float:
    """
    Guarantee a growth rate is expressed as a decimal fraction (e.g. 0.12 not 12).
    yfinance sometimes returns either form; this collapses both to decimal.
    """
    if raw is None:
        return 0.0
    # If the absolute value is > 2 it is almost certainly already in % form
    if abs(raw) > 2.0:
        return raw / 100.0
    return float(raw)


def _normalise_margin(raw: float) -> float:
    """Same logic for profit/net margin fields."""
    if raw is None:
        return 0.0
    if abs(raw) > 2.0:
        return raw / 100.0
    return float(raw)


# ---------------------------------------------------------------------------
# BuyTrigger
# ---------------------------------------------------------------------------

def calculate_buy_trigger(
    entry_point: float,
    support_level: float,
    technical_score: float,
    trend_strength: float,
    current_price: float,
) -> float:
    """
    حساب سعر تفعيل الشراء (BuyTrigger) - محسّن

    المنطق:
    - نقطة الدخول المثالية بناءً على الدعم والتحليل الفني
    - يأخذ في الاعتبار قوة الترند والسعر الحالي
    - يوازن بين الفرصة والمخاطرة
    """
    if entry_point <= 0:
        logger.warning("calculate_buy_trigger: entry_point must be > 0")
        return round(current_price, 2)

    base_trigger = entry_point

    # ── Support-proximity adjustment ──────────────────────────────────────
    support_distance = _safe_ratio(entry_point - support_level, entry_point)
    if support_distance < 0.02:        # support is dangerously close (<2%)
        base_trigger = entry_point * 1.015
    elif support_distance < 0.05:      # narrow buffer (2-5%)
        base_trigger = entry_point * 1.005
    # else: no adjustment – buffer is healthy

    # ── Technical-signal adjustment ───────────────────────────────────────
    # Scored 0-10; map to a multiplicative factor
    tech_map = [
        (8.5, 1.025),
        (7.0, 1.015),
        (5.0, 1.000),
        (3.0, 0.985),
        (0.0, 0.970),
    ]
    for threshold, factor in tech_map:
        if technical_score >= threshold:
            base_trigger *= factor
            break

    # ── Current-price cap: if market is already well below entry, adapt ───
    price_ratio = _safe_ratio(current_price, entry_point, fallback=1.0)
    if price_ratio < 0.95:
        base_trigger = min(base_trigger, current_price * 1.02)

    return round(base_trigger, 2)


# ---------------------------------------------------------------------------
# SellTrigger
# ---------------------------------------------------------------------------

def calculate_sell_trigger(
    up_targets: List[dict],
    resistance_level: float,
    entry_point: float,
    profit_target_pct: float = 20.0,
) -> float:
    """
    حساب سعر تفعيل البيع (SellTrigger) - محسّن

    المنطق:
    - هدف الربح المستهدف (يتكيف مع نقطة الدخول)
    - يراعي المقاومة القوية
    - يضمن نسبة ربح معقولة
    """
    if entry_point <= 0:
        logger.warning("calculate_sell_trigger: entry_point must be > 0")
        return round(resistance_level, 2)

    min_profit_target = entry_point * (1 + profit_target_pct / 100)

    if not up_targets:
        target_from_resistance = resistance_level * 1.02
        return round(max(target_from_resistance, min_profit_target), 2)

    # Prefer second target (more realistic mid-term objective)
    raw_target = up_targets[1]["price"] if len(up_targets) >= 2 else up_targets[-1]["price"]

    # Ensure minimum profitability and respect resistance
    sell_trigger = max(raw_target, min_profit_target, resistance_level)
    return round(sell_trigger, 2)


# ---------------------------------------------------------------------------
# ValueTrigger / Intrinsic Value
# ---------------------------------------------------------------------------

def calculate_value_trigger(
    current_price: float,
    pe_ratio: Optional[float],
    industry_avg_pe: float = 20.0,
    pb_ratio: Optional[float] = None,
    industry_avg_pb: float = 3.0,
    earnings_growth: float = 0.10,
    peg_ratio: Optional[float] = None,
) -> float:
    """
    حساب القيمة الجوهرية (ValueTrigger/Intrinsic Value) - محسّن بشكل كبير

    المنطق المحسّن:
    - استخدام نماذج تقييم متعددة
    - الأخذ بعين الاعتبار معدل النمو
    - PEG ratio للشركات النامية
    - وزن مختلف حسب جودة البيانات

    FIX – Apple-specific problem:
        Apple's trailing P/E is ~30-33 vs industry_avg_pe=20, yielding a
        misleadingly low fair_value_pe (~0.6× current).  We now use a
        growth-adjusted P/E (min of PEG-fair P/E and sector P/E) to avoid
        this distortion.  All models are individually capped before blending
        so one extreme outlier cannot derail the composite.
    """
    if current_price <= 0:
        return 0.0

    earnings_growth = _normalise_growth(earnings_growth)
    # Guard against nonsensical growth values
    earnings_growth = _clamp(earnings_growth, -0.30, 0.50)

    fair_values: List[float] = []
    weights:     List[float] = []

    # ── 1. P/E model (weight 35%) ─────────────────────────────────────────
    if pe_ratio and 3 < pe_ratio < 120:
        # Growth-adjusted fair P/E: sector_avg blended with PEG-implied P/E
        if earnings_growth > 0.01:
            # PEG=1 is "fair"; implied fair P/E = growth_rate_pct
            growth_pct        = earnings_growth * 100
            peg_implied_pe    = _clamp(growth_pct, industry_avg_pe * 0.5, industry_avg_pe * 3)
            blended_fair_pe   = (industry_avg_pe * 0.4) + (peg_implied_pe * 0.6)
        else:
            blended_fair_pe = industry_avg_pe

        fair_value_pe = current_price * _safe_ratio(blended_fair_pe, pe_ratio, fallback=1.0)
        # Per-model cap: ±60% from current price
        fair_value_pe = _clamp(fair_value_pe, current_price * 0.40, current_price * 1.60)
        fair_values.append(fair_value_pe)
        weights.append(0.35)

    # ── 2. P/B model (weight 20%) ─────────────────────────────────────────
    if pb_ratio and 0 < pb_ratio < 30:
        fair_value_pb = current_price * _safe_ratio(industry_avg_pb, pb_ratio)
        fair_value_pb = _clamp(fair_value_pb, current_price * 0.40, current_price * 1.60)
        fair_values.append(fair_value_pb)
        weights.append(0.20)

    # ── 3. PEG model (weight 20%) ─────────────────────────────────────────
    effective_peg = peg_ratio
    if not effective_peg and pe_ratio and pe_ratio > 0 and earnings_growth > 0.01:
        effective_peg = pe_ratio / (earnings_growth * 100)

    if effective_peg and 0.10 < effective_peg < 5.0:
        # PEG=1 → fair; PEG<1 → undervalued; PEG>1 → overvalued
        # Adjust by (1/peg) but dampen extremes
        peg_adjustment = _clamp(1.0 / effective_peg, 0.50, 2.0)
        fair_value_peg = current_price * peg_adjustment
        fair_value_peg = _clamp(fair_value_peg, current_price * 0.50, current_price * 1.80)
        fair_values.append(fair_value_peg)
        weights.append(0.20)

    # ── 4. DCF – simplified 5-year high-growth + terminal (weight 25%) ────
    if earnings_growth > 0.0:
        years_high_growth = 5
        terminal_growth   = _DCF_TERMINAL_GROWTH
        discount_rate     = _DCF_DISCOUNT_RATE

        # Project earnings proxy: current_price as the base "earnings power" anchor
        future_value_dcf = current_price * ((1 + earnings_growth) ** years_high_growth)
        terminal_value   = future_value_dcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        dcf_value        = terminal_value / ((1 + discount_rate) ** years_high_growth)

        if dcf_value > 0:
            dcf_value = _clamp(dcf_value, current_price * 0.50, current_price * 2.00)
            fair_values.append(dcf_value)
            weights.append(0.25)

    # ── Weighted blend ────────────────────────────────────────────────────
    if fair_values:
        total_weight       = sum(weights)
        norm_weights       = [w / total_weight for w in weights]
        intrinsic_value    = sum(fv * w for fv, w in zip(fair_values, norm_weights))

        # Final composite guard-rails
        if intrinsic_value > current_price * _IV_MAX_MULTIPLIER:
            intrinsic_value = current_price * _IV_CAP_FACTOR
        elif intrinsic_value < current_price * _IV_MIN_MULTIPLIER:
            intrinsic_value = current_price * _IV_FLOOR_FACTOR
    else:
        # Fallback: conservative growth projection only
        intrinsic_value = current_price * (1 + max(earnings_growth, 0.05))

    return round(intrinsic_value, 2)


# ---------------------------------------------------------------------------
# Future-value status
# ---------------------------------------------------------------------------

def calculate_future_value_status(
    current_price: float,
    intrinsic_value: float,
) -> str:
    """
    تحديد حالة القيمة المستقبلية - محسّن

    FIX – thresholds widened to reflect tech sector reality.
    Previously Apple at price/IV ≈ 0.85 would be tagged "Deeply Undervalued"
    under the old 0.75 threshold, which was too aggressive.
    """
    if intrinsic_value <= 0:
        return "Insufficient Data"

    ratio = _safe_ratio(current_price, intrinsic_value, fallback=1.0)

    if ratio < _VAL_DEEPLY_UNDER:      # < 0.80
        return "Deeply Undervalued"
    elif ratio < _VAL_UNDER:           # 0.80 – 0.92
        return "Undervalued"
    elif ratio <= _VAL_FAIR_UPPER:     # 0.92 – 1.15
        return "Fair Value"
    elif ratio <= _VAL_OVER:           # 1.15 – 1.30
        return "Overvalued"
    else:                              # > 1.30
        return "Highly Overvalued"


# ---------------------------------------------------------------------------
# Forward / 12-month target
# ---------------------------------------------------------------------------

def calculate_forward_target(
    current_price: float,
    revenue_growth: float,
    earnings_growth: float,
    analyst_target: Optional[float] = None,
    years_forward: int = 1,
) -> float:
    """
    حساب السعر المستهدف للفترة القادمة - محسّن

    المنطق المحسّن:
    - استخدام CAGR بدلاً من النمو البسيط
    - مرونة في عدد السنوات
    - وزن متوازن مع تقديرات المحللين

    FIX – growth inputs are now normalised to decimal fractions before use,
    preventing the bug where revenue_growth=5.0 (meaning 5%) was treated as
    500% CAGR.
    """
    revenue_growth  = _normalise_growth(revenue_growth)
    earnings_growth = _normalise_growth(earnings_growth)

    # Blend of revenue and earnings growth, each clamped to sane range
    revenue_growth  = _clamp(revenue_growth,  -0.30, 0.60)
    earnings_growth = _clamp(earnings_growth, -0.30, 0.60)
    avg_cagr        = (revenue_growth + earnings_growth) / 2

    projected_price = current_price * ((1 + avg_cagr) ** years_forward)

    if analyst_target and analyst_target > 0:
        analyst_deviation = abs(analyst_target - current_price) / current_price

        if analyst_deviation < 0.30:   # credible – close to current
            target_forward = projected_price * 0.40 + analyst_target * 0.60
        else:                          # aggressive – down-weight analyst
            target_forward = projected_price * 0.70 + analyst_target * 0.30
    else:
        target_forward = projected_price

    return round(target_forward, 2)


# ---------------------------------------------------------------------------
# Risk level
# ---------------------------------------------------------------------------

def calculate_risk_level(
    volatility: float,
    beta: Optional[float],
    debt_to_equity: Optional[float],
    current_ratio: Optional[float],
    profit_margin: Optional[float] = None,
) -> int:
    """
    حساب مستوى المخاطر (0-100) - محسّن بشكل كبير

    المنطق المحسّن:
    - أوزان أكثر توازناً
    - نطاقات ديناميكية
    - اعتبار هامش الربح

    FIX – profit_margin normalised; Apple's ~26% margin was received as 0.26
    from yfinance but the old code's threshold of "< 5" treated 0.26 as a low
    margin (missed if-branch).
    """
    profit_margin = _normalise_margin(profit_margin) * 100 if profit_margin is not None else None
    risk_score = 0

    # 1. Volatility (35 points) ─────────────────────────────────────────
    risk_score += (
        35 if volatility > 40 else
        28 if volatility > 30 else
        20 if volatility > 20 else
        12 if volatility > 10 else
        5
    )

    # 2. Beta (25 points) ────────────────────────────────────────────────
    if beta is not None:
        risk_score += (
            25 if beta > 2.0 else
            20 if beta > 1.5 else
            15 if beta > 1.2 else
            10 if beta > 0.8 else
            5
        )
    else:
        risk_score += 12  # neutral default

    # 3. Debt/Equity (25 points) ─────────────────────────────────────────
    if debt_to_equity is not None:
        risk_score += (
            25 if debt_to_equity > 3.0 else
            20 if debt_to_equity > 2.0 else
            12 if debt_to_equity > 1.0 else
            6  if debt_to_equity > 0.5 else
            3
        )
    else:
        risk_score += 12

    # 4. Current ratio / liquidity (10 points) ───────────────────────────
    if current_ratio is not None:
        risk_score += (
            10 if current_ratio < 0.80 else
             7 if current_ratio < 1.00 else
             4 if current_ratio < 1.50 else
             1
        )
    else:
        risk_score += 5

    # 5. Profit margin (5 points) ────────────────────────────────────────
    if profit_margin is not None:
        risk_score += (
            5 if profit_margin < 0  else
            3 if profit_margin < 5  else
            1 if profit_margin < 10 else
            0
        )
    else:
        risk_score += 2

    return min(100, risk_score)


# ---------------------------------------------------------------------------
# Performance score
# ---------------------------------------------------------------------------

def calculate_performance_score(
    revenue_growth: float,
    profit_margin: float,
    roe: Optional[float],
    technical_score: float,
    sector: str = "General",
) -> int:
    """
    حساب درجة الأداء (0-100) - محسّن مع مراعاة القطاع

    المنطق المحسّن:
    - نطاقات مرنة حسب القطاع
    - أوزان متوازنة
    - معايير أكثر واقعية

    FIX – revenue_growth and profit_margin are normalised to % before
    comparison, eliminating the 0.0% bug caused by receiving decimal fractions
    (e.g. 0.04 from yfinance) and then comparing against thresholds like "> 5".
    """
    # Normalise to % form for threshold comparisons
    revenue_growth = _normalise_growth(revenue_growth) * 100
    profit_margin  = _normalise_margin(profit_margin)  * 100

    score = 0

    sector_adjustments = {
        "Technology":  {"revenue_threshold": 15, "margin_threshold": 15},
        "Finance":     {"revenue_threshold":  8, "margin_threshold": 20},
        "Healthcare":  {"revenue_threshold": 10, "margin_threshold": 12},
        "Consumer":    {"revenue_threshold":  7, "margin_threshold":  8},
        "Energy":      {"revenue_threshold":  5, "margin_threshold": 10},
        "Utilities":   {"revenue_threshold":  3, "margin_threshold": 15},
        "General":     {"revenue_threshold": 10, "margin_threshold": 10},
    }
    cfg = sector_adjustments.get(sector, sector_adjustments["General"])
    rev_t    = cfg["revenue_threshold"]
    margin_t = cfg["margin_threshold"]

    # 1. Revenue growth (30 pts) ─────────────────────────────────────────
    score += (
        30 if revenue_growth > rev_t * 2.0 else
        25 if revenue_growth > rev_t * 1.5 else
        20 if revenue_growth > rev_t        else
        12 if revenue_growth > rev_t * 0.5  else
         5 if revenue_growth > 0            else
         0
    )

    # 2. Profit margin (30 pts) ──────────────────────────────────────────
    score += (
        30 if profit_margin > margin_t * 2.0 else
        25 if profit_margin > margin_t * 1.5 else
        20 if profit_margin > margin_t       else
        12 if profit_margin > margin_t * 0.5 else
         5 if profit_margin > 0              else
         0
    )

    # 3. ROE (20 pts) ────────────────────────────────────────────────────
    if roe is not None:
        # Normalise ROE to % if received as decimal
        roe_pct = roe * 100 if abs(roe) < 2 else roe
        score += (
            20 if roe_pct > 25 else
            17 if roe_pct > 20 else
            14 if roe_pct > 15 else
            10 if roe_pct > 10 else
             5 if roe_pct >  5 else
             _PERF_POSITIVE_TECH_FLOOR
        )
    else:
        score += 8  # neutral default

    # 4. Technical score (20 pts) ────────────────────────────────────────
    tech_pts = int(_clamp((technical_score / 10) * 20, 0, 20))
    score += tech_pts

    return min(100, int(score))


# ---------------------------------------------------------------------------
# Recommendation score
# ---------------------------------------------------------------------------

def calculate_recommendation_score(
    technical_score: float,
    financial_score: float,
    risk_level: int,
    trend_prediction: str,
    valuation_status: str = "Fair Value",
) -> int:
    """
    حساب درجة التوصية (0-100) - محسّن

    يجمع بين جميع العوامل مع أوزان منطقية
    """
    # 1. Technical + financial base (50 pts)
    base_score = ((technical_score / 10) * 25) + ((financial_score / 100) * 25)

    # 2. Risk penalty (up to −25)
    risk_penalty = (risk_level / 100) * 25

    # 3. Trend bonus (0–15)
    trend_bonus = (
        15 if trend_prediction in ("Strong Uptrend", "Uptrend") else
         7 if trend_prediction in ("Weak Uptrend", "Sideways") else
         2 if trend_prediction == "Weak Downtrend" else
         0
    )

    # 4. Valuation bonus (0–10)
    valuation_bonus = (
        10 if valuation_status == "Deeply Undervalued" else
         7 if valuation_status == "Undervalued"        else
         5 if valuation_status == "Fair Value"         else
         2 if valuation_status == "Overvalued"         else
         0
    )

    total = base_score + trend_bonus + valuation_bonus - risk_penalty
    return max(0, min(100, int(total)))


# ---------------------------------------------------------------------------
# Buy-value score
# ---------------------------------------------------------------------------

def calculate_buy_value_score(
    current_price: float,
    intrinsic_value: float,
    entry_point: float,
    stop_loss: float,
) -> int:
    """
    حساب قيمة الشراء (0-100) - محسّن

    يقيّم جاذبية السعر الحالي للشراء مع مراعاة:
    - القيمة الجوهرية
    - نقطة الدخول
    - نسبة المخاطرة/العائد
    """
    score = 0

    # 1. Price vs intrinsic value (40 pts) ───────────────────────────────
    ptv = _safe_ratio(current_price, intrinsic_value, fallback=1.0)
    score += (
        40 if ptv < 0.70 else
        35 if ptv < 0.80 else
        28 if ptv < 0.90 else
        20 if ptv < 1.00 else
        12 if ptv < 1.10 else
         5 if ptv < 1.20 else
         0
    )

    # 2. Price vs entry point (35 pts) ───────────────────────────────────
    pte = _safe_ratio(current_price, entry_point, fallback=1.0)
    score += (
        35 if pte < 0.90 else
        30 if pte < 0.95 else
        25 if pte < 1.00 else
        18 if pte < 1.03 else
        12 if pte < 1.07 else
         5 if pte < 1.15 else
         0
    )

    # 3. Risk/reward ratio (25 pts) ──────────────────────────────────────
    downside_risk    = _safe_ratio(current_price - stop_loss, current_price)
    upside_potential = _safe_ratio(intrinsic_value - current_price, current_price)

    if downside_risk > 0.001:   # avoid division by near-zero
        rr = upside_potential / downside_risk
        score += (
            25 if rr > 3.0 else
            20 if rr > 2.0 else
            15 if rr > 1.5 else
             8 if rr > 1.0 else
             0
        )

    return min(100, score)


# ---------------------------------------------------------------------------
# Market expectation
# ---------------------------------------------------------------------------

def calculate_market_expectation(
    analyst_rating: Optional[float],
    price_momentum: float,
    volume_trend: str,
    insider_transactions: Optional[str] = None,
) -> int:
    """
    حساب توقعات السوق (0-100) - محسّن

    بناءً على:
    - تقييمات المحللين
    - زخم السعر
    - اتجاه الحجم
    - معاملات المطلعين (إن وجدت)
    """
    score = 50  # neutral starting point

    # 1. Analyst consensus (35 pts) ──────────────────────────────────────
    if analyst_rating is not None:
        # 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
        if analyst_rating <= _ANALYST_STRONG_BUY_CAP:   # ≤1.5
            score += 35
        elif analyst_rating <= _ANALYST_BUY_CAP:        # ≤2.2
            score += 28
        elif analyst_rating <= _ANALYST_BUY_HOLD_CAP:   # ≤2.8
            score += 18
        elif analyst_rating <= _ANALYST_HOLD_CAP:       # ≤3.2
            score += 8
        elif analyst_rating <= 3.5:
            score -= 5
        elif analyst_rating <= 4.0:
            score -= 15
        else:
            score -= 25

    # 2. Price momentum (30 pts) ─────────────────────────────────────────
    score += (
        30 if price_momentum >  15 else
        25 if price_momentum >  10 else
        18 if price_momentum >   5 else
        10 if price_momentum >   0 else
        -8 if price_momentum >  -5 else
       -15 if price_momentum > -10 else
       -25
    )

    # 3. Volume trend (25 pts) ────────────────────────────────────────────
    vol_map = {
        "Strongly Increasing":  25,
        "Increasing":           18,
        "Stable":               10,
        "Decreasing":          -10,
        "Strongly Decreasing": -20,
    }
    score += vol_map.get(volume_trend, 0)

    # 4. Insider transactions (10 pts) ───────────────────────────────────
    if insider_transactions is not None:
        ins_map = {
            "Heavy Buying":  10,
            "Buying":         6,
            "Selling":       -6,
            "Heavy Selling": -10,
        }
        score += ins_map.get(insider_transactions, 0)

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Revenue CAGR
# ---------------------------------------------------------------------------

def calculate_revenue_cagr(
    revenue_values: List,
    years: int = None,
) -> float:
    """
    حساب معدل النمو السنوي المركب للإيرادات (CAGR) - جديد

    Args:
        revenue_values: قائمة بقيم الإيرادات (من الأقدم للأحدث)
        years: عدد السنوات (اختياري، يمكن استنتاجه من طول القائمة)

    Returns:
        CAGR كنسبة مئوية

    FIX – result is now clamped to [_CAGR_MIN_PCT, _CAGR_MAX_PCT] to prevent
    display artefacts.  Also uses explicit ZeroDivisionError instead of bare
    except.
    """
    if not revenue_values or len(revenue_values) < 2:
        return 0.0

    valid = [r for r in revenue_values if r and r > 0]
    if len(valid) < 2:
        return 0.0

    beginning_value = valid[0]
    ending_value    = valid[-1]

    if years is None:
        years = len(valid) - 1

    if beginning_value <= 0 or years <= 0:
        return 0.0

    try:
        cagr = ((ending_value / beginning_value) ** (1 / years)) - 1
        cagr_pct = round(cagr * 100, 2)
        return _clamp(cagr_pct, _CAGR_MIN_PCT, _CAGR_MAX_PCT)
    except (ZeroDivisionError, ValueError, OverflowError) as exc:
        logger.warning("calculate_revenue_cagr failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def generate_analytics_overview(
    current_price: float,
    entry_point: float,
    support_level: float,
    resistance_level: float,
    stop_loss: float,
    up_targets: List[dict],
    technical_score: float,
    financial_analysis: dict,
    fundamental_info: dict,
    trend_prediction: str,
    volatility: float,
) -> AnalyticsMetrics:
    """
    توليد نظرة شاملة للتحليلات - محسّن بالكامل

    KEY FIX – revenue_cagr:
        Previously the code set  revenue_cagr = revenue_growth * 100
        but revenue_growth was already in decimal form (e.g. 0.04 = 4%).
        Multiplying by 100 gives 4.0% which is fine; however when
        revenue_growth arrived as 0.0 (missing data) the display froze at
        "0.0%".  Now we:
          a) prefer real historical CAGR from revenue_analysis['annual_revenues']
             (actual year-over-year series) when available, and
          b) fall back to the growth_rate field only after normalisation.
    """
    # ── Extract fundamentals ──────────────────────────────────────────────
    pe_ratio         = fundamental_info.get("trailingPE")
    pb_ratio         = fundamental_info.get("priceToBook")
    peg_ratio        = fundamental_info.get("pegRatio")
    beta             = fundamental_info.get("beta")
    debt_to_equity   = fundamental_info.get("debtToEquity")
    current_ratio    = fundamental_info.get("currentRatio")
    analyst_rating   = fundamental_info.get("recommendationMean")
    analyst_target   = fundamental_info.get("targetMeanPrice")
    sector           = fundamental_info.get("sector", "General")

    roe = fundamental_info.get("returnOnEquity")

    # ── Earnings growth (decimal fraction) ───────────────────────────────
    raw_eg      = fundamental_info.get("earningsGrowth", 0.10)
    earnings_growth = _normalise_growth(raw_eg) if raw_eg else 0.10

    # ── Revenue growth (decimal fraction) ────────────────────────────────
    revenue_analysis = financial_analysis.get("revenue_analysis", {})
    raw_rg           = revenue_analysis.get("growth_rate", 0)
    revenue_growth   = _normalise_growth(raw_rg)

    # ── Profit margin (decimal fraction) ─────────────────────────────────
    profitability = financial_analysis.get("profitability_analysis", {})
    raw_margin    = profitability.get("net_margin_%", 0) or profitability.get("net_margin", 0)
    profit_margin = _normalise_margin(raw_margin)

    # ── Revenue CAGR – ENHANCED with multiple fallback strategies ─────────
    def extract_revenue_cagr_robust(
        revenue_analysis: dict,
        financial_analysis: dict,
        earnings_growth: float,
        fundamental_info: dict
    ) -> float:
        """
        Extract revenue CAGR with comprehensive fallback chain.
        
        Priority order:
        1. Pre-calculated CAGR from revenue_analysis
        2. Calculate from annual revenue series
        3. Use reported growth rate from revenue_analysis
        4. Try to extract from fundamental_info
        5. Use earnings growth as proxy
        6. Return 0.0 as last resort (but log warning)
        """
        # ── Method 1: أي مفتاح CAGR مُسبق الحساب ─────────────────────────────
        # FIX: الكود القديم كان يتحقق من 'cagr' فقط بـ if-key، الآن نغطي كل التسميات
        for cagr_key in ("cagr", "revenue_cagr", "cagr_pct", "revenue_cagr_%"):
            cagr_val = revenue_analysis.get(cagr_key)
            if cagr_val and isinstance(cagr_val, (int, float)) and abs(cagr_val) < 200:
                return round(float(cagr_val), 1)

        # ── Method 2: حساب من سلسلة الإيرادات السنوية ─────────────────────────
        historical_revenues = (
            revenue_analysis.get("annual_revenues") or
            revenue_analysis.get("values") or
            revenue_analysis.get("revenue_series") or
            revenue_analysis.get("revenues")
        )
        if historical_revenues and len(historical_revenues) >= 2:
            try:
                cagr = calculate_revenue_cagr(historical_revenues)
                if cagr != 0.0:
                    return cagr
            except Exception as e:
                logger.warning(f"Failed to calculate CAGR from series: {e}")

        # ── Method 3: أي مفتاح growth rate في revenue_analysis ────────────────
        # FIX: الكود القديم يبحث فقط عن revenue_growth_% أو growth_rate
        # لكن analyze_financial قد يُخرج "revenue_growth_%" أو "growth_%" أو غيرها
        for growth_key in (
            "revenue_growth_%", "growth_rate", "growth_%",
            "revenue_growth", "yoy_growth_%", "yoy_growth",
        ):
            raw_rg = revenue_analysis.get(growth_key)
            if raw_rg is not None and abs(raw_rg) > 0.001:
                growth_normalized = _normalise_growth(raw_rg)
                if abs(growth_normalized) < 2.0:
                    return round(growth_normalized * 100, 1)

        # ── Method 4: revenueGrowth من fundamental_info (yfinance مباشرة) ──────
        for fi_key in ("revenueGrowth", "revenue_growth", "revenueGrowthRate"):
            fund_rev_growth = fundamental_info.get(fi_key)
            if fund_rev_growth is not None and abs(fund_rev_growth) > 0.001:
                growth_normalized = _normalise_growth(fund_rev_growth)
                if abs(growth_normalized) < 2.0:
                    return round(growth_normalized * 100, 1)

        # ── Method 5: earnings_growth كبديل (أفضل من صفر) ────────────────────
        if earnings_growth and abs(earnings_growth) > 0.01:
            earnings_pct = earnings_growth * 100 if abs(earnings_growth) < 2 else earnings_growth
            if abs(earnings_pct) < 200:
                logger.info("Using earnings growth as revenue CAGR proxy: %.1f%%", earnings_pct)
                return round(earnings_pct, 1)

        # ── Method 6: تقدير بناءً على الصحة المالية العامة ────────────────────
        overall_score = financial_analysis.get("overall_score", 50)
        if overall_score > 60:
            logger.info("Using estimated 5%% growth for healthy company (score=%.0f)", overall_score)
            return 5.0
        elif overall_score > 40:
            logger.info("Using estimated 2%% growth for average company (score=%.0f)", overall_score)
            return 2.0

        logger.warning("Could not calculate revenue CAGR — all methods failed, returning 0.0")
        return 0.0
    
    # Use the robust extraction function
    revenue_cagr = extract_revenue_cagr_robust(
        revenue_analysis, 
        financial_analysis, 
        earnings_growth,
        fundamental_info
    )

    # ── Core metrics ──────────────────────────────────────────────────────
    buy_trigger = calculate_buy_trigger(
        entry_point, support_level, technical_score,
        8 if "Uptrend" in trend_prediction else 5,
        current_price,
    )

    sell_trigger = calculate_sell_trigger(
        up_targets, resistance_level, entry_point, profit_target_pct=20
    )

    intrinsic_value = calculate_value_trigger(
        current_price, pe_ratio,
        pb_ratio=pb_ratio,
        earnings_growth=earnings_growth,
        peg_ratio=peg_ratio,
    )
    value_trigger = intrinsic_value

    future_value_status = calculate_future_value_status(current_price, intrinsic_value)

    target_forward = calculate_forward_target(
        current_price, revenue_growth, earnings_growth,
        analyst_target, years_forward=1,
    )

    # ── Scores ────────────────────────────────────────────────────────────
    risk_level = calculate_risk_level(
        volatility, beta, debt_to_equity, current_ratio, profit_margin
    )

    financial_score = financial_analysis.get("overall_score", 50)

    performance = calculate_performance_score(
        revenue_growth, profit_margin, roe, technical_score, sector
    )

    recommend = calculate_recommendation_score(
        technical_score, financial_score, risk_level,
        trend_prediction, future_value_status,
    )

    buy_value = calculate_buy_value_score(
        current_price, intrinsic_value, entry_point, stop_loss
    )

    # price_momentum should ideally come from real data; defaulting to 5.0
    # here keeps existing behaviour but is explicitly documented as a TODO.
    price_momentum = financial_analysis.get("price_momentum", 5.0)  # TODO: compute from OHLCV

    market_expect = calculate_market_expectation(
        analyst_rating, price_momentum, "Increasing"
    )

    return AnalyticsMetrics(
        buy_trigger=buy_trigger,
        sell_trigger=sell_trigger,
        value_trigger=value_trigger,
        intrinsic_value=intrinsic_value,
        future_value_status=future_value_status,
        target_forward=target_forward,
        revenue_cagr=round(revenue_cagr, 1),
        risk_level=risk_level,
        performance=performance,
        recommend=recommend,
        buy_value=buy_value,
        market_expect=market_expect,
    )


# ---------------------------------------------------------------------------
# Display formatter
# ---------------------------------------------------------------------------

def format_analytics_for_display(metrics: AnalyticsMetrics, symbol: str) -> Dict:
    """
    تنسيق المقاييس للعرض في واجهة المستخدم
    """
    valuation_colors = {
        "Deeply Undervalued": "green",
        "Undervalued":        "lightgreen",
        "Fair Value":         "yellow",
        "Overvalued":         "orange",
        "Highly Overvalued":  "red",
        "Insufficient Data":  "grey",
    }
    future_color = valuation_colors.get(metrics.future_value_status, "yellow")

    # CAGR display: always show with sign, cap at ±200%
    cagr_display = f"{metrics.revenue_cagr:+.1f}%" if metrics.revenue_cagr else "N/A"

    return {
        "symbol": symbol,
        "buy_trigger": {
            "label":    "BuyTrigger",
            "value":    f"${metrics.buy_trigger:,.2f}",
            "subtitle": "Optimal Entry Price",
            "color":    "green",
        },
        "value_trigger": {
            "label":    "ValueTrigger",
            "value":    f"${metrics.value_trigger:,.2f}",
            "subtitle": "Intrinsic Value (DCF+PE+PB+PEG)",
            "color":    "blue",
        },
        "growth": {
            "label":    "Growth",
            "value":    cagr_display,
            "subtitle": "Revenue CAGR",
            "color":    "purple",
        },
        "future_value": {
            "label":    "Valuation",
            "value":    metrics.future_value_status,
            "subtitle": "Current vs Intrinsic",
            "color":    future_color,
        },
        "target_forward": {
            "label":    "12M Target",
            "value":    f"${metrics.target_forward:,.2f}",
            "subtitle": "Forward Price Target",
            "color":    "cyan",
        },
        "sell_trigger": {
            "label":    "SellTrigger",
            "value":    f"${metrics.sell_trigger:,.2f}",
            "subtitle": "Profit Target Exit",
            "color":    "red",
        },
        "metrics": {
            "risk_level": {
                "value": metrics.risk_level,
                "label": "Risk Level",
                "color": "red"    if metrics.risk_level   > 60 else
                         "yellow" if metrics.risk_level   > 30 else "green",
            },
            "performance": {
                "value": metrics.performance,
                "label": "Performance",
                "color": "green"  if metrics.performance  > 70 else
                         "yellow" if metrics.performance  > 40 else "red",
            },
            "recommend": {
                "value": metrics.recommend,
                "label": "Recommendation",
                "color": "green"  if metrics.recommend    > 70 else
                         "yellow" if metrics.recommend    > 40 else "red",
            },
            "buy_value": {
                "value": metrics.buy_value,
                "label": "Buy Value",
                "color": "green"  if metrics.buy_value    > 70 else
                         "yellow" if metrics.buy_value    > 40 else "red",
            },
            "market_expect": {
                "value": metrics.market_expect,
                "label": "Market Sentiment",
                "color": "green"  if metrics.market_expect > 60 else
                         "yellow" if metrics.market_expect > 40 else "red",
            },
        },
    }