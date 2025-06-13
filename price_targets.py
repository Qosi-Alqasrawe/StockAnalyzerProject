import numpy as np
from typing import Dict, List, Tuple, Optional

def calculate_price_targets(
    current_price: float,
    volatility: float,
    bb_upper: Optional[float] = None,
    bb_lower: Optional[float] = None,
    resistance: Optional[float] = None,
    support: Optional[float] = None,
    short_resistance: Optional[float] = None,
    long_resistance: Optional[float] = None,
    short_support: Optional[float] = None,
    long_support: Optional[float] = None,
    fib_levels: Dict[str, float] = None,
    pivot_levels: Dict[str, float] = None,
    sr_zones: List[float] = None,
    trend_prediction: str = "Sideways"
) -> Tuple[List[float], List[float]]:
    """
    حساب أهداف الأسعار بناءً على التحليل الفني وتوقع الاتجاه،
    مع دعم الحقول: short_support, short_resistance, long_support, long_resistance, sr_zones
    """
    fib_levels = fib_levels or {}
    pivot_levels = pivot_levels or {}
    sr_zones = sr_zones or []

    # ===== دوال مساعدة =====
    def _pct_levels(vol, base1=0.02, step1=0.03, step2=0.05):
        pct1 = max(base1, vol / 100)
        pct2 = pct1 + step1
        pct3 = pct2 + step2
        return pct1, pct2, pct3

    # ===== بناء المرشحين =====
    # ملاحظة: مستويات short/long تظهر أولاً في القاموس لإعطائها أولوية عند البحث
    up_candidates = {
        'Short_Resistance': short_resistance,
        'Long_Resistance': long_resistance,
        'BB_Upper': bb_upper,
        'Resistance': resistance,
        **{k: fib_levels[k] for k in ['Fib_23.6', 'Fib_38.2'] if k in fib_levels},
        **{k: pivot_levels[k] for k in ['R1', 'R2'] if k in pivot_levels}
    }
    down_candidates = {
        'Short_Support': short_support,
        'Long_Support': long_support,
        'BB_Lower': bb_lower,
        'Support': support,
        **{k: fib_levels[k] for k in ['Fib_61.8', 'Fib_78.6'] if k in fib_levels},
        **{k: pivot_levels[k] for k in ['S1', 'S2'] if k in pivot_levels}
    }

    # أضف SR zones كمستويات تقنية إضافية (إن وُجدت)
    for zone in sr_zones:
        if isinstance(zone, (int, float)):
            if zone >= current_price:
                up_candidates[f"SR_{zone}"] = zone
            if zone <= current_price:
                down_candidates[f"SR_{zone}"] = zone

    # ===== ترتيب الحقول حسب الأولوية (short/long أولاً) =====
    def _prioritized_levels(levels_dict: Dict[str, float], threshold: float, above=True):
        # أولوية: Short > Long > باقي الحقول
        priority_keys = ['Short_Resistance', 'Long_Resistance'] if above else ['Short_Support', 'Long_Support']
        # جرّب أولاً الحقول ذات الأولوية (وتكون رقمية وصحيحة)
        for key in priority_keys:
            val = levels_dict.get(key)
            if isinstance(val, (int, float)):
                if (above and val >= threshold) or (not above and val <= threshold):
                    return val
        # بعدها، جرب باقي الحقول كالمعتاد
        candidates = [
            lvl for k, lvl in levels_dict.items()
            if isinstance(lvl, (int, float)) and ((above and lvl >= threshold) or (not above and lvl <= threshold)) and k not in priority_keys
        ]
        if not candidates:
            return None
        return min(candidates) if above else max(candidates)

    # ===== 1) حساب نسب الأهداف =====
    pct1, pct2, pct3 = _pct_levels(volatility)

    # ===== 3) بناء القوائم الأولية للأهداف =====
    raw_up_targets: List[float] = []
    raw_down_targets: List[float] = []

    if trend_prediction == "Strong Uptrend":
        # ---- أهداف صعودية ----
        raw_up_targets.append(current_price * (1 + pct1))
        thr2 = current_price * (1 + pct1)
        lvl2 = _prioritized_levels(up_candidates, thr2, above=True)
        if lvl2 is None:
            raw_up_targets.append(current_price * (1 + pct2))
        else:
            raw_up_targets.append(max(thr2, lvl2))
        thr3 = current_price * (1 + pct2)
        lvl3 = _prioritized_levels(up_candidates, thr3, above=True)
        if lvl3 is None:
            raw_up_targets.append(current_price * (1 + pct3))
        else:
            raw_up_targets.append(max(thr3, lvl3))
        # ---- أهداف هبوطية (وقف الخسارة) ----
        raw_down_targets.append(current_price * (1 - pct1))
        thr2d = current_price * (1 - pct1)
        lvl2d = _prioritized_levels(down_candidates, thr2d, above=False)
        if lvl2d is None:
            raw_down_targets.append(current_price * (1 - pct2))
        else:
            raw_down_targets.append(min(thr2d, lvl2d))
        thr3d = current_price * (1 - pct2)
        lvl3d = _prioritized_levels(down_candidates, thr3d, above=False)
        if lvl3d is None:
            raw_down_targets.append(current_price * (1 - pct3))
        else:
            raw_down_targets.append(min(thr3d, lvl3d))

    elif trend_prediction == "Strong Downtrend":
        # ---- أهداف هبوطية ----
        raw_down_targets.append(current_price * (1 - pct1))
        thr2d = current_price * (1 - pct1)
        lvl2d = _prioritized_levels(down_candidates, thr2d, above=False)
        if lvl2d is None:
            raw_down_targets.append(current_price * (1 - pct2))
        else:
            raw_down_targets.append(min(thr2d, lvl2d))
        thr3d = current_price * (1 - pct2)
        lvl3d = _prioritized_levels(down_candidates, thr3d, above=False)
        if lvl3d is None:
            raw_down_targets.append(current_price * (1 - pct3))
        else:
            raw_down_targets.append(min(thr3d, lvl3d))
        # ---- أهداف صعودية (وقف خسارة عكسي) ----
        raw_up_targets.append(current_price * (1 + pct1))
        thr2 = current_price * (1 + pct1)
        lvl2 = _prioritized_levels(up_candidates, thr2, above=True)
        if lvl2 is None:
            raw_up_targets.append(current_price * (1 + pct2))
        else:
            raw_up_targets.append(max(thr2, lvl2))
        thr3 = current_price * (1 + pct2)
        lvl3 = _prioritized_levels(up_candidates, thr3, above=True)
        if lvl3 is None:
            raw_up_targets.append(current_price * (1 + pct3))
        else:
            raw_up_targets.append(max(thr3, lvl3))

    else:  # Sideways
        raw_up_targets.append(current_price * (1 + pct1))
        raw_up_targets.append(current_price * (1 + pct2))
        raw_down_targets.append(current_price * (1 - pct1))
        raw_down_targets.append(current_price * (1 - pct2))

    # ===== 4) ترتيب القوائم النهائية (بالتسلسل دون استخدام set) =====
    up_targets = [round(t, 4) for t in raw_up_targets]
    down_targets = [round(t, 4) for t in raw_down_targets]

    return up_targets, down_targets
