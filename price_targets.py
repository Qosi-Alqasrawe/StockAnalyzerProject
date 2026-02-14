import numpy as np
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_UP_PRIORITY_KEYS:   Tuple[str, ...] = ('Short_Resistance', 'Long_Resistance')
_DOWN_PRIORITY_KEYS: Tuple[str, ...] = ('Short_Support',    'Long_Support')
_UP_FIB_KEYS:   Tuple[str, ...] = ('Fib_23.6', 'Fib_38.2')
_DOWN_FIB_KEYS: Tuple[str, ...] = ('Fib_61.8', 'Fib_78.6')
_UP_PIVOT_KEYS:   Tuple[str, ...] = ('R1', 'R2')
_DOWN_PIVOT_KEYS: Tuple[str, ...] = ('S1', 'S2')

_PCT2_DELTA       = 0.03
_PCT3_DELTA       = 0.08
_PCT3_EXTRA       = 0.05
_MIN_VOLATILITY_PCT = 0.02


def _build_candidates(
    current_price: float,
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    resistance: Optional[float],
    support: Optional[float],
    short_resistance: Optional[float],
    long_resistance: Optional[float],
    short_support: Optional[float],
    long_support: Optional[float],
    fib_levels: Dict[str, float],
    pivot_levels: Dict[str, float],
    sr_zones: List[float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    up:   Dict[str, float] = {}
    down: Dict[str, float] = {}

    # Priority levels — must be correctly above/below current_price
    if short_resistance is not None and short_resistance > current_price:
        up['Short_Resistance'] = short_resistance
    if long_resistance is not None and long_resistance > current_price:
        up['Long_Resistance'] = long_resistance
    if short_support is not None and short_support < current_price:
        down['Short_Support'] = short_support
    if long_support is not None and long_support < current_price:
        down['Long_Support'] = long_support

    # Secondary levels
    if bb_upper is not None and bb_upper > current_price:
        up['BB_Upper'] = bb_upper
    if resistance is not None and resistance > current_price:
        up['Resistance'] = resistance

    if bb_lower is not None and bb_lower < current_price:
        down['BB_Lower'] = bb_lower
    if support is not None and support < current_price:
        down['Support'] = support

    # Fibonacci levels
    for key in _UP_FIB_KEYS:
        if key in fib_levels and fib_levels[key] > current_price:
            up[key] = fib_levels[key]
    for key in _DOWN_FIB_KEYS:
        if key in fib_levels and fib_levels[key] < current_price:
            down[key] = fib_levels[key]

    # Pivot levels
    for key in _UP_PIVOT_KEYS:
        if key in pivot_levels and pivot_levels[key] > current_price:
            up[key] = pivot_levels[key]
    for key in _DOWN_PIVOT_KEYS:
        if key in pivot_levels and pivot_levels[key] < current_price:
            down[key] = pivot_levels[key]

    # SR zones
    for idx, zone in enumerate(sr_zones):
        if not isinstance(zone, (int, float)) or not np.isfinite(zone):
            continue
        if zone > current_price:
            up[f"SR_{idx}"]   = zone
        elif zone < current_price:
            down[f"SR_{idx}"] = zone

    return up, down


def _find_best_level(
    levels_dict: Dict[str, float],
    threshold: float,
    above: bool,
    priority_keys: Tuple[str, ...],
) -> Optional[float]:
    """
    Returns closest valid level to threshold.
    - above=True  => smallest value >= threshold
    - above=False => largest value  <= threshold
    Priority keys win only on tie distance.
    """
    best: Optional[float] = None
    best_is_priority: bool = False

    all_keys = list(priority_keys) + [k for k in levels_dict if k not in priority_keys]

    for key in all_keys:
        val = levels_dict.get(key)
        if val is None:
            continue

        valid = (val >= threshold) if above else (val <= threshold)
        if not valid:
            continue

        is_priority = key in priority_keys

        if best is None:
            best = val
            best_is_priority = is_priority
        elif above:
            if val < best or (val == best and is_priority and not best_is_priority):
                best = val
                best_is_priority = is_priority
        else:
            if val > best or (val == best and is_priority and not best_is_priority):
                best = val
                best_is_priority = is_priority

    return best


def _build_up_targets(
    up_candidates: Dict[str, float],
    price_up_1: float,
    price_up_2: float,
) -> List[float]:
    t1 = _find_best_level(up_candidates, price_up_1, True, _UP_PRIORITY_KEYS)
    t1 = max(price_up_1, t1) if t1 is not None else price_up_1

    _eps = t1 * 1e-9
    t2 = _find_best_level(up_candidates, t1 + _eps, True, _UP_PRIORITY_KEYS)
    if t2 is None or t2 <= t1:
        t2 = max(price_up_2, t1 * 1.01)

    _eps2 = t2 * 1e-9
    t3 = _find_best_level(up_candidates, t2 + _eps2, True, _UP_PRIORITY_KEYS)
    if t3 is None or t3 <= t2:
        t3 = t2 * (1 + _PCT3_EXTRA)

    return [t1, t2, t3]


def _build_down_targets(
    down_candidates: Dict[str, float],
    price_down_1: float,
    price_down_2: float,
    price_down_3: float,
) -> List[float]:
    t1 = _find_best_level(down_candidates, price_down_1, False, _DOWN_PRIORITY_KEYS)
    t1 = min(price_down_1, t1) if t1 is not None else price_down_1

    _eps = t1 * 1e-9
    t2 = _find_best_level(down_candidates, t1 - _eps, False, _DOWN_PRIORITY_KEYS)
    if t2 is None or t2 >= t1:
        t2 = min(price_down_2, t1 * 0.99)

    _eps2 = t2 * 1e-9
    t3 = _find_best_level(down_candidates, t2 - _eps2, False, _DOWN_PRIORITY_KEYS)
    if t3 is None or t3 >= t2:
        t3 = min(price_down_3, t2 * 0.97)

    return [t1, t2, t3]


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
    trend_prediction: str = "Sideways",
    entry_point: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """
    Returns: (up_targets, down_targets)

    IMPORTANT FIX:
    - If entry_point is provided => Targets are validated/fallback around ENTRY (not current).
      This guarantees: up_targets > entry_point and down_targets < entry_point (when entry is provided).
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")

    if not np.isfinite(volatility) or volatility < 0:
        volatility = 0.0

    fib_levels   = fib_levels   or {}
    pivot_levels = pivot_levels or {}
    sr_zones     = sr_zones     or []

    # Reference base for final validation & fallback
    base = entry_point if (entry_point is not None and entry_point > 0) else current_price

    # Volatility-derived thresholds (still derived from base for coherence)
    pct1 = max(_MIN_VOLATILITY_PCT, volatility / 100)
    pct2 = pct1 + _PCT2_DELTA
    pct3 = pct1 + _PCT3_DELTA

    price_up_1   = base * (1 + pct1)
    price_up_2   = base * (1 + pct2)
    price_down_1 = base * (1 - pct1)
    price_down_2 = base * (1 - pct2)
    price_down_3 = base * (1 - pct3)

    # Candidate pool built relative to current market price (what is above/below NOW)
    up_candidates, down_candidates = _build_candidates(
        current_price,
        bb_upper, bb_lower, resistance, support,
        short_resistance, long_resistance, short_support, long_support,
        fib_levels, pivot_levels, sr_zones,
    )

    # Construct raw targets
    if trend_prediction == "Strong Uptrend":
        raw_up   = _build_up_targets(up_candidates, price_up_1, price_up_2)
        raw_down = _build_down_targets(down_candidates, price_down_1, price_down_2, price_down_3)

    elif trend_prediction == "Strong Downtrend":
        raw_down = _build_down_targets(down_candidates, price_down_1, price_down_2, price_down_3)
        raw_up   = _build_up_targets(up_candidates, price_up_1, price_up_2)

    elif trend_prediction in ("Uptrend", "Weak Uptrend"):
        raw_up   = _build_up_targets(up_candidates, price_up_1, price_up_2)[:2]
        raw_down = [_build_down_targets(down_candidates, price_down_1, price_down_2, price_down_3)[0]]

    elif trend_prediction in ("Downtrend", "Weak Downtrend"):
        raw_down = _build_down_targets(down_candidates, price_down_1, price_down_2, price_down_3)[:2]
        raw_up   = [_build_up_targets(up_candidates, price_up_1, price_up_2)[0]]

    else:  # Sideways
        raw_up   = [price_up_1,   price_up_2]
        raw_down = [price_down_1, price_down_2]

    # FINAL SAFETY (FIX): validate around BASE (entry if provided, else current)
    up_targets   = [round(t, 4) for t in raw_up   if t > base]
    down_targets = [round(t, 4) for t in raw_down if t < base]

    # If filtered out => fallback around base
    if not up_targets:
        up_targets = [round(base * (1 + pct1), 4), round(base * (1 + pct2), 4)]
    if not down_targets:
        down_targets = [round(base * (1 - pct1), 4), round(base * (1 - pct2), 4)]

    # Stop-loss must be below entry_point (when entry provided)
    if entry_point is not None and entry_point > 0 and down_targets:
        sl_candidate = down_targets[0]
        if sl_candidate >= entry_point:
            corrected_sl = round(entry_point * (1 - pct1), 4)
            # Keep it also under current if possible
            corrected_sl = min(corrected_sl, round(current_price * (1 - pct1), 4))
            down_targets[0] = corrected_sl
            down_targets = sorted(set(down_targets), reverse=True)

    return up_targets, down_targets
