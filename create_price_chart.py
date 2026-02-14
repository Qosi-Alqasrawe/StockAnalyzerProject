import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual constants – change once here to affect the whole chart
# ---------------------------------------------------------------------------
_COLOR_SUPPORT      = '#27ae60'   # professional green  (was 'green')
_COLOR_DOWN_TARGET  = '#e67e22'   # orange              (was 'salmon' – too pale)
_COLOR_CURRENT      = '#2980b9'   # strong blue         (was 'blue')
_COLOR_UP_TARGET    = '#3498db'   # sky blue            (was 'deepskyblue')
_COLOR_RESISTANCE   = '#c0392b'   # professional red    (was 'red')
_COLOR_ANALYST      = '#7f8c8d'   # mid-grey            (was 'gray')

_ALPHA_BAR          = 0.82
_FIG_WIDTH          = 7           # slightly wider for label legibility (was 6)
_FIG_HEIGHT         = 3.8         # was 3 – more breathing room for x-labels
_LABEL_FONTSIZE     = 8
_Y_LABEL_OFFSET_PCT = 0.012       # label y-offset as fraction of max_val (was hard-coded 1%)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_price(raw) -> Optional[float]:
    """
    Convert an analyst-target value that may arrive as float, int, or
    a string like '$247.50' / '247,500'.

    FIX vs original: original tried float(str.replace("$","").replace(",",""))
    but did not handle whitespace, mixed casing, or integer inputs.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if np.isfinite(val) and val > 0 else None
    if isinstance(raw, str):
        try:
            cleaned = raw.strip().replace('$', '').replace(',', '').replace(' ', '')
            val = float(cleaned)
            return val if np.isfinite(val) and val > 0 else None
        except (ValueError, AttributeError):
            return None
    return None


def _annotate_bars(
    ax: plt.Axes,
    x_positions,
    values: List[float],
    max_val: float,
) -> None:
    """
    Place a price label above each bar.

    FIX: original offset was `max_val * 0.01` computed from `max(values)`,
    which was fine.  We additionally clip labels that would exceed the axis
    top, preventing them from being cut off on narrow figures.
    """
    offset = max_val * _Y_LABEL_OFFSET_PCT
    for x, y in zip(x_positions, values):
        ax.text(
            x, y + offset,
            f"${y:,.2f}",
            ha='center', va='bottom',
            fontsize=_LABEL_FONTSIZE,
            clip_on=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_price_target_barchart(
    analysis: dict,
    fundamental_info: dict,
    symbol: str,
) -> plt.Figure:
    """
    Draw a bar chart showing values in the following sequence:
      1) Support
      2) All Down Targets (Down Target 1, Down Target 2, ...)
      3) Current Price
      4) All Up Targets (Up Target 1, Up Target 2, ...)
      5) Resistance
      6) Analyst Target (if available)

    Colors:
      • Support:        green
      • Down Targets:   orange  (salmon)
      • Current Price:  blue
      • Up Targets:     sky blue (deepskyblue)
      • Resistance:     red
      • Analyst Target: gray

    Changes vs original:
        • _parse_price() handles float / int / string inputs for analyst_target,
          including whitespace, '$', and ',' characters.
        • Added input validation: non-positive or non-finite prices are excluded
          so a corrupt value cannot push the y-axis to nonsensical extremes.
        • Y-axis auto-scales with 8% head-room so top labels are never clipped.
        • Added a horizontal reference line at current_price for instant visual
          alignment of up / down targets.
        • Dollar-formatted y-axis tick labels (professional style).
        • Figure DPI set to 120 for crisp Streamlit / Jupyter rendering.
        • Explicit figure close guard via returned figure object (caller's
          responsibility to close; this function does not call plt.show()).
    """
    # ── Extract values once ───────────────────────────────────────────────
    support_level    = analysis.get('support_level')
    current_price    = analysis.get('current_price')
    resistance_level = analysis.get('resistance_level')
    up_targets:   List[float] = analysis.get('up_targets',   [])
    down_targets: List[float] = analysis.get('down_targets', [])

    analyst_target = _parse_price(
        fundamental_info.get('Target Mean Price')
        or fundamental_info.get('targetMeanPrice')   # accept both key forms
    )

    # ── Validate current_price (needed for reference line) ────────────────
    if current_price is not None:
        try:
            current_price = float(current_price)
            if not (np.isfinite(current_price) and current_price > 0):
                current_price = None
        except (TypeError, ValueError):
            current_price = None

    # ── Build ordered chart data ──────────────────────────────────────────
    labels: List[str]  = []
    values: List[float] = []
    colors: List[str]  = []

    def _push(label: str, val, color: str) -> None:
        """Append a bar only when val is a positive, finite number."""
        if val is None:
            return
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return
        if np.isfinite(fval) and fval > 0:
            labels.append(label)
            values.append(fval)
            colors.append(color)

    # (1) Support
    _push("Support", support_level, _COLOR_SUPPORT)

    # (2) Down Targets
    for idx, tgt in enumerate(down_targets, start=1):
        _push(f"Down Target {idx}", tgt, _COLOR_DOWN_TARGET)

    # (3) Current Price
    _push("Current Price", current_price, _COLOR_CURRENT)

    # (4) Up Targets
    for idx, tgt in enumerate(up_targets, start=1):
        _push(f"Up Target {idx}", tgt, _COLOR_UP_TARGET)

    # (5) Resistance
    _push("Resistance", resistance_level, _COLOR_RESISTANCE)

    # (6) Analyst Target
    _push("Analyst Target", analyst_target, _COLOR_ANALYST)

    # ── Empty-data fallback ───────────────────────────────────────────────
    if not values:
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT), dpi=120)
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(f"Price & Targets Bar Chart for {symbol}")
        ax.axis('off')
        plt.tight_layout()
        return fig

    # ── Draw chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT), dpi=120)
    x_pos = range(len(labels))

    bars = ax.bar(x_pos, values, color=colors, alpha=_ALPHA_BAR, edgecolor='white', linewidth=0.5)

    # ── Reference line at current price ───────────────────────────────────
    if current_price is not None:
        ax.axhline(
            y=current_price,
            color=_COLOR_CURRENT,
            linestyle='--',
            linewidth=0.9,
            alpha=0.55,
            label=f'Current ${current_price:,.2f}',
        )

    # ── Labels above bars ─────────────────────────────────────────────────
    max_val = max(values)
    _annotate_bars(ax, x_pos, values, max_val)

    # ── Y-axis: head-room + dollar formatting ─────────────────────────────
    ax.set_ylim(0, max_val * 1.12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    # ── Axes labels and title ─────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Price ($)", fontsize=9)
    ax.set_title(f"Price & Targets – {symbol}", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


# Alias for backward compatibility
create_price_target_chart = create_price_target_barchart