import matplotlib.pyplot as plt


def create_price_target_barchart(analysis: dict, fundamental_info: dict, symbol: str):
    """
    يرسم بار تشارت يوضح القيم بالتسلسل التالي دائمًا:
      1) Support
      2) كل Down Targets (Down Target 1, Down Target 2, …)
      3) Current Price
      4) كل Up Targets (Up Target 1, Up Target 2, …)
      5) Resistance
      6) Analyst Target (إن وُجد)

    الألوان:
      • Support: أخضر
      • Down Targets: برتقالي فاتح (salmon)
      • Current Price: أزرق (blue)
      • Up Targets: أزرق فاتح (deepskyblue)
      • Resistance: أحمر (red)
      • Analyst Target: رمادي (gray)
    """

    # 1) جلب القيم من analysis
    support_level    = analysis.get('support_level', None)
    current_price    = analysis.get('current_price', None)
    resistance_level = analysis.get('resistance_level', None)
    up_targets       = analysis.get('up_targets', [])
    down_targets     = analysis.get('down_targets', [])

    # 1.1) جلب Analyst Target من fundamental_info (هي أصلًا float أو None)
    analyst_target = fundamental_info.get('Target Mean Price', None)
    # إذا أتى كنص (مثلاً يحتوي "$")، نحذف "$" والفواصل ثم نجرب نحول إلى float
    if isinstance(analyst_target, str):
        cleaned = analyst_target.replace("$", "").replace(",", "").strip()
        try:
            analyst_target = float(cleaned)
        except:
            analyst_target = None

    # 2) إنشاء قوائم (labels, values, colors) بناءً على التسلسل المطلوب
    labels = []
    values = []
    colors = []

    # -- (1) Support --
    if support_level is not None:
        labels.append("Support")
        values.append(support_level)
        colors.append("green")

    # -- (2) Down Targets (كلها) --
    for idx, tgt in enumerate(down_targets):
        labels.append(f"Down Target {idx+1}")
        values.append(tgt)
        colors.append("salmon")

    # -- (3) Current Price --
    if current_price is not None:
        labels.append("Current Price")
        values.append(current_price)
        colors.append("blue")

    # -- (4) Up Targets (كلها) --
    for idx, tgt in enumerate(up_targets):
        labels.append(f"Up Target {idx+1}")
        values.append(tgt)
        colors.append("deepskyblue")

    # -- (5) Resistance --
    if resistance_level is not None:
        labels.append("Resistance")
        values.append(resistance_level)
        colors.append("red")

    # -- (6) Analyst Target (إن وُجد) --
    if analyst_target is not None:
        labels.append("Analyst Target")
        values.append(analyst_target)
        colors.append("gray")

    # 3) رسم البار تشارت
    fig, ax = plt.subplots(figsize=(6, 3))
    x_positions = list(range(len(labels)))
    ax.bar(x_positions, values, color=colors, alpha=0.8)

    # 4) إضافة قيم الأشرطة كنص فوق كل عمود
    if values:
        max_val = max(values)
        for x, y in zip(x_positions, values):
            ax.text(x, y + (max_val * 0.01), f"{y:.2f}", ha='center', va='bottom', fontsize=8)

    # 5) تنسيق المحاور والعنوان
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Price ($)")
    ax.set_title(f"Price & Targets Bar Chart for {symbol}")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# إذا أردت أن يُمكن ui.py من استدعاء الاسم القديم:
create_price_target_chart = create_price_target_barchart
