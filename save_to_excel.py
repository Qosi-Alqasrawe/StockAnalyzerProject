# save_to_excel.py - Enhanced Professional Version
"""
Optimized Excel Report Generation Module

Performance improvements:
- Pre-defined formatting objects (reused across sheets)
- Batch data extraction to minimize dictionary lookups
- Unified rating system to reduce code duplication
- Type hints for better maintainability
- Comprehensive error handling
- Efficient data validation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as XLImage
from typing import Dict, List, Tuple, Any, Optional
import os


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

# Pre-defined formatting objects (created once, reused everywhere)
class ExcelFormats:
    """Centralized formatting definitions for Excel exports"""
    
    # Fonts
    HEADER_FONT = Font(name='Calibri', size=12, bold=True, color='FFFFFF')
    SECTION_FONT = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
    BOLD_FONT = Font(name='Calibri', size=11, bold=True)
    NORMAL_FONT = Font(name='Calibri', size=11)
    
    # Fills
    HEADER_FILL = PatternFill(start_color='4F81BD', end_color='4F81BD', fill_type='solid')
    BUY_FILL = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
    SELL_FILL = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
    HOLD_FILL = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
    
    STRONG_BUY_FILL = PatternFill(start_color='00C851', end_color='00C851', fill_type='solid')
    STRONG_SELL_FILL = PatternFill(start_color='FF4444', end_color='FF4444', fill_type='solid')
    NEUTRAL_FILL = PatternFill(start_color='FFBB33', end_color='FFBB33', fill_type='solid')
    
    # Borders
    THIN_BORDER = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Alignments
    CENTER_ALIGN = Alignment(horizontal='center', vertical='center')
    LEFT_ALIGN = Alignment(horizontal='left', vertical='center')
    RIGHT_ALIGN = Alignment(horizontal='right', vertical='center')


# Rating thresholds (centralized configuration)
class RatingThresholds:
    """Centralized rating threshold definitions"""
    
    MARGIN = [(20, '⭐⭐⭐ Excellent'), (10, '⭐⭐ Good'), (5, '⭐ Fair'), (0, '❌ Poor')]
    ROE = [(20, '⭐⭐⭐ Excellent'), (15, '⭐⭐ Good'), (10, '⭐ Fair'), (0, '❌ Poor')]
    DEBT = [(0.5, '⭐⭐⭐ Low Risk'), (1.0, '⭐⭐ Moderate'), (2.0, '⭐ High'), (float('inf'), '❌ Very High')]
    LIQUIDITY = [(2.0, '⭐⭐⭐ Excellent'), (1.5, '⭐⭐ Good'), (1.0, '⭐ Fair'), (0, '❌ Poor')]
    PE = [(10, 20, '⭐⭐⭐ Attractive'), (20, 30, '⭐⭐ Fair'), (0, 10, '⭐ Undervalued?'), (30, float('inf'), '❌ Overvalued?')]
    PB = [(1.5, '⭐⭐⭐ Attractive'), (3.0, '⭐⭐ Fair'), (float('inf'), '❌ Expensive')]
    SCORE = [(70, '🟢 Strong'), (50, '🟡 Moderate'), (0, '🔴 Weak')]
    ADX = [(25, '🟢 Strong Trend'), (20, '🟡 Trend Forming'), (0, '⚪ Weak Trend')]


# ============================================================================
# MAIN EXPORT FUNCTION
# ============================================================================

def save_report(analysis: Dict[str, Any], symbol: str, download_path: str) -> str:
    """
    Save comprehensive analysis to Excel with professional formatting.
    
    Performance optimizations:
    - Batch data extraction at start
    - Pre-defined formatting objects
    - Single-pass sheet creation
    - Efficient error handling
    
    Args:
        analysis: Analysis results dictionary
        symbol: Stock ticker symbol
        download_path: Directory to save report
        
    Returns:
        Path to created Excel file
    """
    # Ensure download path exists
    os.makedirs(download_path, exist_ok=True)
    excel_filename = os.path.join(download_path, f"{symbol}_Investment_Analysis.xlsx")

    # Extract common data once (batch extraction for performance)
    common_data = extract_common_data(analysis)

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # Create all sheets
        create_executive_dashboard(writer, analysis, symbol, common_data)
        create_investment_decision_sheet(writer, analysis, common_data)
        create_price_targets_sheet(writer, analysis, common_data)
        create_technical_summary_sheet(writer, analysis, common_data)
        create_financial_summary_sheet(writer, analysis, common_data)
        create_swot_sheet(writer, analysis)
        create_analytics_sheet(writer, analysis, common_data)
        create_detailed_technical_sheet(writer, analysis)
        create_financial_statements_sheets(writer, analysis)
        
        # Apply professional formatting to all sheets
        apply_professional_formatting(writer)

    return excel_filename


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_common_data(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract commonly used data in one batch operation.
    Reduces repeated dictionary lookups throughout the code.
    
    Args:
        analysis: Analysis results dictionary
        
    Returns:
        Dictionary with extracted common data
    """
    # Safe extraction with defaults
    financial_analysis = analysis.get('financial_analysis', {})
    fundamental_info = analysis.get('fundamental_info', {})
    up_targets = analysis.get('up_targets', [0, 0])
    down_targets = analysis.get('down_targets', [])
    swot = analysis.get('swot', {})
    
    return {
        # Prices
        'current_price': float(analysis.get('current_price', 0)),
        'entry_point': float(analysis.get('entry_point', 0)),
        'stop_loss': float(analysis.get('stop_loss', 0)),
        'support_level': float(analysis.get('support_level', 0)),
        'resistance_level': float(analysis.get('resistance_level', 0)),
        
        # Targets
        'target_1': float(up_targets[0]) if len(up_targets) > 0 else 0.0,
        'target_2': float(up_targets[1]) if len(up_targets) > 1 else 0.0,
        'up_targets': up_targets,
        'down_targets': down_targets,
        
        # Decision
        'decision': str(analysis.get('decision', 'HOLD')),
        'confidence': float(analysis.get('confidence', 0)),
        'risk_rating': str(analysis.get('risk_rating', 'Medium')),
        'reward_to_risk': float(analysis.get('reward_to_risk', 0)),
        'prediction': str(analysis.get('prediction', 'N/A')),
        
        # Scores
        'technical_score': float(analysis.get('technical_score', 0)),
        'financial_score': float(financial_analysis.get('overall_score', 0)),
        'swot_score': float(analysis.get('swot_score', 0)),
        
        # Investment sizing
        'shares_can_buy': int(analysis.get('shares_can_buy', 0)),
        'total_invested': float(analysis.get('total_invested', 0)),
        'remaining_cash': float(analysis.get('remaining_cash', 0)),
        
        # Financial metrics
        'net_margin': float(financial_analysis.get('profitability_analysis', {}).get('net_margin_%', 0)),
        'gross_margin': float(financial_analysis.get('profitability_analysis', {}).get('gross_margin_%', 0)),
        'roe': float(financial_analysis.get('profitability_analysis', {}).get('ROE_%', 0)),
        'debt_to_equity': float(financial_analysis.get('balance_sheet_analysis', {}).get('debt_to_equity', 0)),
        'current_ratio': float(financial_analysis.get('balance_sheet_analysis', {}).get('current_ratio', 0)),
        
        # Valuation
        'trailing_pe': float(fundamental_info.get('trailingPE', 0)) if fundamental_info.get('trailingPE') else 0.0,
        'price_to_book': float(fundamental_info.get('priceToBook', 0)) if fundamental_info.get('priceToBook') else 0.0,
        'peg_ratio': float(fundamental_info.get('pegRatio', 0)) if fundamental_info.get('pegRatio') else None,
        'beta': float(fundamental_info.get('beta', 1.0)) if fundamental_info.get('beta') else 1.0,
        
        # SWOT
        'strengths': swot.get('Strengths', []),
        'weaknesses': swot.get('Weaknesses', []),
        'opportunities': swot.get('Opportunities', []),
        'threats': swot.get('Threats', []),
        
        # Full analysis objects
        'financial_analysis': financial_analysis,
        'fundamental_info': fundamental_info,
    }


# ============================================================================
# SHEET CREATION FUNCTIONS
# ============================================================================

def create_executive_dashboard(writer: pd.ExcelWriter, analysis: Dict[str, Any], 
                              symbol: str, common_data: Dict[str, Any]) -> None:
    """Create Executive Summary Dashboard - The Most Important Sheet"""
    
    # Extract data efficiently
    current_price = common_data['current_price']
    entry_point = common_data['entry_point']
    stop_loss = common_data['stop_loss']
    target_1 = common_data['target_1']
    target_2 = common_data['target_2']
    
    # Calculate returns (batch calculation)
    if entry_point > 0:
        potential_profit_t1 = ((target_1 / entry_point) - 1) * 100
        potential_profit_t2 = ((target_2 / entry_point) - 1) * 100
        potential_loss = ((stop_loss / entry_point) - 1) * 100
    else:
        potential_profit_t1 = potential_profit_t2 = potential_loss = 0.0
    
    # Risk/Reward Ratio
    risk = abs(entry_point - stop_loss)
    reward_t1 = abs(target_1 - entry_point)
    risk_reward = reward_t1 / risk if risk > 0 else 0.0
    
    # Overall rating (composite score)
    overall_rating = (common_data['technical_score'] + 
                     common_data['financial_score'] + 
                     common_data['swot_score']) / 3
    
    # Build dashboard data
    dashboard_data = [
        ['', ''],  # Empty row for spacing
        ['📊 INVESTMENT RECOMMENDATION', ''],
        ['Stock Symbol', symbol],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ['', ''],
        
        ['🎯 FINAL DECISION', common_data['decision'].upper()],
        ['Confidence Level', f"{common_data['confidence']:.1f}%"],
        ['', ''],
        
        ['💰 PRICE LEVELS', ''],
        ['Current Price', f"${current_price:.2f}"],
        ['Entry Point', f"${entry_point:.2f}"],
        ['Stop Loss', f"${stop_loss:.2f}"],
        ['Target 1', f"${target_1:.2f}"],
        ['Target 2', f"${target_2:.2f}"],
        ['', ''],
        
        ['📈 POTENTIAL RETURNS (from Entry)', ''],
        ['Target 1 Profit', f"{potential_profit_t1:+.1f}%"],
        ['Target 2 Profit', f"{potential_profit_t2:+.1f}%"],
        ['Maximum Loss', f"{potential_loss:.1f}%"],
        ['Risk/Reward Ratio', f"1:{risk_reward:.2f}"],
        ['', ''],
        
        ['📊 ANALYSIS SCORES', ''],
        ['Technical Analysis', f"{common_data['technical_score']:.0f}/100"],
        ['Financial Health', f"{common_data['financial_score']:.0f}/100"],
        ['SWOT Score', f"{common_data['swot_score']:.0f}/100"],
        ['Overall Rating', f"{overall_rating:.0f}/100"],
        ['', ''],
        
        ['🎯 INVESTMENT SIZING', ''],
        ['Shares to Buy', f"{common_data['shares_can_buy']}"],
        ['Total Investment', f"${common_data['total_invested']:.2f}"],
        ['Remaining Cash', f"${common_data['remaining_cash']:.2f}"],
        ['', ''],
        
        ['⚠️ KEY INSIGHTS', ''],
        ['Trend Prediction', common_data['prediction']],
        ['Risk Level', common_data['risk_rating']],
        ['Market Sentiment', get_market_sentiment(common_data)],
    ]
    
    # Add recommendation rationale
    dashboard_data.extend([
        ['', ''],
        ['💡 WHY THIS RECOMMENDATION?', ''],
        ['Main Opportunity', get_main_item(common_data['opportunities'])],
        ['Main Risk', get_main_item(common_data['threats'])],
    ])
    
    df = pd.DataFrame(dashboard_data, columns=['Metric', 'Value'])
    df.to_excel(writer, sheet_name='📊 Executive Summary', index=False)
    
    # Apply special formatting to this sheet
    ws = writer.sheets['📊 Executive Summary']
    
    # Make decision cell prominent
    decision_row = 6  # Row with FINAL DECISION
    decision_cell = ws.cell(row=decision_row, column=2)
    
    decision_upper = common_data['decision'].upper()
    if 'STRONG BUY' in decision_upper or decision_upper == 'BUY':
        decision_cell.fill = ExcelFormats.STRONG_BUY_FILL
    elif 'STRONG SELL' in decision_upper or decision_upper == 'SELL':
        decision_cell.fill = ExcelFormats.STRONG_SELL_FILL
    else:  # HOLD
        decision_cell.fill = ExcelFormats.NEUTRAL_FILL
    
    decision_cell.font = Font(size=16, bold=True, color='FFFFFF')
    decision_cell.alignment = ExcelFormats.CENTER_ALIGN
    
    # Color code returns (batch processing)
    for row in range(17, 20):  # Potential returns rows
        cell = ws.cell(row=row, column=2)
        try:
            value_str = str(cell.value).replace('%', '').replace('+', '').strip()
            value = float(value_str)
            if value > 0:
                cell.font = Font(color='00C851', bold=True)
            elif value < 0:
                cell.font = Font(color='FF4444', bold=True)
        except (ValueError, AttributeError):
            pass


def create_investment_decision_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any],
                                    common_data: Dict[str, Any]) -> None:
    """Detailed Investment Decision with reasoning"""
    
    decision_data = [
        ['Investment Decision Report', ''],
        ['', ''],
        ['Final Recommendation', common_data['decision']],
        ['Confidence Level', f"{common_data['confidence']:.1f}%"],
        ['Risk Rating', common_data['risk_rating']],
        ['Reward/Risk Ratio', f"{common_data['reward_to_risk']:.2f}:1"],
        ['', ''],
        
        ['Price Action Analysis', ''],
        ['Current Price', f"${common_data['current_price']:.2f}"],
        ['Recommended Entry', f"${common_data['entry_point']:.2f}"],
        ['Stop Loss Level', f"${common_data['stop_loss']:.2f}"],
        ['Exit Target 1', f"${common_data['target_1']:.2f}"],
        ['Exit Target 2', f"${common_data['target_2']:.2f}"],
        ['', ''],
        
        ['Supporting Factors', ''],
    ]
    
    # Add supporting factors (max 3 each)
    for i, strength in enumerate(common_data['strengths'][:3], 1):
        decision_data.append([f'Strength {i}', strength])
    
    for i, opp in enumerate(common_data['opportunities'][:3], 1):
        decision_data.append([f'Opportunity {i}', opp])
    
    decision_data.append(['', ''])
    decision_data.append(['Risk Factors', ''])
    
    # Add risk factors (max 3 each)
    for i, weakness in enumerate(common_data['weaknesses'][:3], 1):
        decision_data.append([f'Weakness {i}', weakness])
    
    for i, threat in enumerate(common_data['threats'][:3], 1):
        decision_data.append([f'Threat {i}', threat])
    
    df = pd.DataFrame(decision_data, columns=['Category', 'Details'])
    df.to_excel(writer, sheet_name='💼 Investment Decision', index=False)


def create_price_targets_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any],
                               common_data: Dict[str, Any]) -> None:
    """Create comprehensive price targets sheet"""
    
    entry = common_data['entry_point']
    current = common_data['current_price']
    stop_loss = common_data['stop_loss']
    
    # Calculate returns (vectorized where possible)
    def calc_returns(price: float, base_entry: float, base_current: float) -> Tuple[str, str]:
        """Calculate return percentages from entry and current price"""
        from_entry = f"{((price/base_entry - 1) * 100):+.1f}%" if base_entry > 0 else "N/A"
        from_current = f"{((price/base_current - 1) * 100):+.1f}%" if base_current > 0 else "N/A"
        return from_entry, from_current
    
    # Build price targets table
    targets_data = [
        ['Level', 'Type', 'Price', 'From Entry (%)', 'From Current (%)', 'Notes']
    ]
    
    # Current price row
    from_entry, from_current = calc_returns(current, entry, current)
    targets_data.append([
        'Current Price', 'Reference', f"${current:.2f}",
        from_entry, '0.0%', 'Market Price'
    ])
    
    # Entry point row
    from_entry_e, from_current_e = calc_returns(entry, entry, current)
    targets_data.append([
        'Entry Point', 'Buy Zone', f"${entry:.2f}",
        '0.0%', from_current_e, 'Optimal Entry'
    ])
    
    # Stop loss row
    from_entry_s, from_current_s = calc_returns(stop_loss, entry, current)
    targets_data.append([
        'Stop Loss', 'Exit', f"${stop_loss:.2f}",
        from_entry_s, from_current_s, 'Maximum Loss'
    ])
    
    # Add upside targets
    for i, target in enumerate(common_data['up_targets'][:3], 1):
        from_entry_t, from_current_t = calc_returns(target, entry, current)
        targets_data.append([
            f'Target {i}',
            'Profit Target',
            f"${target:.2f}",
            from_entry_t,
            from_current_t,
            f'Take Profit Level {i}'
        ])
    
    # Add analyst target if available
    analyst_avg = analysis.get('analyst_avg_target')
    if analyst_avg and not np.isnan(analyst_avg):
        from_entry_a, from_current_a = calc_returns(analyst_avg, entry, current)
        targets_data.append([
            'Analyst Average',
            'Consensus',
            f"${analyst_avg:.2f}",
            from_entry_a,
            from_current_a,
            'Analyst Consensus Target'
        ])
    
    df = pd.DataFrame(targets_data[1:], columns=targets_data[0])
    df.to_excel(writer, sheet_name='🎯 Price Targets', index=False)


def create_technical_summary_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any],
                                   common_data: Dict[str, Any]) -> None:
    """Technical analysis summary with key indicators"""
    
    tech_df = analysis.get('technical_data', pd.DataFrame())
    
    if tech_df.empty:
        df = pd.DataFrame([['No technical data available', '']], columns=['Indicator', 'Value'])
        df.to_excel(writer, sheet_name='📈 Technical Summary', index=False)
        return
    
    latest = tech_df.iloc[-1]
    
    # Extract values once
    close_price = latest.get('Close', 0)
    sma_20 = latest.get('SMA_20', 0)
    sma_50 = latest.get('SMA_50', 0)
    rsi_14 = latest.get('RSI_14', 0)
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    adx = latest.get('ADX', 0)
    stoch_k = latest.get('Stoch_K', 0)
    bb_pct = latest.get('Bollinger_%B', 0)
    
    # Key technical indicators
    tech_summary = [
        ['Technical Indicator', 'Current Value', 'Signal', 'Interpretation'],
        
        ['Trend Analysis', '', '', ''],
        ['SMA 20', f"${sma_20:.2f}", 
         get_sma_signal(close_price, sma_20),
         'Short-term trend'],
        ['SMA 50', f"${sma_50:.2f}", 
         get_sma_signal(close_price, sma_50),
         'Medium-term trend'],
        ['ADX', f"{adx:.2f}", 
         apply_rating(adx, RatingThresholds.ADX),
         'Trend strength'],
        
        ['Momentum Indicators', '', '', ''],
        ['RSI (14)', f"{rsi_14:.2f}", 
         get_rsi_signal(rsi_14),
         'Overbought/Oversold'],
        ['MACD', f"{macd:.3f}", 
         get_macd_signal(macd, macd_signal),
         'Momentum direction'],
        ['Stochastic', f"{stoch_k:.2f}", 
         get_stoch_signal(stoch_k),
         'Momentum oscillator'],
        
        ['Volatility & Support', '', '', ''],
        ['Bollinger %B', f"{bb_pct:.3f}", 
         get_bb_signal(bb_pct),
         'Price position in bands'],
        ['Support Level', f"${common_data['support_level']:.2f}", 
         'Key Level', 'Major support'],
        ['Resistance Level', f"${common_data['resistance_level']:.2f}", 
         'Key Level', 'Major resistance'],
        
        ['Overall Assessment', '', '', ''],
        ['Technical Score', f"{common_data['technical_score']:.0f}/100", 
         apply_rating(common_data['technical_score'], RatingThresholds.SCORE),
         'Combined technical rating'],
        ['Prediction', common_data['prediction'], 
         common_data['prediction'],
         'Expected trend'],
    ]
    
    df = pd.DataFrame(tech_summary[1:], columns=tech_summary[0])
    df.to_excel(writer, sheet_name='📈 Technical Summary', index=False)


def create_financial_summary_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any],
                                   common_data: Dict[str, Any]) -> None:
    """Financial health summary"""
    
    fin_analysis = common_data['financial_analysis']
    
    # Financial summary data
    fin_summary = [
        ['Financial Metric', 'Value', 'Rating', 'Industry Avg'],
        
        ['Overall Health', '', '', ''],
        ['Financial Score', f"{common_data['financial_score']:.0f}/100",
         fin_analysis.get('health_rating', 'N/A'), 'N/A'],
        ['Health Status', fin_analysis.get('health_description', 'N/A'), '', ''],
        
        ['Profitability', '', '', ''],
        ['Net Margin', f"{common_data['net_margin']:.1f}%",
         apply_rating(common_data['net_margin'], RatingThresholds.MARGIN),
         '10-15%'],
        ['Gross Margin', f"{common_data['gross_margin']:.1f}%",
         apply_rating(common_data['gross_margin'], RatingThresholds.MARGIN),
         '30-40%'],
        ['ROE', f"{common_data['roe']:.1f}%",
         apply_rating(common_data['roe'], RatingThresholds.ROE),
         '15-20%'],
        
        ['Growth', '', '', ''],
        ['Revenue Growth', f"{fin_analysis.get('revenue_analysis', {}).get('revenue_growth_%', 0):.1f}%",
         fin_analysis.get('revenue_analysis', {}).get('revenue_trend', 'N/A'),
         '10-15%'],
        
        ['Financial Stability', '', '', ''],
        ['Debt/Equity', f"{common_data['debt_to_equity']:.2f}",
         apply_rating(common_data['debt_to_equity'], RatingThresholds.DEBT),
         '<1.0'],
        ['Current Ratio', f"{common_data['current_ratio']:.2f}",
         apply_rating(common_data['current_ratio'], RatingThresholds.LIQUIDITY),
         '>1.5'],
        
        ['Valuation', '', '', ''],
        ['P/E Ratio', f"{common_data['trailing_pe']:.2f}" if common_data['trailing_pe'] else 'N/A',
         apply_pe_rating(common_data['trailing_pe']),
         '15-25'],
        ['P/B Ratio', f"{common_data['price_to_book']:.2f}" if common_data['price_to_book'] else 'N/A',
         apply_rating(common_data['price_to_book'], RatingThresholds.PB) if common_data['price_to_book'] else 'N/A',
         '<3.0'],
        ['PEG Ratio', f"{common_data['peg_ratio']:.2f}" if common_data['peg_ratio'] else 'N/A',
         apply_peg_rating(common_data['peg_ratio']),
         '<1.5'],
    ]
    
    df = pd.DataFrame(fin_summary[1:], columns=fin_summary[0])
    df.to_excel(writer, sheet_name='💰 Financial Health', index=False)


def create_swot_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any]) -> None:
    """SWOT Analysis with better formatting"""
    
    swot = analysis.get('swot', {})
    
    swot_data = [
        ['SWOT Category', 'Factor', 'Details'],
    ]
    
    for category in ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']:
        items = swot.get(category, [])
        if items:
            for i, item in enumerate(items, 1):
                swot_data.append([
                    category if i == 1 else '',
                    f'{category[:-1]} {i}',
                    item
                ])
        else:
            swot_data.append([category, 'N/A', 'No data available'])
    
    df = pd.DataFrame(swot_data[1:], columns=swot_data[0])
    df.to_excel(writer, sheet_name='🎯 SWOT Analysis', index=False)


def create_analytics_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any],
                           common_data: Dict[str, Any]) -> None:
    """Advanced analytics metrics"""
    
    overall_score = (common_data['technical_score'] + 
                    common_data['financial_score'] + 
                    common_data['swot_score']) / 3
    
    analytics_data = [
        ['Advanced Metric', 'Value', 'Interpretation'],
        
        ['Entry & Exit Strategy', '', ''],
        ['Buy Trigger', f"${common_data['entry_point']:.2f}", 'Optimal buy price'],
        ['Sell Trigger', f"${common_data['target_1']:.2f}", 'Profit target price'],
        ['Stop Loss', f"${common_data['stop_loss']:.2f}", 'Maximum loss level'],
        
        ['Risk Assessment', '', ''],
        ['Risk Level', common_data['risk_rating'], 'Overall risk category'],
        ['Volatility', f"{get_volatility(analysis):.1f}%", 'Price volatility'],
        ['Beta', f"{common_data['beta']:.2f}", 'Market correlation'],
        
        ['Performance Metrics', '', ''],
        ['Technical Score', f"{common_data['technical_score']:.0f}/100", 'Technical strength'],
        ['Financial Score', f"{common_data['financial_score']:.0f}/100", 'Financial strength'],
        ['SWOT Score', f"{common_data['swot_score']:.0f}/100", 'Strategic position'],
        ['Overall Score', f"{overall_score:.0f}/100", 'Combined rating'],
    ]
    
    df = pd.DataFrame(analytics_data[1:], columns=analytics_data[0])
    df.to_excel(writer, sheet_name='📊 Analytics', index=False)


def create_detailed_technical_sheet(writer: pd.ExcelWriter, analysis: Dict[str, Any]) -> None:
    """Detailed technical data (last 30 days)"""
    
    technical_df = analysis.get('technical_data', pd.DataFrame())
    
    if technical_df.empty:
        df = pd.DataFrame([['No data', '']], columns=['Info', 'Value'])
        df.to_excel(writer, sheet_name='📉 Technical Data', index=False)
        return
    
    # Select important columns (predefined list)
    cols_to_export = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal',
        'ADX', 'Bollinger_%B', 'BB_Upper', 'BB_Lower',
        'Stoch_K', 'Pivot', 'R1', 'S1'
    ]
    
    available_cols = [c for c in cols_to_export if c in technical_df.columns]
    
    # Last 30 days
    df = technical_df[available_cols].tail(30).copy()
    
    # Format date if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    df.to_excel(writer, sheet_name='📉 Technical Data', index=False)


def create_financial_statements_sheets(writer: pd.ExcelWriter, analysis: Dict[str, Any]) -> None:
    """Create financial statements sheets"""
    
    fundamental_data = analysis.get('fundamental_data_full', [pd.DataFrame()] * 8)
    
    # Unpack with safe defaults
    if len(fundamental_data) >= 8:
        (financials, balance_sheet, cashflow,
         quarterly_financials, quarterly_balance_sheet,
         quarterly_cashflow, earnings, quarterly_earnings) = fundamental_data
    else:
        # Handle case where not all dataframes are present
        financials = balance_sheet = cashflow = pd.DataFrame()
        quarterly_financials = quarterly_balance_sheet = quarterly_cashflow = pd.DataFrame()
        earnings = quarterly_earnings = pd.DataFrame()
    
    # Define sheet mappings
    financial_sheets = [
        (financials, '📋 Income Statement'),
        (balance_sheet, '📋 Balance Sheet'),
        (cashflow, '📋 Cash Flow'),
        (earnings, '📋 Earnings'),
    ]
    
    # Process sheets
    for df, sheet_name in financial_sheets:
        if df is not None and not df.empty:
            try:
                # Convert datetime columns
                df_copy = df.copy()
                df_copy.columns = [
                    col.strftime('%Y-%m-%d') if isinstance(col, (datetime, pd.Timestamp)) else col
                    for col in df_copy.columns
                ]
                df_copy.to_excel(writer, sheet_name=sheet_name, index=True)
            except Exception:
                # Skip sheets that fail to process
                continue


def apply_professional_formatting(writer: pd.ExcelWriter) -> None:
    """Apply professional formatting to all sheets"""
    
    workbook = writer.book
    
    for sheet_name, sheet in writer.sheets.items():
        # Freeze panes
        if sheet.max_row > 1:
            sheet.freeze_panes = 'B2'
        
        # Header formatting (first row)
        for cell in sheet[1]:
            cell.font = ExcelFormats.HEADER_FONT
            cell.fill = ExcelFormats.HEADER_FILL
            cell.alignment = ExcelFormats.CENTER_ALIGN
        
        # Auto-adjust column widths (optimized)
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if cell.value:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass
            
            # Set width with limits
            adjusted_width = min(max_length + 2, 50)
            sheet.column_dimensions[column_letter].width = max(adjusted_width, 10)


# ============================================================================
# UNIFIED RATING SYSTEM
# ============================================================================

def apply_rating(value: float, thresholds: List[Tuple]) -> str:
    """
    Unified rating function using threshold lists.
    Reduces code duplication significantly.
    
    Args:
        value: Value to rate
        thresholds: List of (threshold, rating) tuples
        
    Returns:
        Rating string
    """
    for threshold, rating in thresholds:
        if isinstance(threshold, tuple):
            # Range-based rating (for PE ratio)
            continue
        if value >= threshold:
            return rating
    return thresholds[-1][1]  # Return last rating as default


def apply_pe_rating(pe: float) -> str:
    """Special handler for PE ratio (range-based)"""
    if not pe or pe <= 0:
        return 'N/A'
    for min_val, max_val, rating in RatingThresholds.PE:
        if min_val < pe <= max_val:
            return rating
    return '❌ Overvalued?'


def apply_peg_rating(peg: Optional[float]) -> str:
    """Rate PEG ratio"""
    if peg is None or peg <= 0:
        return 'N/A'
    if peg < 1:
        return '⭐⭐⭐ Undervalued'
    elif peg < 1.5:
        return '⭐⭐ Fair'
    return '❌ Overvalued'


# ============================================================================
# SIGNAL HELPER FUNCTIONS
# ============================================================================

def get_sma_signal(price: float, sma: float) -> str:
    """Get SMA signal"""
    if price > sma:
        return '🟢 Bullish'
    elif price < sma:
        return '🔴 Bearish'
    return '🟡 Neutral'


def get_rsi_signal(rsi: float) -> str:
    """Get RSI signal"""
    if rsi > 70:
        return '🔴 Overbought'
    elif rsi < 30:
        return '🟢 Oversold'
    return '🟡 Neutral'


def get_macd_signal(macd: float, signal: float) -> str:
    """Get MACD signal"""
    return '🟢 Bullish' if macd > signal else '🔴 Bearish'


def get_stoch_signal(stoch: float) -> str:
    """Get Stochastic signal"""
    if stoch > 80:
        return '🔴 Overbought'
    elif stoch < 20:
        return '🟢 Oversold'
    return '🟡 Neutral'


def get_bb_signal(bb_pct: float) -> str:
    """Get Bollinger Band signal"""
    if bb_pct > 1:
        return '🔴 Above Bands'
    elif bb_pct < 0:
        return '🟢 Below Bands'
    return '🟡 Within Bands'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_market_sentiment(common_data: Dict[str, Any]) -> str:
    """Get market sentiment from analysis"""
    decision = common_data['decision']
    confidence = common_data['confidence']
    
    if 'BUY' in decision.upper() and confidence > 70:
        return '🟢 Very Bullish'
    elif 'BUY' in decision.upper():
        return '🟢 Bullish'
    elif 'SELL' in decision.upper() and confidence > 70:
        return '🔴 Very Bearish'
    elif 'SELL' in decision.upper():
        return '🔴 Bearish'
    return '🟡 Neutral'


def get_main_item(items: List[str]) -> str:
    """Get first item from list or N/A"""
    return items[0] if items else 'N/A'


def get_volatility(analysis: Dict[str, Any]) -> float:
    """Calculate volatility from technical data"""
    tech_df = analysis.get('technical_data', pd.DataFrame())
    if not tech_df.empty and 'Close' in tech_df.columns:
        try:
            returns = tech_df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            return float(volatility)
        except:
            pass
    return 20.0  # Default volatility