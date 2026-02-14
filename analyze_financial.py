import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, Optional
warnings.filterwarnings('ignore')

def debug_dataframes(income_stmt, balance_sheet, cash_flow, symbol=""):
    """
    Debug function to check what data is available in the dataframes
    """
    print(f"\n=== DEBUG INFO FOR {symbol} ===")
    
    print(f"\nIncome Statement Shape: {income_stmt.shape}")
    print("Income Statement Index:", list(income_stmt.index) if not income_stmt.empty else "Empty")
    
    print(f"\nBalance Sheet Shape: {balance_sheet.shape}")  
    print("Balance Sheet Index:", list(balance_sheet.index) if not balance_sheet.empty else "Empty")
    
    print(f"\nCash Flow Shape: {cash_flow.shape}")
    print("Cash Flow Index:", list(cash_flow.index) if not cash_flow.empty else "Empty")
    
    return True

# Pre-computed key variations as a module-level constant for better performance
_KEY_VARIATIONS = {
    'Total Revenue': [
        'Total Revenue', 'Revenue', 'Net Sales', 'Total Net Revenues', 'Net Revenue',
        'Sales', 'Net Sales Revenue', 'Total Revenues'
    ],
    'Net Income': [
        'Net Income', 'Net Income Common Stockholders', 
        'Net Income Applicable To Common Shares', 'Net Income Available to Common Stockholders',
        'Net Earnings', 'Net Income (Loss)'
    ],
    'Gross Profit': [
        'Gross Profit', 'Total Gross Profit', 'Gross Income'
    ],
    'Operating Income': [
        'Operating Income', 'Operating Income Or Loss', 'Income From Operations',
        'Operating Profit', 'Operating Earnings'
    ],
    'Total Assets': [
        'Total Assets', 'Assets', 'Total Assets'
    ],
    'Total Debt': [
        'Total Debt', 'Total Debt And Capital Lease Obligation', 
        'Long Term Debt And Capital Lease Obligation', 'Net Debt',
        'Long Term Debt', 'Short Long Term Debt Total', 'Total Debt',
        'Current Debt And Capital Lease Obligation', 'Short Term Debt'
    ],
    'Stockholders Equity': [
        'Stockholders Equity', 'Total Stockholders Equity', 
        'Total Equity Gross Minority Interest', 'Shareholders Equity',
        'Total Equity', 'Stockholders\' Equity'
    ],
    'Current Assets': [
        'Current Assets', 'Total Current Assets'
    ],
    'Current Liabilities': [
        'Current Liabilities', 'Total Current Liabilities'
    ],
    'Operating Cash Flow': [
        'Operating Cash Flow', 'Total Cash From Operating Activities', 
        'Cash Flow From Operations', 'Net Cash Provided By Operating Activities',
        'Cash From Operating Activities', 'Operating Activities',
        'Net Cash From Operating Activities', 'Cash Flow From Operating Activities',
        'Cash Flow From Continuing Operating Activities'
    ],
    'Capital Expenditure': [
        'Capital Expenditure', 'Capital Expenditures', 'Purchase Of PPE',
        'Purchases Of Property Plant And Equipment', 'Capital Spending',
        'Purchase Of Property And Equipment', 'Capex', 'Purchase Of Business',
        'Purchase Of Property, Plant And Equipment', 'Purchase of Property, Plant, and Equipment',
        'Purchases of property, plant and equipment'
    ],
    'EBIT': [
        'EBIT', 'Earnings Before Interest And Taxes', 'Operating Income',
        'Operating Profit'
    ],
    'Interest Expense': [
        'Interest Expense', 'Interest Expense Non Operating', 'Net Interest Expense',
        'Interest Paid', 'Interest Expense, Net'
    ],
    'Inventory': [
        'Inventory', 'Total Inventory', 'Inventories'
    ],
    'Common Stock': [
        'Common Stock', 'Ordinary Shares Number', 'Share Issued',
        'Common Shares Outstanding', 'Shares Outstanding'
    ]
}

def safe_get_value(df: pd.DataFrame, key: str, column_idx: int = 0, 
                   default: float = 0.0) -> float:
    """
    Optimized safe extraction of values from DataFrame with multiple fallback options.
    
    Performance optimizations:
    - Early exit on empty DataFrames
    - Direct index lookup first (most common case)
    - Fallback to variations only when needed
    - Single try-except per variation
    
    Args:
        df: Source DataFrame
        key: Primary key to lookup
        column_idx: Column index to extract
        default: Default value if not found
        
    Returns:
        Extracted float value or default
    """
    # Early exit for empty dataframes or invalid column
    if df.empty or len(df.columns) <= column_idx:
        return default
    
    # Direct match attempt (most common case - optimize for this)
    if key in df.index:
        try:
            value = df.loc[key].iloc[column_idx]
            if pd.notna(value):
                return float(value)
        except (ValueError, TypeError, IndexError):
            pass
    
    # Fallback to variations only if direct match fails
    variations = _KEY_VARIATIONS.get(key)
    if variations:
        for variation in variations:
            if variation in df.index:
                try:
                    value = df.loc[variation].iloc[column_idx]
                    if pd.notna(value):
                        return float(value)
                except (ValueError, TypeError, IndexError):
                    continue
    
    return default

def get_cash_flow_data(cash_flow: pd.DataFrame, income_stmt: pd.DataFrame, 
                       basic_info: dict) -> Tuple[float, float, float]:
    """
    Optimized cash flow data extraction with better fallback logic.
    
    Returns:
        Tuple of (operating_cash_flow, capex, free_cash_flow)
    """
    # Operating Cash Flow - try DataFrame first, then fallback
    ocf = safe_get_value(cash_flow, 'Operating Cash Flow', 0, 0.0)
    if ocf == 0.0:
        ocf_info = basic_info.get('operatingCashflow')
        if ocf_info:
            try:
                ocf = float(ocf_info)
            except (ValueError, TypeError):
                ocf = 0.0
    
    # Capital Expenditure - ensure absolute value
    capex = abs(safe_get_value(cash_flow, 'Capital Expenditure', 0, 0.0))
    
    # Fallback for CapEx from basic_info
    if capex == 0.0:
        for key in ('capitalExpenditures', 'capex'):
            val = basic_info.get(key)
            if val:
                try:
                    capex = abs(float(val))
                    break
                except (ValueError, TypeError):
                    continue
    
    # Free Cash Flow calculation
    fcf = ocf - capex
    
    # Override with direct FCF if available and more reliable
    fcf_direct = basic_info.get('freeCashflow')
    if fcf_direct:
        try:
            fcf_direct = float(fcf_direct)
            if fcf_direct != 0:
                fcf = fcf_direct
        except (ValueError, TypeError):
            pass
    
    return ocf, capex, fcf

def get_balance_sheet_ratios(balance_sheet: pd.DataFrame, 
                            basic_info: dict) -> Tuple[float, float, float]:
    """
    Optimized balance sheet ratio extraction.
    
    Returns:
        Tuple of (total_debt, total_assets, debt_to_assets_ratio)
    """
    # Total Debt - try direct, then sum components, then basic_info
    total_debt = safe_get_value(balance_sheet, 'Total Debt', 0, 0.0)
    
    if total_debt == 0.0:
        long_term_debt = safe_get_value(balance_sheet, 'Long Term Debt', 0, 0.0)
        short_term_debt = safe_get_value(balance_sheet, 'Current Debt', 0, 0.0)
        total_debt = long_term_debt + short_term_debt
        
        if total_debt == 0.0:
            debt_info = basic_info.get('totalDebt')
            if debt_info:
                try:
                    total_debt = float(debt_info)
                except (ValueError, TypeError):
                    total_debt = 0.0
    
    # Total Assets - try DataFrame first, then basic_info
    total_assets = safe_get_value(balance_sheet, 'Total Assets', 0, 0.0)
    if total_assets == 0.0:
        assets_info = basic_info.get('totalAssets')
        if assets_info:
            try:
                total_assets = float(assets_info)
            except (ValueError, TypeError):
                total_assets = 0.0
    
    # Debt to Assets Ratio - single calculation
    debt_to_assets = (total_debt / total_assets) if total_assets > 0 else 0.0
    
    return total_debt, total_assets, debt_to_assets

def _safe_float_conversion(value) -> float:
    """
    Helper for safe float conversion from basic_info.
    Handles None, empty strings, and 'N/A' values.
    """
    if value is None or value == '' or value == 'N/A':
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def _calculate_growth_rate(current: float, previous: float) -> float:
    """
    Centralized growth rate calculation with validation.
    
    Returns:
        Growth rate as percentage, or 0.0 if invalid
    """
    if previous <= 0 or current == previous:
        return 0.0
    return ((current - previous) / abs(previous)) * 100

def _score_range(value: float, thresholds: list, scores: list) -> float:
    """
    Generic scoring function based on thresholds.
    Assumes thresholds are in descending order.
    """
    for threshold, score in zip(thresholds, scores):
        if value > threshold:
            return score
    return scores[-1]

def analyze_financial_performance(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    quarterly_income: pd.DataFrame,
    quarterly_balance: pd.DataFrame,
    quarterly_cash: pd.DataFrame,
    basic_info: dict,
    industry_pe: Optional[float] = None,
    debug: bool = False
) -> dict:
    """
    Optimized comprehensive financial analysis with improved data extraction.
    
    Performance improvements:
    - Batch data extraction to minimize DataFrame lookups
    - Pre-computed ratios to avoid redundant calculations
    - Efficient scoring with lookup tables
    - Better null handling and validation
    
    Financial logic improvements:
    - Non-linear scoring for nuanced assessment
    - Multi-factor risk analysis
    - Trend detection in growth metrics
    - Industry-relative valuation
    
    Args:
        income_stmt: Annual income statement
        balance_sheet: Annual balance sheet
        cash_flow: Annual cash flow statement
        quarterly_income: Quarterly income statement
        quarterly_balance: Quarterly balance sheet
        quarterly_cash: Quarterly cash flow statement
        basic_info: Company basic information
        industry_pe: Industry average P/E ratio
        debug: Enable debug output
        
    Returns:
        Comprehensive financial analysis dictionary
    """
    if debug:
        debug_dataframes(income_stmt, balance_sheet, cash_flow, basic_info.get('symbol', ''))
    
    # Initialize analysis structure once
    analysis = {
        'overall_score': 0,
        'health_rating': 'N/A',
        'health_description': 'Insufficient data',
        'ratios': {},
        'revenue_analysis': {},
        'profitability_analysis': {},
        'balance_sheet_analysis': {},
        'cash_flow_analysis': {},
        'valuation_analysis': {},
        'asset_efficiency': {},
        'risk_analysis': {},
    }
    
    score = 0
    
    try:
        # ===== BATCH DATA EXTRACTION =====
        # Extract all needed values in one section to minimize DataFrame operations
        
        # Revenue metrics (current and previous year)
        curr_rev = safe_get_value(income_stmt, 'Total Revenue', 0, 0.0)
        prev_rev = safe_get_value(income_stmt, 'Total Revenue', 1, curr_rev)
        
        # Income metrics
        curr_net_income = safe_get_value(income_stmt, 'Net Income', 0, 0.0)
        prev_net_income = safe_get_value(income_stmt, 'Net Income', 1, curr_net_income)
        gross_profit = safe_get_value(income_stmt, 'Gross Profit', 0, 0.0)
        operating_income = safe_get_value(income_stmt, 'Operating Income', 0, 0.0)
        
        # Balance sheet metrics
        stockholders_equity = safe_get_value(balance_sheet, 'Stockholders Equity', 0, 1.0)
        curr_assets = safe_get_value(balance_sheet, 'Current Assets', 0, 0.0)
        curr_liab = safe_get_value(balance_sheet, 'Current Liabilities', 0, 1.0)
        
        # ===== PRE-COMPUTE COMMON VALUES =====
        # Avoid repeated division operations
        inv_rev = 1.0 / curr_rev if curr_rev > 0 else 0.0
        inv_equity = 1.0 / stockholders_equity if stockholders_equity > 0 else 0.0
        
        # ===== REVENUE ANALYSIS =====
        rev_growth = _calculate_growth_rate(curr_rev, prev_rev)
        
        # Enhanced revenue scoring with non-linear scale
        if rev_growth > 25:
            rev_score = 25
            rev_trend = '🟢 Exceptional Growth'
        elif rev_growth > 15:
            rev_score = 20
            rev_trend = '🟢 Strong Growth'
        elif rev_growth > 8:
            rev_score = 15
            rev_trend = '🟡 Solid Growth'
        elif rev_growth > 0:
            rev_score = 10
            rev_trend = '🟡 Positive Growth'
        elif rev_growth > -5:
            rev_score = 5
            rev_trend = '🟠 Slight Decline'
        elif rev_growth > -15:
            rev_score = -5
            rev_trend = '🔴 Moderate Decline'
        else:
            rev_score = -15
            rev_trend = '🔴 Severe Decline'
        
        score += rev_score
        
        analysis['revenue_analysis'] = {
            'current_revenue': curr_rev,
            'previous_revenue': prev_rev,
            'revenue_growth_%': round(rev_growth, 2),
            'revenue_trend': rev_trend
        }

        # ===== PROFITABILITY ANALYSIS =====
        # Calculate all margins in one section
        gross_margin = (gross_profit * inv_rev * 100) if curr_rev > 0 else 0.0
        operating_margin = (operating_income * inv_rev * 100) if curr_rev > 0 else 0.0
        net_margin = (curr_net_income * inv_rev * 100) if curr_rev > 0 else 0.0
        
        # ROE and ROA
        roe = (curr_net_income * inv_equity * 100) if stockholders_equity > 0 else 0.0
        total_debt, total_assets, debt_to_assets = get_balance_sheet_ratios(balance_sheet, basic_info)
        roa = (curr_net_income / total_assets * 100) if total_assets > 0 else 0.0
        
        # Net income growth
        net_income_growth = _calculate_growth_rate(curr_net_income, prev_net_income)
        
        # Enhanced profitability scoring
        profit_score = 0
        
        # Net margin scoring (non-linear for better differentiation)
        if net_margin > 20:
            profit_score += 15
            margin_trend = '🟢 Excellent Margins'
        elif net_margin > 12:
            profit_score += 12
            margin_trend = '🟢 Strong Margins'
        elif net_margin > 6:
            profit_score += 8
            margin_trend = '🟡 Good Margins'
        elif net_margin > 0:
            profit_score += 4
            margin_trend = '🟡 Positive Margins'
        else:
            profit_score -= 10
            margin_trend = '🔴 Negative Margins'
        
        # ROE scoring with industry context
        if roe > 20:
            profit_score += 10
            roe_trend = '🟢 Superior ROE'
        elif roe > 12:
            profit_score += 7
            roe_trend = '🟢 Strong ROE'
        elif roe > 6:
            profit_score += 4
            roe_trend = '🟡 Decent ROE'
        elif roe > 0:
            profit_score += 1
            roe_trend = '🟠 Low ROE'
        else:
            profit_score -= 5
            roe_trend = '🔴 Negative ROE'
        
        score += profit_score
        
        analysis['profitability_analysis'] = {
            'net_income': curr_net_income,
            'gross_margin_%': round(gross_margin, 2),
            'operating_margin_%': round(operating_margin, 2),
            'net_margin_%': round(net_margin, 2),
            'ROE_%': round(roe, 2),
            'net_income_growth_%': round(net_income_growth, 2),
            'margin_trend': margin_trend,
            'roe_trend': roe_trend
        }
        
        # Store ratios for external use
        analysis['ratios'] = {
            'current_ratio': 0.0,
            'debt_to_equity': 0.0,
            'roe': roe
        }

        # ===== BALANCE SHEET ANALYSIS =====
        # Debt to equity calculation
        debt_to_equity = (total_debt / stockholders_equity) if stockholders_equity > 0 else 0.0
        
        # Liquidity ratios
        current_ratio = 0.0
        if curr_liab > 0:
            current_ratio = curr_assets / curr_liab

        # Quick ratio calculation
        inventory = safe_get_value(balance_sheet, 'Inventory', 0, 0.0)
        quick_ratio = ((curr_assets - inventory) / curr_liab) if curr_liab > 0 else 0.0

        # Enhanced balance sheet scoring
        balance_score = 0
        
        # Leverage scoring (non-linear for risk assessment)
        if debt_to_equity < 0.2:
            balance_score += 18
            debt_trend = '🟢 Very Conservative'
        elif debt_to_equity < 0.4:
            balance_score += 15
            debt_trend = '🟢 Conservative Leverage'
        elif debt_to_equity < 0.7:
            balance_score += 10
            debt_trend = '🟡 Moderate Leverage'
        elif debt_to_equity < 1.2:
            balance_score += 4
            debt_trend = '🟠 Higher Leverage'
        elif debt_to_equity < 2.0:
            balance_score -= 5
            debt_trend = '🔴 High Leverage'
        else:
            balance_score -= 15
            debt_trend = '🔴 Excessive Leverage Risk'

        # Liquidity scoring
        if current_ratio > 3.0:
            balance_score += 12
            liq_trend = '🟢 Exceptional Liquidity'
        elif current_ratio > 2.0:
            balance_score += 10
            liq_trend = '🟢 Excellent Liquidity'
        elif current_ratio > 1.5:
            balance_score += 7
            liq_trend = '🟡 Good Liquidity'
        elif current_ratio > 1.0:
            balance_score += 4
            liq_trend = '🟠 Adequate Liquidity'
        elif current_ratio > 0.75:
            balance_score -= 5
            liq_trend = '🔴 Liquidity Concern'
        else:
            balance_score -= 12
            liq_trend = '🔴 Severe Liquidity Risk'

        score += balance_score
        
        # Update ratios
        analysis['ratios']['current_ratio'] = current_ratio
        analysis['ratios']['debt_to_equity'] = debt_to_equity

        analysis['balance_sheet_analysis'] = {
            'total_debt': total_debt,
            'total_assets': total_assets,
            'debt_to_equity': round(debt_to_equity, 2),
            'debt_to_assets_%': round(debt_to_assets * 100, 2),
            'current_ratio': round(current_ratio, 2),
            'quick_ratio': round(quick_ratio, 2),
            'debt_trend': debt_trend,
            'liquidity_trend': liq_trend
        }

        # ===== CASH FLOW ANALYSIS =====
        ocf, capex, fcf = get_cash_flow_data(cash_flow, income_stmt, basic_info)
        
        # Cash flow to revenue ratios
        ocf_to_revenue = (ocf * inv_rev * 100) if curr_rev > 0 else 0.0
        fcf_to_revenue = (fcf * inv_rev * 100) if curr_rev > 0 else 0.0

        # Interest coverage calculation (financial health indicator)
        ebit = safe_get_value(income_stmt, 'EBIT', 0, operating_income)
        interest_expense = abs(safe_get_value(income_stmt, 'Interest Expense', 0, 0.0))
        interest_coverage = (ebit / interest_expense) if interest_expense > 0 else None

        # Shares outstanding change (dilution analysis)
        shares_current = safe_get_value(balance_sheet, 'Common Stock', 0, 0.0)
        shares_previous = safe_get_value(balance_sheet, 'Common Stock', 1, shares_current)
        shares_change = _calculate_growth_rate(shares_current, shares_previous) if shares_previous > 0 else None

        # Enhanced cash flow scoring
        cash_score = 0
        
        if ocf > 0 and fcf > 0:
            if fcf_to_revenue > 15:
                cash_score = 20
                fcf_trend = '🟢 Exceptional Cash Generation'
            elif fcf_to_revenue > 10:
                cash_score = 16
                fcf_trend = '🟢 Strong Cash Generation'
            else:
                cash_score = 12
                fcf_trend = '🟢 Healthy Cash Flow'
        elif ocf > 0:
            cash_score = 8
            fcf_trend = '🟡 Positive Operating Cash Flow'
        elif fcf > 0:
            cash_score = 4
            fcf_trend = '🟠 Positive Free Cash Flow Only'
        elif ocf > -curr_rev * 0.05:  # Small negative OCF relative to revenue
            cash_score = -5
            fcf_trend = '🔴 Minor Cash Flow Issues'
        else:
            cash_score = -15
            fcf_trend = '🔴 Severe Cash Flow Concerns'

        score += cash_score

        analysis['cash_flow_analysis'] = {
            'operating_cash_flow': ocf,
            'free_cash_flow': fcf,
            'capital_expenditure': capex,
            'ocf_to_revenue_%': round(ocf_to_revenue, 2),
            'fcf_to_revenue_%': round(fcf_to_revenue, 2),
            'fcf_trend': fcf_trend,
            'interest_coverage': round(interest_coverage, 2) if interest_coverage else None,
            'shares_change': round(shares_change, 2) if shares_change else None
        }

        # ===== VALUATION ANALYSIS =====
        # Optimized PE lookup - single loop with early exit
        pe_val = None
        for pe_key in ('trailingPE', 'forwardPE', 'P/E TTM', 'P/E Ratio (TTM)'):
            val = basic_info.get(pe_key)
            if val not in (None, 'N/A', ''):
                pe_val = _safe_float_conversion(val)
                if pe_val > 0:  # Valid PE ratio
                    break
                pe_val = None

        # P/B ratio
        pb_val = _safe_float_conversion(basic_info.get('priceToBook'))
        if pb_val <= 0:
            pb_val = None

        # PEG ratio calculation (PE relative to growth)
        peg_ratio = None
        if pe_val and rev_growth > 0:
            peg_ratio = pe_val / rev_growth

        # Valuation trends with industry comparison
        if pe_val:
            if industry_pe and pe_val < industry_pe * 0.7:
                pe_trend = '🟢 Significantly Undervalued'
            elif industry_pe and pe_val < industry_pe * 0.9:
                pe_trend = '🟢 Attractive Valuation'
            elif pe_val < 15:
                pe_trend = '🟢 Low Valuation'
            elif pe_val < 22:
                pe_trend = '🟡 Fair Valuation'
            elif pe_val < 30:
                pe_trend = '🟠 Higher Valuation'
            elif pe_val < 40:
                pe_trend = '🔴 Expensive'
            else:
                pe_trend = '🔴 Very Expensive'
        else:
            pe_trend = '⚪ No Comparison Available'

        if pb_val:
            if pb_val < 0.8:
                pb_trend = '🟢 Deep Value (P/B < 0.8)'
            elif pb_val < 1.2:
                pb_trend = '🟢 Book Value Discount'
            elif pb_val < 2.5:
                pb_trend = '🟡 Reasonable P/B'
            elif pb_val < 4:
                pb_trend = '🟠 Higher P/B'
            else:
                pb_trend = '🔴 High P/B Premium'
        else:
            pb_trend = '⚪ P/B Not Available'

        analysis['valuation_analysis'] = {
            'P/E': pe_val,
            'P/B': pb_val,
            'PEG': round(peg_ratio, 2) if peg_ratio else None,
            'industry_pe': industry_pe,
            'pe_trend': pe_trend,
            'pb_trend': pb_trend
        }

        # ===== ASSET EFFICIENCY =====
        asset_turnover = (curr_rev / total_assets) if total_assets > 0 else 0.0

        # Multi-factor efficiency rating
        if roa > 15 and asset_turnover > 1.0:
            efficiency_rating = '🟢 Exceptional'
        elif roa > 10 or asset_turnover > 0.8:
            efficiency_rating = '🟢 Excellent'
        elif roa > 6 or asset_turnover > 0.6:
            efficiency_rating = '🟡 Good'
        elif roa > 3 or asset_turnover > 0.4:
            efficiency_rating = '🟠 Average'
        else:
            efficiency_rating = '🔴 Poor'

        analysis['asset_efficiency'] = {
            'ROA_%': round(roa, 2),
            'asset_turnover': round(asset_turnover, 2),
            'efficiency_rating': efficiency_rating
        }

        # ===== COMPREHENSIVE RISK ANALYSIS =====
        risk_factors = []
        risk_scores = []  # Numerical scoring for risk level
        
        # Leverage risk
        if debt_to_equity > 1.5:
            risk_factors.append('Excessive leverage (D/E > 1.5)')
            risk_scores.append(3)
        elif debt_to_equity > 1.0:
            risk_factors.append('High leverage (D/E > 1.0)')
            risk_scores.append(2)
        elif debt_to_equity > 0.6:
            risk_factors.append('Moderate leverage')
            risk_scores.append(1)

        # Liquidity risk
        if current_ratio < 1.0:
            risk_factors.append('Critical liquidity risk (CR < 1.0)')
            risk_scores.append(3)
        elif current_ratio < 1.2:
            risk_factors.append('Liquidity concerns')
            risk_scores.append(2)

        # Revenue risk
        if rev_growth < -15:
            risk_factors.append('Severe revenue decline (>15%)')
            risk_scores.append(3)
        elif rev_growth < -5:
            risk_factors.append('Declining revenue')
            risk_scores.append(2)

        # Profitability risk
        if net_margin < 0:
            risk_factors.append('Unprofitable operations')
            risk_scores.append(3)
        elif net_margin < 3:
            risk_factors.append('Low profit margins (<3%)')
            risk_scores.append(1)

        # Cash flow risk
        if fcf < 0 and ocf < 0:
            risk_factors.append('Negative cash flow (OCF & FCF)')
            risk_scores.append(3)
        elif fcf < 0:
            risk_factors.append('Negative free cash flow')
            risk_scores.append(2)

        # Interest coverage risk
        if interest_coverage and interest_coverage < 2:
            risk_factors.append('Low interest coverage (<2x)')
            risk_scores.append(2)

        # Calculate overall risk level
        if not risk_scores:
            risk_level = 'Low'
        else:
            avg_risk = sum(risk_scores) / len(risk_scores)
            if avg_risk >= 2.5:
                risk_level = 'Very High'
            elif avg_risk >= 2.0:
                risk_level = 'High'
            elif avg_risk >= 1.5:
                risk_level = 'Medium-High'
            elif avg_risk >= 1.0:
                risk_level = 'Medium'
            else:
                risk_level = 'Low-Medium'

        analysis['risk_analysis'] = {
            'risk_level': risk_level,
            'risk_factors': risk_factors if risk_factors else ['No significant risks identified']
        }

    except Exception as e:
        analysis['health_description'] = f"Analysis error: {str(e)}"
        score = 25

    # ===== FINAL SCORE AND RATING =====
    final_score = max(0, min(100, score))
    analysis['overall_score'] = final_score

    # Enhanced rating system with more granular categories
    if final_score >= 90:
        health_rating = 'Outstanding 🌟⭐⭐'
        health_description = 'Exceptional financial strength across all metrics - best-in-class'
    elif final_score >= 80:
        health_rating = 'Exceptional 🌟⭐'
        health_description = 'Outstanding financial performance with excellent fundamentals'
    elif final_score >= 70:
        health_rating = 'Excellent 🌟'
        health_description = 'Strong financial health with solid growth and stability'
    elif final_score >= 60:
        health_rating = 'Good ✅+'
        health_description = 'Above-average performance with good growth prospects'
    elif final_score >= 50:
        health_rating = 'Good ✅'
        health_description = 'Solid performance with some areas for improvement'
    elif final_score >= 40:
        health_rating = 'Average ⚠️'
        health_description = 'Mixed performance requiring attention to key areas'
    elif final_score >= 30:
        health_rating = 'Below Average 🟠'
        health_description = 'Concerning performance with significant improvement needed'
    elif final_score >= 20:
        health_rating = 'Poor 🔴'
        health_description = 'Weak financial position requiring immediate attention'
    else:
        health_rating = 'Critical 🔴⚠️'
        health_description = 'Severe financial distress - high risk'

    analysis['health_rating'] = health_rating
    analysis['health_description'] = health_description

    # ===== COMPATIBILITY KEYS FOR UI =====
    analysis['income_statement_analysis'] = {
        'revenue_growth': analysis['revenue_analysis']['revenue_growth_%'],
        'profit_margin': analysis['profitability_analysis']['net_margin_%']
    }

    return analysis