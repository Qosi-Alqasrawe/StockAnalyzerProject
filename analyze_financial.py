import pandas as pd
import numpy as np
import warnings
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

def safe_get_value(df, key, column_idx=0, default=0.0):
    """
    استخراج آمن للقيم من DataFrame مع خيارات احتياطية متعددة
    """
    if df.empty or len(df.columns) <= column_idx:
        return default
        
    # جرب المطابقة الدقيقة أولاً
    if key in df.index:
        try:
            value = df.loc[key].iloc[column_idx]
            return float(value) if pd.notna(value) else default
        except:
            return default
    
    # جرب المطابقة الجزئية للاختلافات الشائعة
    key_variations = {
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
    
    if key in key_variations:
        for variation in key_variations[key]:
            if variation in df.index:
                try:
                    value = df.loc[variation].iloc[column_idx]
                    return float(value) if pd.notna(value) else default
                except:
                    continue
    
    return default

def get_cash_flow_data(cash_flow, income_stmt, basic_info):
    """
    استخراج بيانات التدفق النقدي مع خيارات احتياطية متعددة
    """
    # Operating Cash Flow
    ocf = safe_get_value(cash_flow, 'Operating Cash Flow', 0, 0.0)
    
    # إذا لم نجد OCF، جرب من basic_info
    if ocf == 0.0 and basic_info.get('operatingCashflow'):
        try:
            ocf = float(basic_info['operatingCashflow'])
        except:
            pass
    
    # Capital Expenditure - تأكد من أنه رقم سالب أو موجب
    capex = safe_get_value(cash_flow, 'Capital Expenditure', 0, 0.0)
    
    # CapEx عادة يكون سالب في Yahoo Finance، لكن نحتاج القيمة المطلقة للحساب
    capex = abs(capex)
    
    # إذا لم نجد CapEx من cash flow، جرب من basic_info
    if capex == 0.0:
        # جرب الحصول على CapEx من مصادر أخرى
        capex_keys = ['capitalExpenditures', 'capex']
        for key in capex_keys:
            if basic_info.get(key):
                try:
                    capex = abs(float(basic_info[key]))
                    break
                except:
                    continue
    
    # Free Cash Flow
    fcf = ocf - capex
    
    # جرب الحصول على FCF مباشرة من basic_info إذا كان متاح
    if basic_info.get('freeCashflow'):
        try:
            fcf_direct = float(basic_info['freeCashflow'])
            if fcf_direct != 0:
                fcf = fcf_direct
        except:
            pass
    
    return ocf, capex, fcf

def get_balance_sheet_ratios(balance_sheet, basic_info):
    """
    استخراج نسب الميزانية العمومية مع خيارات احتياطية
    """
    # Total Debt
    total_debt = safe_get_value(balance_sheet, 'Total Debt', 0, 0.0)
    
    # إذا لم نجد Total Debt، اجمع Long Term + Short Term
    if total_debt == 0.0:
        long_term_debt = safe_get_value(balance_sheet, 'Long Term Debt', 0, 0.0)
        short_term_debt = safe_get_value(balance_sheet, 'Current Debt', 0, 0.0)
        total_debt = long_term_debt + short_term_debt
    
    # جرب من basic_info
    if total_debt == 0.0 and basic_info.get('totalDebt'):
        try:
            total_debt = float(basic_info['totalDebt'])
        except:
            pass
    
    # Total Assets
    total_assets = safe_get_value(balance_sheet, 'Total Assets', 0, 0.0)
    
    # جرب من basic_info
    if total_assets == 0.0 and basic_info.get('totalAssets'):
        try:
            total_assets = float(basic_info['totalAssets'])
        except:
            pass
    
    # Debt to Assets Ratio
    debt_to_assets = (total_debt / total_assets) if total_assets > 0 else 0.0
    
    return total_debt, total_assets, debt_to_assets

def analyze_financial_performance(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    quarterly_income: pd.DataFrame,
    quarterly_balance: pd.DataFrame,
    quarterly_cash: pd.DataFrame,
    basic_info: dict,
    industry_pe: float = None,
    debug: bool = False
) -> dict:
    """
    Enhanced financial analysis with better data extraction and debugging
    """
    if debug:
        debug_dataframes(income_stmt, balance_sheet, cash_flow, basic_info.get('symbol', ''))
    
    analysis = {
        'revenue_analysis': {},
        'profitability_analysis': {},
        'balance_sheet_analysis': {},
        'cash_flow_analysis': {},
        'valuation_analysis': {},
        'asset_efficiency': {},
        'risk_analysis': {},
        'overall_score': 0,
        'health_rating': '',
        'health_description': '',
        'recommendations': []
    }
    score = 50
    recommendations = []

    try:
        # ===== Revenue Analysis =====
        curr_rev = safe_get_value(income_stmt, 'Total Revenue', 0, 0.0)
        prev_rev = safe_get_value(income_stmt, 'Total Revenue', 1, curr_rev)
        
        # Calculate revenue growth
        rev_growth = 0.0
        if prev_rev > 0 and curr_rev != prev_rev:
            rev_growth = ((curr_rev - prev_rev) / abs(prev_rev)) * 100
        
        # Net Income Analysis
        curr_net = safe_get_value(income_stmt, 'Net Income', 0, 0.0)
        prev_net = safe_get_value(income_stmt, 'Net Income', 1, curr_net)
        
        net_growth = 0.0
        if prev_net != 0 and curr_net != prev_net:
            net_growth = ((curr_net - prev_net) / abs(prev_net)) * 100

        # Revenue trend classification
        if rev_growth > 15:
            score += 20
            rev_trend = '🚀 Excellent Growth'
        elif rev_growth > 7:
            score += 10
            rev_trend = '➡️ Solid Growth'
        elif rev_growth > 0:
            score += 5
            rev_trend = '📊 Modest Growth'
        elif rev_growth > -10:
            score -= 5
            rev_trend = '📉 Declining Revenue'
        else:
            score -= 15
            rev_trend = '⚠️ Severe Decline'

        analysis['revenue_analysis'] = {
            'current_revenue': curr_rev,
            'revenue_growth_%': round(rev_growth, 2),
            'trend': rev_trend,
            'net_income_growth_%': round(net_growth, 2)
        }

        # ===== Profitability Analysis =====
        gross_profit = safe_get_value(income_stmt, 'Gross Profit', 0, 0.0)
        gross_margin = (gross_profit / curr_rev * 100) if curr_rev > 0 else 0.0

        operating_income = safe_get_value(income_stmt, 'Operating Income', 0, 0.0)
        operating_margin = (operating_income / curr_rev * 100) if curr_rev > 0 else 0.0

        net_margin = (curr_net / curr_rev * 100) if curr_rev > 0 else 0.0

        # ROE calculation with multiple fallbacks
        roe = 0.0
        if basic_info.get('returnOnEquity'):
            try:
                roe_val = basic_info['returnOnEquity']
                if isinstance(roe_val, (int, float)):
                    roe = roe_val * 100 if abs(roe_val) <= 1 else roe_val
            except:
                pass
        
        if roe == 0.0:  # Fallback calculation
            equity_val = safe_get_value(balance_sheet, 'Stockholders Equity', 0, 0.0)
            if curr_net != 0 and equity_val > 0:
                roe = (curr_net / equity_val) * 100

        # ROA calculation
        roa = 0.0
        if basic_info.get('returnOnAssets'):
            try:
                roa_val = basic_info['returnOnAssets']
                if isinstance(roa_val, (int, float)):
                    roa = roa_val * 100 if abs(roa_val) <= 1 else roa_val
            except:
                pass

        # Profitability scoring
        margin_score = 0
        if gross_margin > 40:
            margin_score += 10
        elif gross_margin > 25:
            margin_score += 5

        if operating_margin > 15:
            margin_score += 10
        elif operating_margin > 8:
            margin_score += 5

        if net_margin > 15:
            margin_score += 15
        elif net_margin > 8:
            margin_score += 10
        elif net_margin > 3:
            margin_score += 5

        if roe > 20:
            margin_score += 15
        elif roe > 15:
            margin_score += 10
        elif roe > 10:
            margin_score += 5

        score += margin_score

        analysis['profitability_analysis'] = {
            'gross_margin_%': round(gross_margin, 2),
            'operating_margin_%': round(operating_margin, 2),
            'net_margin_%': round(net_margin, 2),
            'ROE_%': round(roe, 2),
            'ROA_%': round(roa, 2)
        }

        # ===== Balance Sheet Analysis =====
        total_debt, total_assets, debt_to_assets = get_balance_sheet_ratios(balance_sheet, basic_info)
        
        equity_val = safe_get_value(balance_sheet, 'Stockholders Equity', 0, 0.0)
        curr_assets = safe_get_value(balance_sheet, 'Current Assets', 0, 0.0)
        curr_liab = safe_get_value(balance_sheet, 'Current Liabilities', 0, 0.0)

        # Calculate ratios with fallbacks to basic_info
        debt_to_equity = 0.0
        if basic_info.get('debtToEquity'):
            try:
                debt_to_equity = float(basic_info['debtToEquity']) / 100
            except:
                pass
        if debt_to_equity == 0.0 and equity_val > 0:
            debt_to_equity = total_debt / equity_val

        current_ratio = 0.0
        if basic_info.get('currentRatio'):
            try:
                current_ratio = float(basic_info['currentRatio'])
            except:
                pass
        if current_ratio == 0.0 and curr_liab > 0:
            current_ratio = curr_assets / curr_liab

        # Quick ratio calculation
        inventory = safe_get_value(balance_sheet, 'Inventory', 0, 0.0)
        quick_assets = curr_assets - inventory
        quick_ratio = (quick_assets / curr_liab) if curr_liab > 0 else 0.0

        # Balance sheet scoring
        balance_score = 0
        if debt_to_equity < 0.3:
            balance_score += 15
            debt_trend = '🟢 Conservative Leverage'
        elif debt_to_equity < 0.6:
            balance_score += 10
            debt_trend = '🟡 Moderate Leverage'
        elif debt_to_equity < 1.0:
            balance_score += 5
            debt_trend = '🟠 Higher Leverage'
        else:
            balance_score -= 10
            debt_trend = '🔴 High Leverage Risk'

        if current_ratio > 2.5:
            balance_score += 10
            liq_trend = '🟢 Excellent Liquidity'
        elif current_ratio > 1.5:
            balance_score += 8
            liq_trend = '🟡 Good Liquidity'
        elif current_ratio > 1.0:
            balance_score += 5
            liq_trend = '🟠 Adequate Liquidity'
        else:
            balance_score -= 10
            liq_trend = '🔴 Liquidity Concern'

        score += balance_score

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

        # ===== Cash Flow Analysis =====
        ocf, capex, fcf = get_cash_flow_data(cash_flow, income_stmt, basic_info)
        
        ocf_to_revenue = (ocf / curr_rev * 100) if curr_rev > 0 else 0.0
        fcf_to_revenue = (fcf / curr_rev * 100) if curr_rev > 0 else 0.0

        # Interest coverage calculation
        ebit = safe_get_value(income_stmt, 'EBIT', 0, operating_income)
        interest_expense = abs(safe_get_value(income_stmt, 'Interest Expense', 0, 0.0))
        interest_coverage = (ebit / interest_expense) if interest_expense > 0 else None

        # Shares change calculation
        shares_current = safe_get_value(balance_sheet, 'Common Stock', 0, 0.0)
        shares_previous = safe_get_value(balance_sheet, 'Common Stock', 1, shares_current)
        shares_change = ((shares_current - shares_previous) / shares_previous * 100) if shares_previous > 0 else None

        # Cash flow scoring
        cash_score = 0
        if ocf > 0 and fcf > 0:
            cash_score += 15
            fcf_trend = '🟢 Strong Cash Generation'
        elif ocf > 0:
            cash_score += 10
            fcf_trend = '🟡 Positive Operating Cash Flow'
        elif fcf > 0:
            cash_score += 5
            fcf_trend = '🟠 Positive Free Cash Flow'
        else:
            cash_score -= 15
            fcf_trend = '🔴 Cash Flow Concerns'

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

        # ===== Valuation Analysis =====
        pe_val = None
        pb_val = None
        
        # Try multiple PE sources
        for pe_key in ['trailingPE', 'forwardPE', 'P/E TTM', 'P/E Ratio (TTM)']:
            if basic_info.get(pe_key) and basic_info[pe_key] not in (None, 'N/A', ''):
                try:
                    pe_val = float(basic_info[pe_key])
                    break
                except:
                    continue

        # Try P/B ratio
        if basic_info.get('priceToBook'):
            try:
                pb_val = float(basic_info['priceToBook'])
            except:
                pass

        # PEG ratio calculation
        peg_ratio = None
        if pe_val and rev_growth > 0:
            peg_ratio = pe_val / rev_growth

        # Valuation trends
        pe_trend = '⚪ No Comparison Available'
        pb_trend = '⚪ P/B Not Available'
        
        if pe_val:
            if pe_val < 15:
                pe_trend = '🟢 Attractive Valuation'
            elif pe_val < 25:
                pe_trend = '🟡 Fair Valuation'
            elif pe_val < 35:
                pe_trend = '🟠 Higher Valuation'
            else:
                pe_trend = '🔴 Expensive'

        if pb_val:
            if pb_val < 1:
                pb_trend = '🟢 Book Value Discount'
            elif pb_val < 2:
                pb_trend = '🟡 Reasonable P/B'
            elif pb_val < 4:
                pb_trend = '🟠 Higher P/B'
            else:
                pb_trend = '🔴 High P/B Premium'

        analysis['valuation_analysis'] = {
            'P/E': pe_val,
            'P/B': pb_val,
            'PEG': round(peg_ratio, 2) if peg_ratio else None,
            'industry_pe': industry_pe,
            'pe_trend': pe_trend,
            'pb_trend': pb_trend
        }

        # ===== Asset Efficiency =====
        asset_turnover = (curr_rev / total_assets) if total_assets > 0 else 0.0

        analysis['asset_efficiency'] = {
            'ROA_%': round(roa, 2),
            'asset_turnover': round(asset_turnover, 2),
            'efficiency_rating': (
                '🟢 Excellent' if roa > 15 else
                '🟡 Good' if roa > 8 else
                '🟠 Average' if roa > 4 else
                '🔴 Poor'
            )
        }

        # ===== Risk Analysis =====
        risk_factors = []
        risk_level = 'Low'

        if debt_to_equity > 1.0:
            risk_factors.append('High leverage')
            risk_level = 'High'
        elif debt_to_equity > 0.6:
            risk_factors.append('Moderate leverage')
            risk_level = 'Medium'

        if current_ratio < 1.2:
            risk_factors.append('Liquidity concerns')
            risk_level = 'High'

        if rev_growth < -10:
            risk_factors.append('Declining revenue')
            risk_level = 'High'

        if fcf < 0:
            risk_factors.append('Negative free cash flow')
            if risk_level == 'Low':
                risk_level = 'Medium'

        analysis['risk_analysis'] = {
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }

    except Exception as e:
        analysis['health_description'] = f"Analysis error: {str(e)}"
        score = 25

    # ===== Final Score and Rating =====
    final_score = max(0, min(100, score))
    analysis['overall_score'] = final_score

    if final_score >= 85:
        analysis['health_rating'] = 'Exceptional 🌟⭐'
        analysis['health_description'] = 'Outstanding financial performance across all metrics'
    elif final_score >= 70:
        analysis['health_rating'] = 'Excellent 🌟'
        analysis['health_description'] = 'Strong financial health with minor areas for optimization'
    elif final_score >= 55:
        analysis['health_rating'] = 'Good ✅'
        analysis['health_description'] = 'Solid performance with some improvement opportunities'
    elif final_score >= 40:
        analysis['health_rating'] = 'Average ⚠️'
        analysis['health_description'] = 'Mixed performance requiring attention to key areas'
    elif final_score >= 25:
        analysis['health_rating'] = 'Below Average 🟠'
        analysis['health_description'] = 'Concerning performance with significant improvement needed'
    else:
        analysis['health_rating'] = 'Poor 🔴'
        analysis['health_description'] = 'Weak financial position requiring immediate action'

    # ===== Add compatibility keys for UI =====
    analysis['income_statement_analysis'] = {
        'revenue_growth': analysis['revenue_analysis']['revenue_growth_%'],
        'profit_margin': analysis['profitability_analysis']['net_margin_%']
    }

    return analysis