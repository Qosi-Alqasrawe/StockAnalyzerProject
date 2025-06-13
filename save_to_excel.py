# save_to_excel.py

import pandas as pd
from datetime import datetime
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import os


def save_report(analysis, symbol, download_path):
    """Save analysis results to an Excel file with improved formatting."""
    # Ensure download path exists
    os.makedirs(download_path, exist_ok=True)
    excel_filename = os.path.join(download_path, f"{symbol}_Final_Analysis.xlsx")

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 1) Summary sheet
        summary = pd.DataFrame({
            'Parameter': [
                'Final Decision', 'Confidence Level', 'Current Price',
                'Support Level', 'Resistance Level', 'Entry Point',
                'Exit Point', 'Stop Loss', 'Shares to Buy', 'Total Invested', 'Remaining Cash'
            ],
            'Value': [
                analysis.get('decision', ''),
                f"{analysis.get('confidence', 0):.1f}%",  # Confidence
                f"${analysis.get('current_price', 0):.2f}",
                f"${analysis.get('support_level', 0):.2f}",
                f"${analysis.get('resistance_level', 0):.2f}",
                f"${analysis.get('entry_point', 0):.2f}",
                f"${analysis.get('exit_point', 0):.2f}",
                f"${analysis.get('stop_loss', 0):.2f}",
                analysis.get('shares_can_buy', 0),
                f"${analysis.get('total_invested', 0):.2f}",
                f"${analysis.get('remaining_cash', 0):.2f}"
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

        # 2) SWOT Analysis sheet
        swot = analysis.get('swot', {})
        swot_rows = []
        for category in ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']:
            details = swot.get(category, [])
            swot_rows.append({'Category': category, 'Details': '; '.join(details) if details else ''})
        swot_df = pd.DataFrame(swot_rows)
        swot_df.to_excel(writer, sheet_name='SWOT', index=False)

        # 3) Price Targets sheet
        price_targets_df = analysis.get('price_targets_df', pd.DataFrame())
        if not price_targets_df.empty:
            price_targets_df.to_excel(writer, sheet_name='Price Targets', index=False)

        # 4) Technical Data sheet (يشمل التاريخ وسعر الإغلاق وباقي البيانات والمؤشرات)
        technical_df = analysis.get('technical_data', pd.DataFrame())

        # —— نظّف كامل الأعمدة من أي NaN قبل التصدير ——
        technical_df = technical_df.dropna(how='any').reset_index(drop=True)

        if not technical_df.empty:
            cols_to_export = [
                # بيانات التاريخ والسعر
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                # مؤشرات الحساب
                'SMA_20', 'SMA_50', 'RSI_7', 'RSI_14', 'RSI_21',
                'MACD', 'MACD_Signal', 'MACD_Histogram', 'ADX', 'Bollinger_%B',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'Stoch_K', 'Stoch_D', 'Stoch_Signal',
                'OBV', 'EMA_signal', 'Pivot', 'R1', 'S1', 'R2', 'S2',
                'Fib_23.6', 'Fib_38.2', 'Fib_50', 'Fib_61.8', 'Fib_78.6',
                'sr_zones', 'Long_Resistance', 'Long_Support', 'Plus_DI', 'Minus_DI',
                'ADX_Signal', 'MACD_Trade_Signal', 'OBV_Signal', 'BB_Signal',
                # أعمدة النتيجة
                'trend_buy_score', 'trend_sell_score',
                'momentum_buy_score', 'momentum_sell_score',
                'volume_buy_score', 'volume_sell_score',
                'strength_buy_score', 'strength_sell_score',
                'sr_buy_score', 'sr_sell_score',
                'Important_Buy_Score', 'Important_Sell_Score', 'Important_Net_Score', 'Important_Signal'
            ]
            cols_to_export = [c for c in cols_to_export if c in technical_df.columns]
            technical_df[cols_to_export].to_excel(
                writer,
                sheet_name='Technical Data',
                index=False
            )


        # 5) Fundamental Info sheet
        basic_info = analysis.get('fundamental_info', {})
        if basic_info:
            basic_info_df = pd.DataFrame(list(basic_info.items()), columns=['Metric', 'Value'])
            basic_info_df.to_excel(writer, sheet_name='Fundamental Info', index=False)

        # 6) Fibonacci & Pivot sheets
        fib = analysis.get('fib_levels', {})
        pivot = analysis.get('pivot_levels', {})
        if fib:
            fib_df = pd.DataFrame(list(fib.items()), columns=['Fibonacci Level', 'Price'])
            fib_df.to_excel(writer, sheet_name='Fibonacci Levels', index=False)
        if pivot:
            pivot_df = pd.DataFrame(list(pivot.items()), columns=['Pivot Level', 'Price'])
            pivot_df.to_excel(writer, sheet_name='Pivot Points', index=False)

        # 7) Financial Statements sheets
        (financials, balance_sheet, cashflow,
         quarterly_financials, quarterly_balance_sheet,
         quarterly_cashflow, earnings, quarterly_earnings) = analysis.get('fundamental_data_full', [pd.DataFrame()]*8)

        def _write_and_format_df(df, sheet_name):
            if df is None or df.empty:
                return
            # Convert datetime columns to string
            df.columns = df.columns.map(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, pd.Timestamp)) else x)
            df.to_excel(writer, sheet_name=sheet_name, index=True)

        _write_and_format_df(financials, 'Income Statement')
        _write_and_format_df(balance_sheet, 'Balance Sheet')
        _write_and_format_df(cashflow, 'Cash Flow')
        _write_and_format_df(quarterly_financials, 'Quarterly Income')
        _write_and_format_df(quarterly_balance_sheet, 'Quarterly Balance')
        _write_and_format_df(quarterly_cashflow, 'Quarterly CashFlow')
        _write_and_format_df(earnings, 'Earnings')
        _write_and_format_df(quarterly_earnings, 'Quarterly Earnings')

        # 8) Financial Analysis Summary sheet
        fin = analysis.get('financial_analysis', {})
        if fin:
            # High-level summary
            financial_summary = []
            financial_summary.append({
                'Metric': 'Financial Health Score',
                'Value': f"{fin.get('overall_score', 0):.1f}",
                'Rating': fin.get('health_rating', '')
            })
            # Subsections: revenue, profitability, balance sheet, cash flow
            rev = fin.get('revenue_analysis', {})
            financial_summary.append({
                'Metric': 'Revenue Growth',
                'Value': f"{rev.get('growth_rate', 0):.1f}%",
                'Rating': rev.get('trend', '')
            })
            prof = fin.get('profitability_analysis', {})
            financial_summary.append({
                'Metric': 'Profit Margin',
                'Value': f"{prof.get('profit_margin', 0):.1f}%",
                'Rating': prof.get('trend', '')
            })
            bs = fin.get('balance_sheet_analysis', {})
            financial_summary.append({
                'Metric': 'Debt-to-Equity',
                'Value': f"{bs.get('debt_to_equity', 0):.2f}",
                'Rating': bs.get('debt_trend', '')
            })
            financial_summary.append({
                'Metric': 'Current Ratio',
                'Value': f"{bs.get('current_ratio', 0):.2f}",
                'Rating': bs.get('liquidity_trend', '')
            })
            cf = fin.get('cash_flow_analysis', {})
            financial_summary.append({
                'Metric': 'Cash Flow Growth',
                'Value': f"{cf.get('cf_growth', 0):.1f}%",
                'Rating': cf.get('trend', '')
            })

            fin_summary_df = pd.DataFrame(financial_summary)
            fin_summary_df.to_excel(writer, sheet_name='Financial Analysis', index=False)

        # Apply formatting to all sheets
        workbook = writer.book
        for sheet_name, sheet in writer.sheets.items():
            # Freeze top row and first column for better navigation
            sheet.freeze_panes = sheet['B2']

            # Apply header formatting only to the first row
            header_font = Font(name='Calibri', size=12, bold=True, color='FFFFFF')
            header_fill = PatternFill(start_color='4F81BD', end_color='4F81BD', fill_type='solid')
            thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                                 top=Side(style='thin'), bottom=Side(style='thin'))

            for cell in sheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = thin_border

            # Adjust column widths based on max content length (with a cap)
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter  # Get Excel column letter
                for cell in col:
                    try:
                        value = str(cell.value) if cell.value is not None else ""
                        max_length = max(max_length, len(value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column].width = min(adjusted_width, 50)

    return excel_filename
