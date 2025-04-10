import os
import pandas as pd

row_map_income = {
    "revenue": "Total Revenue",
    "net_income": "Net Income"
}
row_map_balance = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities"
}

def generate_quarterly_text_wide(income_df, balance_df, multiply_by_1000=False):
    if income_df.index.name != "Line Item":
        income_df = income_df.set_index("Line Item")
    if balance_df.index.name != "Line Item":
        balance_df = balance_df.set_index("Line Item")

    period_columns = [col for col in income_df.columns if col in balance_df.columns]
    period_columns.sort()  # ensure chronological order

    records = []
    prev_revenue = None
    prev_assets = None

    for period in period_columns:
        try:
            revenue = income_df.at[row_map_income["revenue"], period]
        except KeyError:
            revenue = 0

        try:
            net_income = income_df.at[row_map_income["net_income"], period]
        except KeyError:
            net_income = 0

        try:
            total_assets = balance_df.at[row_map_balance["total_assets"], period]
        except KeyError:
            total_assets = 0

        try:
            total_liabilities = balance_df.at[row_map_balance["total_liabilities"], period]
        except KeyError:
            total_liabilities = 0

        revenue = revenue if pd.notnull(revenue) else 0
        net_income = net_income if pd.notnull(net_income) else 0
        total_assets = total_assets if pd.notnull(total_assets) else 0
        total_liabilities = total_liabilities if pd.notnull(total_liabilities) else 0

        if multiply_by_1000:
            revenue *= 1000
            net_income *= 1000
            total_assets *= 1000
            total_liabilities *= 1000

        # Calculate financial ratios
        net_margin = (net_income / revenue * 100) if revenue else 0
        debt_ratio = (total_liabilities / total_assets * 100) if total_assets else 0

        # Generate semantic interpretations
        if net_margin > 15:
            profit_comment = f"a strong net profit margin of {net_margin:.2f}%, indicating healthy profitability"
        elif net_margin > 5:
            profit_comment = f"a moderate profit margin of {net_margin:.2f}%, reflecting stable earnings"
        else:
            profit_comment = f"a low profit margin of {net_margin:.2f}%, indicating weaker profitability"

        if debt_ratio > 70:
            debt_comment = f"liabilities accounted for {debt_ratio:.2f}% of total assets, suggesting a relatively high debt burden"
        elif debt_ratio > 50:
            debt_comment = f"liabilities made up {debt_ratio:.2f}% of assets, reflecting moderate leverage"
        else:
            debt_comment = f"a low debt ratio of {debt_ratio:.2f}%, indicating conservative financing"

        # Optional: growth calculations
        revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else None
        asset_growth = ((total_assets - prev_assets) / prev_assets * 100) if prev_assets else None

        # Assemble text
        text = (
            f"For the period ending on {period}, the company reported a revenue of ${revenue:,.2f} "
            f"and a net income of ${net_income:,.2f}. The balance sheet shows total assets of ${total_assets:,.2f} "
            f"and total liabilities of ${total_liabilities:,.2f}. The company had {profit_comment}. "
            f"Additionally, {debt_comment}."
        )

        if revenue_growth is not None and asset_growth is not None:
            text += (
                f" Compared to the previous period, revenue changed by {revenue_growth:.2f}% "
                f"and total assets changed by {asset_growth:.2f}%."
            )

        records.append({
            "Period End": period,
            "Revenue": revenue,
            "Net Income": net_income,
            "Total Assets": total_assets,
            "Total Liabilities": total_liabilities,
            "Text Description": text
        })

        prev_revenue = revenue
        prev_assets = total_assets

    return pd.DataFrame(records)

def process_company_folder(company_folder, multiply_by_1000=False):
    income_path = os.path.join(company_folder, "IncomeStatement_clean.csv")
    balance_path = os.path.join(company_folder, "BalanceSheet_clean.csv")

    if not (os.path.isfile(income_path) and os.path.isfile(balance_path)):
        print(f"Missing financial data in {company_folder}. Skipping...")
        return

    income_df = pd.read_csv(income_path)
    balance_df = pd.read_csv(balance_path)

    report_df = generate_quarterly_text_wide(income_df, balance_df, multiply_by_1000=multiply_by_1000)

    output_path = os.path.join(company_folder, "Quarterly_Report_Texts.csv")
    report_df.to_csv(output_path, index=False)
    print(f"Saved text descriptions for {company_folder} -> {output_path}")

def process_all_companies(base_folder="NA", multiply_by_1000=False):
    for company in os.listdir(base_folder):
        company_folder = os.path.join(base_folder, company)
        if os.path.isdir(company_folder):
            print(f"Processing company: {company}")
            process_company_folder(company_folder, multiply_by_1000=multiply_by_1000)

if __name__ == "__main__":
    process_all_companies("NA", multiply_by_1000=True)
