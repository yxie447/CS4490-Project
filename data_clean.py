import os
import re
import pandas as pd

def adjust_header_to_period_end(df):
    """
    Adjust the DataFrame's header:
    - Replace the first column header with "Line Item"
    - For each subsequent header that is exactly a 4-digit year (e.g., "2024"),
      replace it with "YYYY-12-31".
    """
    new_columns = []
    for i, col in enumerate(df.columns):
        if i == 0:
            # The first column holds line item names
            new_columns.append("Line Item")
        else:
            # Check if the column header is exactly a 4-digit year
            col_str = str(col).strip()
            if re.fullmatch(r'\d{4}', col_str):
                new_columns.append(f"{col_str}-12-31")
            else:
                new_columns.append(col_str)
    df.columns = new_columns
    return df

def clean_excel_file(file_path, header=0, column_rename_dict=None, date_columns=None, numeric_columns=None, drop_keyword=None, adjust_to_year_end=False):
    """
    Generic function to clean an Excel file.
    
    Parameters:
        file_path (str): Path to the Excel file.
        header (int): Row number to use as header.
        column_rename_dict (dict): Dictionary mapping existing column names to new names.
        date_columns (list): List of column names to convert to datetime.
        numeric_columns (list): List of column names to clean and convert to numeric.
        drop_keyword (str): Keyword; any row containing this in any cell will be dropped.
        adjust_to_year_end (bool): If True, adjust the header:
                                   - set the first column to "Line Item"
                                   - change subsequent columns that are 4-digit years to "YYYY-12-31"
                                   
    Returns:
        DataFrame: Cleaned pandas DataFrame.
    """
    # Read the Excel file
    df = pd.read_excel(file_path, header=header)
    
    # If a column rename mapping is provided, apply it first
    if column_rename_dict:
        df.rename(columns=column_rename_dict, inplace=True)
    
    # Adjust header if required
    if adjust_to_year_end:
        df = adjust_header_to_period_end(df)
    
    # Convert specified date columns to datetime format
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df.dropna(subset=[col], inplace=True)
    
    # Clean and convert specified numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                # Remove common currency symbols and commas
                df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows containing the drop_keyword in any cell
    if drop_keyword:
        df = df[~df.astype(str).apply(lambda x: x.str.contains(drop_keyword, na=False)).any(axis=1)]
    
    return df

# -----------------------------
# Main script to clean data for multiple companies
# -----------------------------

# Path to the main folder containing subfolders for each company
REGIONS = ["NA", "Europe", "Asia"]


# Column renaming dictionaries for Income Statement and Balance Sheet files (if needed)
income_statement_rename = { "Unnamed: 0": "(In Thousands) (USD)" }
balance_sheet_rename = { "Unnamed: 0": "(In Thousands) (USD)" }
stock_price_rename = {
    "Unnamed: 0": "Date",
    "Unnamed: 1": "Open",
    "Unnamed: 2": "High",
    "Unnamed: 3": "Low",
    "Unnamed: 4": "Close",
    "Unnamed: 5": "Volume"
}


# Loop through each company folder and clean the three files
for region in REGIONS:
    print(f"\nProcessing region: {region}")
    region_path = region  # assuming same level as script
    if not os.path.isdir(region_path):
        print(f"Skipping missing region folder: {region}")
        continue

    companies = [name for name in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, name))]

    for company in companies:
        company_path = os.path.join(region_path, company)
        print(f"Company: {company}")

        # 1. Income Statement
        income_path = os.path.join(company_path, "IncomeStatement.xls")
        if os.path.isfile(income_path):
            income_df = clean_excel_file(
                file_path=income_path,
                header=0,
                column_rename_dict=income_statement_rename,
                date_columns=[],
                numeric_columns=[],
                drop_keyword="Powered by Clearbit",
                adjust_to_year_end=True
            )
            income_df.to_csv(os.path.join(company_path, "IncomeStatement_clean.csv"), index=False)
        else:
            print("Missing Income Statement")

        # 2. Balance Sheet
        balance_path = os.path.join(company_path, "BalanceSheet.xls")
        if os.path.isfile(balance_path):
            balance_df = clean_excel_file(
                file_path=balance_path,
                header=0,
                column_rename_dict=balance_sheet_rename,
                date_columns=[],
                numeric_columns=[],
                drop_keyword="Powered by Clearbit",
                adjust_to_year_end=True
            )
            balance_df.to_csv(os.path.join(company_path, "BalanceSheet_clean.csv"), index=False)
        else:
            print("Missing Balance Sheet")

        # 3. Stock Price
        stock_path = os.path.join(company_path, "stockprice.xls")
        if os.path.isfile(stock_path):
            stock_df = clean_excel_file(
                file_path=stock_path,
                header=0,
                column_rename_dict=stock_price_rename,
                date_columns=["Date"],
                numeric_columns=["Open", "High", "Low", "Close", "Volume"],
                drop_keyword="Powered by Clearbit",
                adjust_to_year_end=False
            )
            stock_df.to_csv(os.path.join(company_path, "stockprice_clean.csv"), index=False)
        else:
            print("Missing Stock Price")

print("\nAll regions processed.")


