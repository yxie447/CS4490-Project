import os
import pandas as pd

def compute_label(report_date, stock_df, look_ahead=5):
    report_date = pd.to_datetime(report_date)
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)
    idx = stock_df[stock_df["Date"] >= report_date].index.min()
    if pd.isna(idx):
        return 0
    report_close = stock_df.loc[idx, "Close"]
    target_idx = idx + look_ahead
    if target_idx >= len(stock_df):
        target_idx = len(stock_df) - 1
    target_close = stock_df.loc[target_idx, "Close"]
    return 1 if target_close > report_close else 0

def generate_training_data(quarterly_df, stock_df, look_ahead=5):
    training_records = []
    for _, row in quarterly_df.iterrows():
        report_date = row["Period End"]
        text = row["Text Description"]
        label = compute_label(report_date, stock_df, look_ahead=look_ahead)
        training_records.append({"text": text, "label": label})
    return pd.DataFrame(training_records)

def process_company_folder(company_folder, look_ahead=5):
    quarterly_path = os.path.join(company_folder, "Quarterly_Report_Texts.csv")
    stock_path = os.path.join(company_folder, "stockprice_clean.csv")

    if not (os.path.isfile(quarterly_path) and os.path.isfile(stock_path)):
        print(f"Missing files in {company_folder}. Skipping...")
        return

    quarterly_df = pd.read_csv(quarterly_path)
    quarterly_df["Period End"] = pd.to_datetime(quarterly_df["Period End"], errors="coerce")
    stock_df = pd.read_csv(stock_path)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")

    train_df = generate_training_data(quarterly_df, stock_df, look_ahead=look_ahead)
    output_path = os.path.join(company_folder, "financial_train.csv")
    train_df.to_csv(output_path, index=False)
    print(f"Generated financial_train.csv for {company_folder}")

def process_all_regions(base_folders=["NA", "Europe", "Asia"], look_ahead=5):
    for region in base_folders:
        print(f"\nProcessing region: {region}")
        if not os.path.isdir(region):
            print(f"Skipping missing region: {region}")
            continue
        for company in os.listdir(region):
            company_folder = os.path.join(region, company)
            if os.path.isdir(company_folder):
                print(f"Processing company: {company}")
                process_company_folder(company_folder, look_ahead=look_ahead)

if __name__ == "__main__":
    process_all_regions(["NA", "Europe", "Asia"], look_ahead=5)
