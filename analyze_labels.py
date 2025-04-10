import os
import pandas as pd

NA_FOLDER = "NA"

result_rows = []
total_0 = 0
total_1 = 0
total_all = 0

# Go through each company
for company in os.listdir(NA_FOLDER):
    company_path = os.path.join(NA_FOLDER, company)
    train_file = os.path.join(company_path, "financial_train.csv")
    
    if not os.path.isfile(train_file):
        continue

    df = pd.read_csv(train_file)

    num_0 = (df["label"] == 0).sum()
    num_1 = (df["label"] == 1).sum()
    total = len(df)

    total_0 += num_0
    total_1 += num_1
    total_all += total

    percent_1 = (num_1 / total) * 100 if total else 0

    result_rows.append({
        "Company": company,
        "Total Samples": total,
        "Label 0 (↓)": num_0,
        "Label 1 (↑)": num_1,
        "↑ Percentage": f"{percent_1:.2f}%"
    })

# Output overall situation
overall_percent_1 = (total_1 / total_all) * 100 if total_all else 0
print("===== Overall Summary =====")
print(f"Total Samples: {total_all}")
print(f"Label 0 (↓): {total_0}")
print(f"Label 1 (↑): {total_1}")
print(f"↑ Percentage: {overall_percent_1:.2f}%")

# Save as CSV
df_out = pd.DataFrame(result_rows)
df_out.loc[len(df_out.index)] = {
    "Company": "TOTAL",
    "Total Samples": total_all,
    "Label 0 (↓)": total_0,
    "Label 1 (↑)": total_1,
    "↑ Percentage": f"{overall_percent_1:.2f}%"
}
df_out.to_csv("label_distribution.csv", index=False)
print("Saved label distribution to 'label_distribution.csv'")
