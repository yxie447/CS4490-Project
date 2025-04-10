# CS4490-Project

# FinBERT Cross-Region Financial Sentiment Analysis

This project investigates the cross-regional generalizability of **FinBERT**, a pre-trained financial language model. We trained FinBERT on North American (NA) companies’ financial statement texts and evaluated its performance on companies from **Europe** and **Asia**, analyzing domain shift effects and testing the impact of lightweight fine-tuning.

---

## Project Structure
Project/ │ ├── NA/ # North American company data (excluded from GitHub) ├── Europe/ # European company data (excluded) ├── Asia/ # Asian company data (excluded) │ ├── data_clean.py # Cleans Income Statement and Balance Sheet files ├── generate_text.py # Generates textual financial summaries ├── generate_training_data.py # Creates labeled training samples │ ├── FinBERT_finetune_all.py # Train FinBERT on NA data ├── finetune_asia.py # Fine-tune model on Asia data ├── finetune_europe.py # Fine-tune model on Europe data ├── evaluate_cross_region.py # Evaluate NA-trained model on Europe/Asia ├── evaluate_asia_finetuned.py # Evaluate fine-tuned Asia model ├── evaluate_europe_finetuned.py # Evaluate fine-tuned Europe model │ ├── training_log.csv # Model training and validation history ├── README.md # This file

## Data Access

Raw data files (e.g., `.xls`, `.csv`) are excluded from GitHub due to size limits.
**Download full data ZIP (NA, Europe, Asia):**  
[Google Drive Link](https://drive.google.com/file/d/1RP3lX-zZFYUcnOcy0OoD0x7Qc8G7KGWu/view?usp=drive_link)

## How to Run

### 1. Install dependencies

Install the required Python libraries using pip:
pip install transformers pandas scikit-learn torch

### 2. Clean and prepare data

Before training, run the following scripts to clean the Excel data and generate text-based training samples.

Clean raw financial statements：
python data_clean.py
This script cleans all IncomeStatement.xls, BalanceSheet.xls, and stockprice.xls files across the NA, Europe, and Asia folders.

Generate financial report summaries：
python generate_text.py
This script produces quarterly textual summaries with core financial metrics and financial ratios.

Generate training data：
python generate_training_data.py
This will create financial_train.csv files by assigning binary labels based on stock price trends after each quarterly report.

### 3. Train and evaluate FinBERT on North American data

Train a binary classifier using FinBERT and NA company data:
python FinBERT_finetune_all.py
Evaluate how well the NA-trained model performs on other regions without fine-tuning:
python evaluate_cross_region.py

### 4. Fine-tune and evaluate on other regions

Fine-tune the base FinBERT model using European and Asian datasets respectively:
python finetune_europe.py
python finetune_asia.py

Evaluate the performance of the region-specific fine-tuned models:
python evaluate_europe_finetuned.py
python evaluate_asia_finetuned.py

