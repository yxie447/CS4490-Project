# evaluate_asia_finetuned.py

import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

ASIA_FOLDER = "Asia"
MODEL_PATH = "./finbert_asia_finetuned"

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def evaluate_company(company_folder):
    path = os.path.join(company_folder, "financial_train.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    correct = 0
    total = 0

    for _, row in df.iterrows():
        text = row["text"]
        label = int(row["label"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        correct += int(pred == label)
        total += 1

    return correct, total

total_correct = 0
total_samples = 0
print("\nEvaluating region: Asia (Fine-tuned Model)")

for company in os.listdir(ASIA_FOLDER):
    company_folder = os.path.join(ASIA_FOLDER, company)
    if os.path.isdir(company_folder):
        result = evaluate_company(company_folder)
        if result:
            correct, total = result
            acc = correct / total if total else 0
            print(f"{company}: Accuracy = {acc:.4f} ({total} samples)")
            total_correct += correct
            total_samples += total

region_acc = total_correct / total_samples if total_samples else 0
print(f"\nRegion Summary: Asia (Fine-tuned)\n   Accuracy: {region_acc:.4f}\n   Total Samples: {total_samples}\n   Correct: {total_correct}")
