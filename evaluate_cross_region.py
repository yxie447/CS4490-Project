import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./finbert_na_trained"
REGIONS = ["Europe", "Asia"]

# Load the model and tokenizer
print("Loading model from", MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def predict_texts(texts, batch_size=8):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# Traverse each area
for region in REGIONS:
    print(f"\nEvaluating region: {region}")
    total_preds = []
    total_labels = []
    
    for company in os.listdir(region):
        company_path = os.path.join(region, company)
        train_file = os.path.join(company_path, "financial_train.csv")
        
        if not os.path.isfile(train_file):
            print(f"Skipping {company} (no financial_train.csv)")
            continue

        df = pd.read_csv(train_file)
        if df.empty:
            continue
        texts = df["text"].astype(str).tolist()
        labels = df["label"].tolist()

        preds = predict_texts(texts)

        total_preds.extend(preds)
        total_labels.extend(labels)

        acc = accuracy_score(labels, preds)
        print(f"{company}: Accuracy = {acc:.4f} ({len(labels)} samples)")

    # Regional Summary
    if total_labels:
        overall_acc = accuracy_score(total_labels, total_preds)
        print(f"\nRegion Summary: {region}")
        print(f"   Accuracy: {overall_acc:.4f}")
        print(f"   Total Samples: {len(total_labels)}")
        print(f"   Correct: {sum([p == l for p, l in zip(total_preds, total_labels)])}")
    else:
        print(f"No data available for region {region}")
