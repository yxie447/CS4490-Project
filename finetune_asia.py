# finetune_asia.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configurations
MODEL_PATH = "./finbert_na_trained"  # base model
ASIA_FOLDER = "Asia"
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
OUTPUT_PATH = "./finbert_asia_finetuned"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)

def collate_fn(batch):
    return batch

# Load all Asia data
samples = []
for company in os.listdir(ASIA_FOLDER):
    path = os.path.join(ASIA_FOLDER, company, "financial_train.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            samples.append({"text": row["text"], "label": int(row["label"])})

print(f"Total Asia samples: {len(samples)}")

# Split dataset
train_data, val_data = train_test_split(samples, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)


# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        texts = [item['text'] for item in batch]
        labels_tensor = torch.tensor([item['label'] for item in batch]).to(device)

        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_tensor)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            texts = [item['text'] for item in batch]
            labels_tensor = torch.tensor([item['label'] for item in batch]).to(device)

            encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)
        print("↑ New best model saved.")
