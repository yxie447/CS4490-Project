import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Config
MODEL_NAME = "yiyanghkust/finbert-tone"
TRAIN_DATA_DIR = "NA"  
MODEL_SAVE_PATH = "./finbert_na_trained"
LOG_CSV_PATH = "training_log.csv"
BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 12

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone",
    num_labels=2,
    ignore_mismatched_sizes=True 
)

model.to(device)

# Load training data from all companies in NA
texts, labels = [], []
for company in os.listdir(TRAIN_DATA_DIR):
    path = os.path.join(TRAIN_DATA_DIR, company, "financial_train.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if not df.empty:
            texts.extend(df['text'].astype(str).tolist())
            labels.extend(df['label'].tolist())

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.3, random_state=42)

train_data = [{"text": t, "label": l} for t, l in zip(X_train, y_train)]
val_data = [{"text": t, "label": l} for t, l in zip(X_val, y_val)]
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=lambda x: x)


# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

total_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
training_logs = []
best_acc = 0
patience_counter = 0

print("Loading training data...")

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

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            texts = [item['text'] for item in batch]
            labels_tensor = torch.tensor([item['label'] for item in batch]).to(device)

            encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels_tensor.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Log
    training_logs.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_accuracy": val_acc})

    # Early stopping logic
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print("↑ New best model saved.")
    else:
        patience_counter += 1
        print(f"EarlyStopping counter: {patience_counter} / {PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Save training log
log_df = pd.DataFrame(training_logs)
log_df.to_csv(LOG_CSV_PATH, index=False)
print(f"Training log saved to {LOG_CSV_PATH}")
