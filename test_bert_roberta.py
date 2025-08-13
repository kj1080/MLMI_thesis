import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Config ---
MODEL_DIR = "/home/kej48/rds/hpc-work/Thesis/roberta_results/checkpoint-242"  # Replace with your BERT or RoBERTa model folder
TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/data/mfa_test_data/transcripts"  # Folder containing .txt transcripts
LABELS_PATH = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"      # Path to test_results.txt with ground-truth labels

# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to("cuda")
model.eval()

# --- Load label metadata ---
df_labels = pd.read_csv(LABELS_PATH, sep=";")
df_labels.columns = df_labels.columns.str.strip().str.lower()
df_labels["id"] = df_labels["id"].str.strip().str.upper()
df_labels["label"] = df_labels["label"].astype(int)

# --- Load and prepare transcripts ---
examples = []
for fname in os.listdir(TRANSCRIPT_DIR):
    if fname.endswith(".txt"):
        file_id = os.path.splitext(fname)[0].upper()
        label_row = df_labels[df_labels["id"] == file_id]
        if label_row.empty:
            print(f"Skipping unknown file: {file_id}")
            continue
        with open(os.path.join(TRANSCRIPT_DIR, fname), "r") as f:
            text = f.read().strip()
        if not text:
            continue
        examples.append({
            "id": file_id,
            "text": text,
            "label": int(label_row["label"].values[0])
        })

print(f"Loaded {len(examples)} transcripts.")

# --- Inference ---
batch_size = 8
all_preds = []
true_labels = []

print("Running inference...")
with torch.no_grad():
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        texts = [ex["text"] for ex in batch]
        labels = [ex["label"] for ex in batch]

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds)
        true_labels.extend(labels)

        print(f"Batch {i // batch_size + 1} processed")

# --- Evaluation ---
print("Evaluating predictions...")
acc = accuracy_score(true_labels, all_preds)
cm = confusion_matrix(true_labels, all_preds)
report = classification_report(true_labels, all_preds, target_names=["Control (0)", "AD (1)"])

tn, fp, fn, tp = cm.ravel()

print(f"\nOverall Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
print(f"\nType I Errors (False Positives): {fp}")
print(f"Type II Errors (False Negatives): {fn}")
