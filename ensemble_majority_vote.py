import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mode
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Config ===
BERT_DIRS = [
    "/home/kej48/rds/hpc-work/Thesis/bert_results/checkpoint-396",
    "/home/kej48/rds/hpc-work/Thesis/bert_results/checkpoint-418",
    "/home/kej48/rds/hpc-work/Thesis/bert_results/checkpoint-440",
]

ROBERTA_DIRS = [
    "/home/kej48/rds/hpc-work/Thesis/roberta_results/checkpoint-396",
    "/home/kej48/rds/hpc-work/Thesis/roberta_results/checkpoint-418",
    "/home/kej48/rds/hpc-work/Thesis/roberta_results/checkpoint-440",
]

TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/data/mfa_test_data/transcripts"
LABELS_PATH = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"

# === Load Test Metadata ===
df_labels = pd.read_csv(LABELS_PATH, sep=";")
df_labels.columns = df_labels.columns.str.strip().str.lower()
df_labels["id"] = df_labels["id"].str.strip().str.upper()
df_labels["label"] = df_labels["label"].astype(int)

# === Load Test Examples ===
examples = []
for fname in os.listdir(TRANSCRIPT_DIR):
    if fname.endswith(".txt"):
        file_id = os.path.splitext(fname)[0].upper()
        row = df_labels[df_labels["id"] == file_id]
        if row.empty:
            continue
        with open(os.path.join(TRANSCRIPT_DIR, fname), "r") as f:
            text = f.read().strip()
        if not text:
            continue
        examples.append({
            "id": file_id,
            "text": text,
            "label": int(row["label"].values[0])
        })

texts = [e["text"] for e in examples]
labels = [e["label"] for e in examples]
print(f"Loaded {len(texts)} test examples.")

# === Ensemble Inference ===
def run_model_predictions(model_dirs, model_type="bert"):
    all_logits = []
    for i, dir_path in enumerate(model_dirs):
        print(f"Loading {model_type} model from {dir_path}")
        tokenizer = AutoTokenizer.from_pretrained(dir_path)
        model = AutoModelForSequenceClassification.from_pretrained(dir_path)
        model.to("cuda")
        model.eval()

        logits_list = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                batch = texts[i:i+8]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                logits = model(**inputs).logits.cpu().numpy()
                logits_list.append(logits)
        model_logits = np.vstack(logits_list)
        all_logits.append(model_logits)
    return all_logits

bert_logits = run_model_predictions(BERT_DIRS, "bert")
roberta_logits = run_model_predictions(ROBERTA_DIRS, "roberta")

# === Majority Voting ===
all_model_logits = bert_logits + roberta_logits
all_model_preds = [np.argmax(logits, axis=-1) for logits in all_model_logits]
voted_preds, _ = mode(np.stack(all_model_preds), axis=0)
final_preds = voted_preds.ravel()

# === Evaluation ===
print("\nEvaluating Ensemble on True Test Set...")
acc = accuracy_score(labels, final_preds)
cm = confusion_matrix(labels, final_preds)
report = classification_report(labels, final_preds, target_names=["Control (0)", "AD (1)"])
tn, fp, fn, tp = cm.ravel()

print(f"\nEnsemble Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
print(f"\nType I Errors (False Positives): {fp}")
print(f"Type II Errors (False Negatives): {fn}")
