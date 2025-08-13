import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import mode
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb
import json

# === Config ===
SEED_RANGE = range(42, 43)
pause_encoding = True

if pause_encoding == False:
    TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/data/mfa_test_data/transcripts"
    
elif pause_encoding == True:
    TRANSCRIPT_DIR= "/home/kej48/rds/hpc-work/Thesis/pause_encoding/test"

LABELS_PATH = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"
BASE_BERT_DIR = "/home/kej48/rds/hpc-work/Thesis/bert_results"
BASE_ROBERTA_DIR = "/home/kej48/rds/hpc-work/Thesis/roberta_results"

# === Mode Options ===
USE_LAST_N = 8 # Use last N epochs or best N checkpoints
USE_BEST = True  # If True, use "best" checkpoints by validation loss (assumes names include metric)
RUN_ALL_SEEDS = True  # If False, will only run for seed=42

# === W&B Init ===
wandb.init(
    entity="kj1080-university-of-cambridge",
    project="mlmi_thesis",
    name="ensemble_eval",
    config={
        "ensemble_type": "last_3_epochs" if not USE_BEST else "best_3_checkpoints",
        "model_types": ["bert", "roberta"],
        "num_seeds": len(SEED_RANGE) if RUN_ALL_SEEDS else 1,
    },
)

# === Load Labels and Test Texts ===
df_labels = pd.read_csv(LABELS_PATH, sep=";")
df_labels.columns = df_labels.columns.str.strip().str.lower()
df_labels["id"] = df_labels["id"].str.strip().str.upper()
df_labels["label"] = df_labels["label"].astype(int)

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
        examples.append({"id": file_id, "text": text, "label": int(row["label"].values[0])})

texts = [e["text"] for e in examples]
labels = [e["label"] for e in examples]

# === Ensemble Prediction Per Seed ===
def get_last_n_checkpoints(dir_path, n=USE_LAST_N):
    all_cp = [d for d in os.listdir(dir_path) if d.startswith("checkpoint-")]
    sorted_cp = sorted(all_cp, key=lambda x: int(x.split("-")[-1]))
    return [os.path.join(dir_path, cp) for cp in sorted_cp[-n:]]

def get_best_n_checkpoints(dir_path, n=USE_LAST_N, metric="eval_accuracy"):
    checkpoints = []
    for subdir in os.listdir(dir_path):
        full_path = os.path.join(dir_path, subdir)
        if subdir.startswith("checkpoint-") and os.path.isdir(full_path):
            eval_path = os.path.join(full_path, "trainer_state.json")
            if os.path.exists(eval_path):
                try:
                    with open(eval_path) as f:
                        trainer_state = json.load(f)
                    # Get the last logged metric value for accuracy
                    metrics = trainer_state.get("log_history", [])
                    acc = max(
                        (entry.get(metric) for entry in metrics if metric in entry),
                        default=-1
                    )
                    checkpoints.append((acc, full_path))
                except Exception as e:
                    print(f"Warning: failed to read {eval_path} due to {e}")
    
    top = sorted(checkpoints, key=lambda x: x[0], reverse=True)[:n]
    return [path for _, path in top]


def run_model_predictions(model_dirs, model_type="bert"):
    all_logits = []
    for dir_path in model_dirs:
        tokenizer = AutoTokenizer.from_pretrained(dir_path)
        model = AutoModelForSequenceClassification.from_pretrained(dir_path).to("cuda")
        model.eval()
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                batch = texts[i:i+8]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                logits = model(**inputs).logits.cpu().numpy()
                logits_list.append(logits)
        all_logits.append(np.vstack(logits_list))
    return all_logits

all_accuracies = []
bert_accuracies = []
roberta_accuracies = []
ensemble_accuracies = []
seeds = SEED_RANGE if RUN_ALL_SEEDS else [42]

for seed in seeds:
    print(f"\nRunning ensemble for seed {seed}...")
    
    if USE_BEST:
        bert_ckpts = get_best_n_checkpoints(os.path.join(BASE_BERT_DIR, f"seed_pause_encoded_{seed}"), USE_LAST_N)
        roberta_ckpts = get_best_n_checkpoints(os.path.join(BASE_ROBERTA_DIR, f"seed_pause_encoded_{seed}"), USE_LAST_N)
    else:
        bert_ckpts = get_last_n_checkpoints(os.path.join(BASE_BERT_DIR, f"seed_pause_encoded_{seed}"), USE_LAST_N)
        roberta_ckpts = get_last_n_checkpoints(os.path.join(BASE_ROBERTA_DIR, f"seed_pause_encoded_{seed}"), USE_LAST_N)


    print(f"BERT checkpoints used: {bert_ckpts}")
    print(f"RoBERTa checkpoints used: {roberta_ckpts}")



    bert_logits = run_model_predictions(bert_ckpts, "bert")
    roberta_logits = run_model_predictions(roberta_ckpts, "roberta")

    all_model_logits = bert_logits + roberta_logits
    all_model_preds = [np.argmax(logits, axis=-1) for logits in all_model_logits]
    voted_preds, _ = mode(np.stack(all_model_preds), axis=0)
    final_preds = voted_preds.ravel()

    # Compute individual model accuracies
    bert_preds = [np.argmax(logits, axis=-1) for logits in bert_logits]
    roberta_preds = [np.argmax(logits, axis=-1) for logits in roberta_logits]

    # Majority vote within BERT and RoBERTa
    bert_voted, _ = mode(np.stack(bert_preds), axis=0)
    roberta_voted, _ = mode(np.stack(roberta_preds), axis=0)
    bert_final = bert_voted.ravel()
    roberta_final = roberta_voted.ravel()

    bert_acc = accuracy_score(labels, bert_final)
    roberta_acc = accuracy_score(labels, roberta_final)
    ensemble_acc = accuracy_score(labels, final_preds)

    print(f"[BERT]     Accuracy (seed {seed}): {bert_acc:.4f}")
    print(f"[RoBERTa]  Accuracy (seed {seed}): {roberta_acc:.4f}")
    print(f"[Ensemble] Accuracy (seed {seed}): {ensemble_acc:.4f}")

    wandb.log({
        f"bert_acc_seed_{seed}": bert_acc,
        f"roberta_acc_seed_{seed}": roberta_acc,
        f"ensemble_acc_seed_{seed}": ensemble_acc
    })

    bert_accuracies.append(bert_acc)
    roberta_accuracies.append(roberta_acc)
    ensemble_accuracies.append(ensemble_acc)



# === Summary ===
if RUN_ALL_SEEDS:
    print("\n========== Ensemble Summary ==========")
    print(f"[BERT]     Mean: {np.mean(bert_accuracies):.4f}, Std: {np.std(bert_accuracies):.4f}, Max: {np.max(bert_accuracies):.4f}")
    print(f"[RoBERTa]  Mean: {np.mean(roberta_accuracies):.4f}, Std: {np.std(roberta_accuracies):.4f}, Max: {np.max(roberta_accuracies):.4f}")
    print(f"[Ensemble] Mean: {np.mean(ensemble_accuracies):.4f}, Std: {np.std(ensemble_accuracies):.4f}, Max: {np.max(ensemble_accuracies):.4f}")

    wandb.log({
        "bert_mean_accuracy": np.mean(bert_accuracies),
        "bert_std_accuracy": np.std(bert_accuracies),
        "bert_max_accuracy": np.max(bert_accuracies),
        "roberta_mean_accuracy": np.mean(roberta_accuracies),
        "roberta_std_accuracy": np.std(roberta_accuracies),
        "roberta_max_accuracy": np.max(roberta_accuracies),
        "ensemble_mean_accuracy": np.mean(ensemble_accuracies),
        "ensemble_std_accuracy": np.std(ensemble_accuracies),
        "ensemble_max_accuracy": np.max(ensemble_accuracies)
    })


wandb.finish()