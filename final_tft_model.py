import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import random
import wandb

# --- Constants ---
pause_encoding = True

if pause_encoding == False:
    TRANSCRIPT_DIR = "data/mfa_data/transcripts"
    TRANSCRIPT_TEST_DIR = "data/mfa_test_data/transcripts"
elif pause_encoding == True:
    TRANSCRIPT_DIR = "pause_encoding/train"
    TRANSCRIPT_TEST_DIR = "pause_encoding/test"
    
TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"

CONTROL_IDS = {
    "S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011",
    "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021",
    "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034",
    "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049",
    "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063",
    "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"
}

# --- Load transcript data ---
def load_transcripts(transcript_dir, is_test=False):
    rows = []
    for fname in os.listdir(transcript_dir):
        if fname.endswith(".txt"):
            file_id = os.path.splitext(fname)[0].upper()
            label = None if is_test else (0 if file_id in CONTROL_IDS else 1)
            with open(os.path.join(transcript_dir, fname), "r") as f:
                text = f.read().strip()
            if text:
                rows.append({"id": file_id, "text": text, "label": label})
    return pd.DataFrame(rows)

# --- Load test labels ---
def load_test_set():
    # Load test labels
    df_labels = pd.read_csv(TEST_LABELS_PATH, sep=";")
    df_labels.columns = df_labels.columns.str.strip().str.lower()
    df_labels["id"] = df_labels["id"].str.strip().str.upper()
    df_labels["label"] = df_labels["label"].astype(int)

    # Load transcripts
    rows = []
    for fname in os.listdir(TRANSCRIPT_TEST_DIR):
        if fname.endswith(".txt"):
            file_id = os.path.splitext(fname)[0].upper()
            label_row = df_labels[df_labels["id"] == file_id]
            if label_row.empty:
                continue
            with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
                text = f.read().strip()
            rows.append({
                "id": file_id,
                "text": text,
                "label": int(label_row["label"].values[0])
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["text", "label"])
    return df


# --- Tokenization ---
def tokenize_and_format(df, tokenizer):
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512), batched=True)
    ds = ds.remove_columns(["text", "id"]) if "id" in ds.column_names else ds.remove_columns(["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# --- Seeded Training Loop ---
accuracies = []
seeds = range(42, 43)

train_df_full = load_transcripts(TRANSCRIPT_DIR)
train_df_full = train_df_full.dropna()
test_df = load_test_set()

for seed in seeds:
    print(f"\n========== SEED {seed} ==========")
    run = wandb.init(
    entity="kj1080-university-of-cambridge",
    project="mlmi_thesis",
    name=f"bert_ad_text_pause_encoded_seed_{seed}",
    config={
        "seed": seed,
        "learning_rate": 1e-5,
        "architecture": "bert-base-uncased",
        "dataset": "ADReSS-cleaned-txts",
        "epochs": 20,
    },
    reinit=True,
)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Split train/val
    train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = tokenize_and_format(train_df, tokenizer)
    val_ds = tokenize_and_format(val_df, tokenizer)
    test_df
    test_ds = tokenize_and_format(test_df, tokenizer)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./bert_results/seed_pause_encoded_{seed}",
        evaluation_strategy="epoch",
        save_strategy="None",
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"./bert_logs/seed_pause_encoded_{seed}",
        logging_steps=10,
        report_to="wandb",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    # --- Final test set evaluation ---
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    acc = accuracy_score(labels, preds)
    print(f"Test Accuracy for seed {seed}: {acc:.4f}")
    accuracies.append(acc)
    run.finish()

# --- Summary ---
print("\n========== Summary Over 15 Seeds ==========")
print(f"Mean Accuracy     : {np.mean(accuracies):.4f}")
print(f"Std Deviation     : {np.std(accuracies):.4f}")
print(f"Max Accuracy      : {np.max(accuracies):.4f}")
