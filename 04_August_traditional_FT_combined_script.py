# combined_run.py

import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from transformers import TrainerCallback
import wandb
from scipy.stats import mode

# === Global Constants ===
SEEDS = list(range(42, 57))
N_FOLDS = 0
DO_CV = False
MODELS = ['bert', 'roberta']
ENSEMBLE_LAST_N_EPOCHS = 3
pause_encoding = True
USE_WEIGHTED_VOTING = True  # Set to True for weighted majority


if pause_encoding:
    TRANSCRIPT_DIR = "pause_encoding/train"
    TRANSCRIPT_TEST_DIR = "pause_encoding/test"
else:
    TRANSCRIPT_DIR = "data/mfa_data/transcripts"
    TRANSCRIPT_TEST_DIR = "data/mfa_test_data/transcripts"

TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"
CONTROL_IDS = {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011", "S012", "S013", "S015", "S016",
               "S017", "S018", "S019", "S020", "S021", "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033",
               "S034", "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049", "S051", "S052", "S055",
               "S056", "S058", "S059", "S061", "S062", "S063", "S064", "S067", "S068", "S070", "S071", "S072", "S073",
               "S076", "S077"}

# === Utility Functions ===
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

def load_test_set():
    df_labels = pd.read_csv(TEST_LABELS_PATH, sep=";")
    df_labels.columns = df_labels.columns.str.strip().str.lower()
    df_labels["id"] = df_labels["id"].str.strip().str.upper()
    df_labels["label"] = df_labels["label"].astype(int)

    rows = []
    for fname in os.listdir(TRANSCRIPT_TEST_DIR):
        if fname.endswith(".txt"):
            file_id = os.path.splitext(fname)[0].upper()
            label_row = df_labels[df_labels["id"] == file_id]
            if label_row.empty:
                continue
            with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
                text = f.read().strip()
            rows.append({"id": file_id, "text": text, "label": int(label_row["label"].values[0])})

    df = pd.DataFrame(rows)
    return df.dropna(subset=["text", "label"])

def tokenize_and_format(df, tokenizer):
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512), batched=True)
    ds = ds.remove_columns(["text", "id"] if "id" in ds.column_names else ["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def train_and_collect_logits(model_name, train_df, val_df, test_df, seed, fold_id):
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    train_ds = tokenize_and_format(train_df, tokenizer)
    val_ds = tokenize_and_format(val_df, tokenizer)
    test_ds = tokenize_and_format(test_df, tokenizer)

    logits_epochs = []



class LogitCallback(TrainerCallback):
    def __init__(self, model, test_ds, tokenizer, logits_epochs):
        self.model = model
        self.test_ds = test_ds
        self.tokenizer = tokenizer
        self.logits_epochs = logits_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= (args.num_train_epochs - ENSEMBLE_LAST_N_EPOCHS):
            self.model.eval()
            logits_list = []

            with torch.no_grad():
                for i in range(0, len(self.test_ds), args.per_device_eval_batch_size):
                    batch = self.test_ds[i:i + args.per_device_eval_batch_size]
                    
                    # Stack batch tensors
                    input_ids = torch.stack([x for x in batch["input_ids"]]).to(self.model.device)
                    attention_mask = torch.stack([x for x in batch["attention_mask"]]).to(self.model.device)
                    
                    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

                    outputs = self.model(**inputs)
                    logits = outputs.logits.cpu().numpy()
                    logits_list.append(logits)

            stacked_logits = np.vstack(logits_list)
            self.logits_epochs.append(stacked_logits)

def train_and_collect_logits(model_name, train_df, val_df, test_df, seed, fold_id):
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    train_ds = tokenize_and_format(train_df, tokenizer)
    val_ds = tokenize_and_format(val_df, tokenizer)
    test_ds = tokenize_and_format(test_df, tokenizer)

    logits_epochs = []

    training_args = TrainingArguments(
        output_dir=f"./temp_output/{model_name}_seed{seed}_fold{fold_id}",
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        num_train_epochs=20,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=None,
        logging_steps=10,
        report_to="wandb",
        disable_tqdm=True
    )

    callback = LogitCallback(model, test_ds, tokenizer, logits_epochs)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[callback]
    )

    trainer.train()
    return logits_epochs[-ENSEMBLE_LAST_N_EPOCHS:]



# === Main Loop ===
train_df_full = load_transcripts(TRANSCRIPT_DIR).dropna()
test_df = load_test_set()
labels = test_df["label"].tolist()

summary = {"bert": [], "roberta": [], "ensemble": []}

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run = wandb.init(
        entity="kj1080-university-of-cambridge",
        project="mlmi_thesis",
        name=f"combined_cv_{DO_CV}_seed_{seed}",
        config={"seed": seed, "models": MODELS},
        reinit=True
    )

    fold_logits = {"bert": [], "roberta": []}

    for model_name in MODELS:
        if DO_CV:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold_id, (train_idx, val_idx) in enumerate(skf.split(train_df_full, train_df_full["label"])):
                train_df = train_df_full.iloc[train_idx]
                val_df = train_df_full.iloc[val_idx]
                logits_list = train_and_collect_logits(model_name, train_df, val_df, test_df, seed, fold_id)
                fold_logits[model_name].extend(logits_list)
        else:
            train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)
            logits_list = train_and_collect_logits(model_name, train_df, val_df, test_df, seed, fold_id=0)
            fold_logits[model_name].extend(logits_list)

    for model_name in MODELS:
        logits_stack = np.stack(fold_logits[model_name])  # shape: [n_snapshots, n_samples, 2]

        if USE_WEIGHTED_VOTING:
            # Sum logits (weighted vote)
            summed_logits = np.sum(logits_stack, axis=0)  # [n_samples, 2]
            voted_preds = np.argmax(summed_logits, axis=-1)
        else:
            # Majority vote on class predictions
            preds = np.argmax(logits_stack, axis=-1)  # [n_snapshots, n_samples]
            voted_preds, _ = mode(preds, axis=0)
            voted_preds = voted_preds.ravel()

        
        # logits_stack = np.stack(fold_logits[model_name])
        # preds = np.argmax(logits_stack, axis=-1)
        # voted_preds, _ = mode(preds, axis=0)
        acc = accuracy_score(labels, voted_preds.ravel())
        print(f"[{model_name.upper()}][Seed {seed}] Accuracy: {acc:.4f}")
        summary[model_name].append(acc)

    # Ensemble
    ensemble_logits = fold_logits['bert'] + fold_logits['roberta']
    logits_stack = np.stack(ensemble_logits)  # [n_snapshots, n_samples, 2]

    if USE_WEIGHTED_VOTING:
        summed_logits = np.sum(logits_stack, axis=0)
        voted_preds = np.argmax(summed_logits, axis=-1)
    else:
        all_preds = np.argmax(logits_stack, axis=-1)
        voted_preds, _ = mode(all_preds, axis=0)
        voted_preds = voted_preds.ravel()

    # ensemble_logits = fold_logits['bert'] + fold_logits['roberta']
    # all_preds = [np.argmax(logits, axis=-1) for logits in ensemble_logits]
    # voted_preds, _ = mode(np.stack(all_preds), axis=0)
    ensemble_acc = accuracy_score(labels, voted_preds.ravel())
    print(f"[ENSEMBLE][Seed {seed}] Accuracy: {ensemble_acc:.4f}")
    summary['ensemble'].append(ensemble_acc)

    wandb.log({
        f"bert_acc_seed_{seed}": summary['bert'][-1],
        f"roberta_acc_seed_{seed}": summary['roberta'][-1],
        f"ensemble_acc_seed_{seed}": summary['ensemble'][-1]
    })

    run.finish()

# === Final Summary ===
print("\n========== Final Summary Over All Seeds ==========")
for model_name in ["bert", "roberta", "ensemble"]:
    mean_acc = np.mean(summary[model_name])
    std_acc = np.std(summary[model_name])
    max_acc = np.max(summary[model_name])
    print(f"[{model_name.upper()}] Mean: {mean_acc:.4f}, Std: {std_acc:.4f}, Max: {max_acc:.4f}")

wandb.finish()
