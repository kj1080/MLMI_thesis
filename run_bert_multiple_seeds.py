import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import random
import wandb



# --- Constants ---
pause_encoding = False


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
seeds = range(42, 57)

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
        # save_strategy="epoch",
        # save_total_limit=None,
        # load_best_model_at_end=True,
        save_strategy="no",  # Don't save any model checkpoints
        save_total_limit=0,  # (Optional, but makes intent clear)
        load_best_model_at_end=False,  # Not needed if not saving
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


# import os
# import torch
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from transformers import (
#     BertTokenizer, BertForSequenceClassification,
#     RobertaTokenizer, RobertaForSequenceClassification,
#     TrainingArguments, Trainer, DataCollatorWithPadding,
# )
# import random
# import wandb

# # --- Constants ---
# pause_encoding = True
# NUM_EPOCHS = 20
# LAST_N_EPOCHS = 3
# BATCH_SIZE = 4
# SEEDS = range(42, 43)  # Modify as needed
# MODELS = ['bert', 'roberta']

# if pause_encoding:
#     TRANSCRIPT_DIR = "pause_encoding/train"
#     TRANSCRIPT_TEST_DIR = "pause_encoding/test"
# else:
#     TRANSCRIPT_DIR = "data/mfa_data/transcripts"
#     TRANSCRIPT_TEST_DIR = "data/mfa_test_data/transcripts"

# TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"
# CONTROL_IDS = {...}  # same as before

# # === Utility Functions ===
# def load_transcripts(transcript_dir, is_test=False):
#     rows = []
#     for fname in os.listdir(transcript_dir):
#         if fname.endswith(".txt"):
#             file_id = os.path.splitext(fname)[0].upper()
#             label = None if is_test else (0 if file_id in CONTROL_IDS else 1)
#             with open(os.path.join(transcript_dir, fname), "r") as f:
#                 text = f.read().strip()
#             if text:
#                 rows.append({"id": file_id, "text": text, "label": label})
#     return pd.DataFrame(rows)

# def load_test_set():
#     df_labels = pd.read_csv(TEST_LABELS_PATH, sep=";")
#     df_labels.columns = df_labels.columns.str.strip().str.lower()
#     df_labels["id"] = df_labels["id"].str.strip().str.upper()
#     df_labels["label"] = df_labels["label"].astype(int)

#     rows = []
#     for fname in os.listdir(TRANSCRIPT_TEST_DIR):
#         if fname.endswith(".txt"):
#             file_id = os.path.splitext(fname)[0].upper()
#             label_row = df_labels[df_labels["id"] == file_id]
#             if label_row.empty:
#                 continue
#             with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
#                 text = f.read().strip()
#             rows.append({"id": file_id, "text": text, "label": int(label_row["label"].values[0])})

#     df = pd.DataFrame(rows)
#     return df.dropna(subset=["text", "label"])

# def tokenize_and_format(df, tokenizer):
#     ds = Dataset.from_pandas(df)
#     ds = ds.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512), batched=True)
#     ds = ds.remove_columns(["text", "id"]) if "id" in ds.column_names else ds.remove_columns(["text"])
#     ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#     return ds

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)
#     return {"accuracy": accuracy_score(labels, preds)}

# def get_model_and_tokenizer(model_type):
#     if model_type == "bert":
#         return (
#             BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2),
#             BertTokenizer.from_pretrained("bert-base-uncased")
#         )
#     elif model_type == "roberta":
#         return (
#             RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2),
#             RobertaTokenizer.from_pretrained("roberta-base")
#         )

# def predict_from_last_n_checkpoints(model_type, output_dir, trainer, test_ds, last_n=LAST_N_EPOCHS):
#     preds_list = []

#     # Get all checkpoint dirs and sort by step number
#     checkpoint_dirs = [
#         os.path.join(output_dir, d)
#         for d in os.listdir(output_dir)
#         if d.startswith("checkpoint-")
#     ]
#     checkpoint_dirs = sorted(
#         checkpoint_dirs,
#         key=lambda x: int(x.split("-")[-1])
#     )[-last_n:]

#     if not checkpoint_dirs:
#         raise ValueError(f"No checkpoints found in {output_dir}")

#     model_class = (
#         BertForSequenceClassification if model_type == "bert" else RobertaForSequenceClassification
#     )

#     for ckpt_dir in checkpoint_dirs:
#         model = model_class.from_pretrained(ckpt_dir, num_labels=2)
#         trainer.model = model
#         preds_output = trainer.predict(test_ds)
#         preds_list.append(np.argmax(preds_output.predictions, axis=-1))

#     if not preds_list:
#         raise ValueError("No valid predictions were collected from checkpoints.")

#     return np.stack(preds_list)


# # === Main Training Loop ===
# train_df_full = load_transcripts(TRANSCRIPT_DIR).dropna()
# test_df = load_test_set()
# labels = test_df["label"].tolist()

# summary = {"bert": [], "roberta": [], "ensemble": []}

# for seed in SEEDS:
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     train_df, val_df = train_test_split(
#         train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)

#     test_preds_all = {}

#     for model_type in MODELS:
#         run = wandb.init(
#             entity="kj1080-university-of-cambridge",
#             project="mlmi_thesis",
#             name=f"{model_type}_pause_encoded_seed_{seed}",
#             config={
#                 "seed": seed,
#                 "model": model_type,
#                 "epochs": NUM_EPOCHS,
#                 "pause_encoding": pause_encoding
#             },
#             reinit=True
#         )

#         model, tokenizer = get_model_and_tokenizer(model_type)
#         train_ds = tokenize_and_format(train_df, tokenizer)
#         val_ds = tokenize_and_format(val_df, tokenizer)
#         test_ds = tokenize_and_format(test_df, tokenizer)

#         output_dir = f"./{model_type}_results/seed_{seed}"
#         steps_per_epoch = len(train_ds) // BATCH_SIZE

#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             evaluation_strategy="epoch",
#             save_strategy="epoch",
#             save_total_limit=None,
#             load_best_model_at_end=False,
#             num_train_epochs=NUM_EPOCHS,
#             learning_rate=1e-5,
#             per_device_train_batch_size=BATCH_SIZE,
#             per_device_eval_batch_size=BATCH_SIZE,
#             weight_decay=0.01,
#             warmup_ratio=0.1,
#             logging_dir=f"./{model_type}_logs/seed_{seed}",
#             logging_steps=10,
#             report_to="wandb",
#             disable_tqdm=True
#         )

#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_ds,
#             eval_dataset=val_ds,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorWithPadding(tokenizer),
#             compute_metrics=compute_metrics
#         )

#         trainer.train()

#         # Predict from last N checkpoints
#         pred_stack = predict_from_last_n_checkpoints(model_type, output_dir, trainer, test_ds, steps_per_epoch)
#         majority_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=pred_stack)

#         acc = accuracy_score(labels, majority_preds)
#         print(f"[{model_type.upper()}][Seed {seed}] Accuracy: {acc:.4f}")
#         summary[model_type].append(acc)
#         test_preds_all[model_type] = majority_preds
#         run.finish()

#     # === Ensemble: Majority vote between models ===
#     preds_ensemble = np.stack([test_preds_all["bert"], test_preds_all["roberta"]], axis=0)
#     final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds_ensemble)
#     ensemble_acc = accuracy_score(labels, final_preds)
#     summary["ensemble"].append(ensemble_acc)
#     print(f"[ENSEMBLE][Seed {seed}] Accuracy: {ensemble_acc:.4f}")

# # === Summary ===
# print("\n========== Final Summary Over All Seeds ==========")
# for key in summary:
#     mean_acc = np.mean(summary[key])
#     std_acc = np.std(summary[key])
#     max_acc = np.max(summary[key])
#     print(f"[{key.upper()}] Mean: {mean_acc:.4f}, Std: {std_acc:.4f}, Max: {max_acc:.4f}")
