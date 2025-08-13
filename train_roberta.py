import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.metrics import accuracy_score
import wandb

# --- Configuration ---
TRANSCRIPT_DIR = "data/mfa_data/transcripts"  # Directory with cleaned .txt files
CONTROL_IDS = {
    "S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011",
    "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021",
    "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034",
    "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049",
    "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063",
    "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"
}

# --- W&B Setup ---
run = wandb.init(
    entity="kj1080-university-of-cambridge",
    project="mlmi_thesis",
    name=f"roberta_ad_text_run_{pd.Timestamp.now().strftime('%m%d_%H%M')}",
    config={
        "learning_rate": 1e-5,
        "architecture": "roberta-base",
        "dataset": "ADReSS-cleaned-txts",
        "epochs": 20,
    },
)

# --- Load data ---
print("Reading cleaned transcripts...")
rows = []
for fname in os.listdir(TRANSCRIPT_DIR):
    if fname.endswith(".txt"):
        file_id = os.path.splitext(fname)[0].upper()
        label = 0 if file_id in CONTROL_IDS else 1
        with open(os.path.join(TRANSCRIPT_DIR, fname), "r") as f:
            text = f.read().strip()
        if text:
            rows.append({"text": text, "label": label})

df = pd.DataFrame(rows)
df = df.dropna(subset=["text", "label"])

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# --- Tokenizer ---
print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# --- Tokenize ---
print("Tokenizing...")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --- Model ---
print("Loading RoBERTa model...")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# --- Callbacks ---
class CustomWandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs, step=state.global_step)

class SaveLogitsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Only save logits if evaluation happened after full epoch
        if state.epoch is None:
            return
        output = trainer.predict(test_dataset)
        logits = output.predictions
        labels = output.label_ids
        os.makedirs("roberta_results/checkpoint_logits", exist_ok=True)
        epoch = int(state.epoch)
        np.save(f"roberta_results/checkpoint_logits/epoch_{epoch}_logits.npy", logits)
        np.save("roberta_results/checkpoint_logits/labels.npy", labels)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./roberta_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./roberta_logs",
    logging_steps=5,
    report_to="wandb",
    # save_steps=None, 
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[
        CustomWandbLoggingCallback(),
        SaveLogitsCallback()
    ]
)

# --- Train ---
print("Training started...")
trainer.train()
run.finish()
print("Training complete.")
