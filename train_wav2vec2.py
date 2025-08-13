# import os
# import torch
# import pandas as pd
# import numpy as np
# import soundfile as sf
# import librosa
# import torchaudio
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import (
#     Wav2Vec2FeatureExtractor,
#     Wav2Vec2ForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     DataCollatorWithPadding,
#     TrainerCallback,
# )
# from sklearn.metrics import accuracy_score
# import wandb
# run = wandb.init(
#     entity="kj1080-university-of-cambridge",
#     project="mlmi_thesis",
#     name=f"wav2vec2_run_{pd.Timestamp.now().strftime('%m%d_%H%M')}",
#     config={
#         "learning_rate": 5e-6,
#         "architecture": "wav2vec2",
#         "dataset": "ADReSS",
#         "epochs": 15,
#     },
# )

# # --- Configuration ---
# DATA_DIR = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio"
# LABELS = {"cc": 0, "cd": 1}

# class CustomWandbLoggingCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             logs["step"] = state.global_step
#             logs["epoch_num"] = int(state.epoch or 0)
#             wandb.log(logs, step=state.global_step)

# # --- Load file paths and labels ---
# def load_audio_data(data_dir):
#     data = []
#     for label in os.listdir(data_dir):
#         class_dir = os.path.join(data_dir, label)
#         if not os.path.isdir(class_dir):
#             continue
#         for file in os.listdir(class_dir):
#             if file.endswith(".wav"):
#                 data.append({
#                     "path": os.path.join(class_dir, file),
#                     "label": LABELS[label]
#                 })
#     print(data)
#     return pd.DataFrame(data)

# df = load_audio_data(DATA_DIR)
# train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# # --- Feature extractor ---
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# # --- Preprocess function ---
# def preprocess(example):
#     audio, sr = sf.read(example["path"])

#     # Ensure mono channel
#     if len(audio.shape) > 1:
#         audio = np.mean(audio, axis=1)

#     # Skip if too short to be meaningful or resampled
#     if len(audio) < 300:  # adjust this threshold as needed
#         print(f"Skipping short file: {example['path']} with length {len(audio)}")
#         return {"input_values": None, "label": None}

#     if sr != 16000:
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

#     inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
#     return {
#         "input_values": inputs["input_values"][0],
#         "label": example["label"]
#     }


# train_dataset = Dataset.from_pandas(train_df).map(preprocess)
# test_dataset = Dataset.from_pandas(test_df).map(preprocess)

# # --- Load model ---
# model = Wav2Vec2ForSequenceClassification.from_pretrained(
#     "facebook/wav2vec2-base",
#     num_labels=2,
#     problem_type="single_label_classification"
# )

# # model.freeze_feature_encoder()
# model.gradient_checkpointing_enable()

# # --- Training arguments ---
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="epoch",         # evaluate after every epoch
#     save_strategy="epoch",               # save after every epoch
#     save_total_limit=1,                  # only keep the best checkpoint
#     load_best_model_at_end=True,        # reload the best model after training
#     metric_for_best_model="accuracy",    # metric to track
#     greater_is_better=True,              # higher accuracy is better

#     learning_rate=5e-6,
#     warmup_steps=50,
#     lr_scheduler_type="linear",
#     per_device_train_batch_size=8,  
#     per_device_eval_batch_size=8,
#     num_train_epochs=15,
#     logging_dir="./logs",
#     logging_steps=5,
#     disable_tqdm=False,
#     report_to="wandb",
# )


# # --- Metrics ---
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)
#     return {"accuracy": accuracy_score(labels, preds)}

# # --- Trainer ---
# data_collator = DataCollatorWithPadding(feature_extractor)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=feature_extractor,
#     compute_metrics=compute_metrics,
#     data_collator=data_collator,
#     # callbacks=[CustomWandbLoggingCallback()],
# )

# # --- Train (safe wrapper) ---
# try:
#     trainer.train()
# except Exception as e:
#     import traceback
#     with open("crash_log.txt", "w") as f:
#         traceback.print_exc(file=f)
#     raise

# # Finish the run and upload any remaining data.
# run.finish()


# code to run the sweep
# import os
# import torch
# import pandas as pd
# import numpy as np
# import soundfile as sf
# import librosa
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import (
#     Wav2Vec2FeatureExtractor,
#     Wav2Vec2ForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     DataCollatorWithPadding,
#     TrainerCallback,
#     EarlyStoppingCallback,
# )
# from sklearn.metrics import accuracy_score
# import wandb

# # def main():
# # --- Weights & Biases Setup ---
# run = wandb.init(
#     entity="kj1080-university-of-cambridge",
#     project="mlmi_thesis",
#     name=f"wav2vec2_run_{pd.Timestamp.now().strftime('%m%d_%H%M')}",
#     config={
#         "learning_rate": 1e-6,
#         "architecture": "wav2vec2",
#         "dataset": "ADReSS",
#         "epochs": 15,
#     },
# )

# # --- Configuration ---
# DATA_DIR = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio"
# LABELS = {"cc": 0, "cd": 1}

# class CustomWandbLoggingCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             logs["step"] = state.global_step
#             logs["epoch_num"] = int(state.epoch or 0)
#             wandb.log(logs, step=state.global_step)

# # --- Load audio metadata ---
# def load_audio_data(data_dir):
#     data = []
#     for label in os.listdir(data_dir):
#         class_dir = os.path.join(data_dir, label)
#         if not os.path.isdir(class_dir):
#             continue
#         for file in os.listdir(class_dir):
#             if file.endswith(".wav"):
#                 data.append({
#                     "path": os.path.join(class_dir, file),
#                     "label": LABELS[label]
#                 })
#     print(data)
#     return pd.DataFrame(data)

# df = load_audio_data(DATA_DIR)
# train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# # --- Feature extractor ---
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# # --- Preprocess function ---
# def preprocess(example):
#     audio, sr = sf.read(example["path"])
#     if len(audio.shape) > 1:
#         audio = np.mean(audio, axis=1)
#     if len(audio) < 300:
#         print(f"Skipping short file: {example['path']} with length {len(audio)}")
#         return {"input_values": None, "label": None}
#     if sr != 16000:
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
#     inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
#     return {
#         "input_values": inputs["input_values"][0],
#         "label": example["label"]
#     }

# train_dataset = Dataset.from_pandas(train_df).map(preprocess)
# test_dataset = Dataset.from_pandas(test_df).map(preprocess)

# # --- Model ---
# model = Wav2Vec2ForSequenceClassification.from_pretrained(
#     "facebook/wav2vec2-base",
#     num_labels=2,
#     problem_type="single_label_classification"
# )
# model.gradient_checkpointing_enable()

# # --- Training args ---
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     greater_is_better=True,
#     # learning_rate=wandb.config.learning_rate,
#     learning_rate=5e-6,
#     warmup_steps=50,
#     lr_scheduler_type="linear",
#     # per_device_train_batch_size=wandb.config.batch_size,
#     # per_device_eval_batch_size=wandb.config.batch_size,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     # num_train_epochs=wandb.config.epochs,
#     num_train_epochs=15,
#     logging_dir="./logs",
#     logging_steps=5,
#     disable_tqdm=False,
#     report_to="wandb",
# )

# # --- Metrics ---
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)
#     return {"accuracy": accuracy_score(labels, preds)}

# # --- Trainer ---
# data_collator = DataCollatorWithPadding(feature_extractor)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=feature_extractor,
#     compute_metrics=compute_metrics,
#     data_collator=data_collator,
#     callbacks=[
#         CustomWandbLoggingCallback(),
#         EarlyStoppingCallback(early_stopping_patience=2)
#     ],
# )

# # --- Train safely ---
# try:
#     trainer.train()
# except Exception as e:
#     import traceback
#     with open("crash_log.txt", "w") as f:
#         traceback.print_exc(file=f)
#     raise

# run.finish()

# if __name__ == "__main__":
#     main()


import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import signal
import time
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score
import wandb




print("Script started...")

# --- Weights & Biases Setup ---
run = wandb.init(
    entity="kj1080-university-of-cambridge",
    project="mlmi_thesis",
    name=f"wav2vec2_run_{pd.Timestamp.now().strftime('%m%d_%H%M')}",
    config={
        "learning_rate": 2e-5,
        "architecture": "wav2vec2",
        "dataset": "ADReSS",
        "epochs": 10,
    },
)



# --- Configuration ---
DATA_DIR = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio"

# DATA_DIR = "/home/kej48/rds/hpc-work/Thesis/data/adress_small_testing/train"
LABELS = {"cc": 0, "cd": 1}

class CustomWandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["step"] = state.global_step
            logs["epoch_num"] = int(state.epoch or 0)
            wandb.log(logs, step=state.global_step)

print("Loading audio file paths...")
def load_audio_data(data_dir):
    data = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.endswith(".wav"):
                data.append({
                    "path": os.path.join(class_dir, file),
                    "label": LABELS[label]
                })
    print(f"Loaded {len(data)} audio files")
    return pd.DataFrame(data)

df = load_audio_data(DATA_DIR)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

print("🔧 Loading feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# --- Preprocess function ---
def preprocess(example):
    audio, sr = sf.read(example["path"])
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if len(audio) < 300:
        print(f"Skipping short file: {example['path']}")
        return {"input_values": None, "label": None}
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    return {
        "input_values": inputs["input_values"][0].tolist(),
        "label": int(example["label"])                     
    }

print("Preprocessing datasets...")
start_time = time.time()
train_dataset = Dataset.from_pandas(train_df).map(preprocess)
test_dataset = Dataset.from_pandas(test_df).map(preprocess)
train_dataset = train_dataset.filter(lambda x: x["input_values"] is not None)
test_dataset = test_dataset.filter(lambda x: x["input_values"] is not None)
print(f"Preprocessing done in {time.time() - start_time:.1f}s")
print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

print("Loading model...")
from transformers import Wav2Vec2Config

config = Wav2Vec2Config.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2,
    problem_type="single_label_classification",
    hidden_dropout=0.1,
    attention_dropout=0.1,
    classifier_proj_size=256
)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    config=config
)

# Freeze feature extractor and early encoder layers
model.freeze_feature_extractor()

freeze_layers = 4
for i in range(freeze_layers):
    for param in model.wav2vec2.encoder.layers[i].parameters():
        param.requires_grad = False

# model.gradient_checkpointing_enable()

print("Setting up training arguments...", flush=True)
print("Setting up training arguments...", flush=True)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    dataloader_num_workers=0,
    logging_dir="./logs",
    logging_steps=5,
    disable_tqdm=False,
    report_to="wandb",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

print("Initializing trainer...", flush=True)
data_collator = DataCollatorWithPadding(feature_extractor)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[
        CustomWandbLoggingCallback(),
        EarlyStoppingCallback(early_stopping_patience=5)
    ],
)

# --- Train safely ---
print("Starting training loop...")
try:
    torch.cuda.empty_cache()

    trainer.train()
except Exception as e:
    import traceback
    with open("crash_log.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Training crashed, see crash_log.txt")
    raise
finally:
    run.finish()
    print("Run finished.")
