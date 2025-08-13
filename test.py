import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    DataCollatorWithPadding,
)

# --- Paths ---
MODEL_DIR = "/home/kej48/rds/hpc-work/Thesis/results/checkpoint-132"
AUDIO_DIR = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/Full_wave_enhanced_audio"
METADATA_PATH = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"

# --- Load model and extractor ---
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.to("cuda")
model.eval()

# --- Load test metadata ---
df_meta = pd.read_csv(METADATA_PATH, sep=";")
df_meta.columns = [col.strip().lower() for col in df_meta.columns]
df_meta["id"] = df_meta["id"].str.strip()
df_meta["label"] = df_meta["label"].astype(int)

# --- Prepare audio examples ---
examples = []
for idx, row in df_meta.iterrows():
    subj_id = row["id"]
    print(f"Loaded {idx+1}/{len(df_meta)}: {subj_id}")
    wav_path = os.path.join(AUDIO_DIR, subj_id + ".wav")
    if not os.path.exists(wav_path):
        print(f" Missing audio: {wav_path}")
        continue
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    examples.append({
        "input_values": inputs["input_values"][0],
        "label": row["label"],
        "id": subj_id
    })

# --- Inference in mini-batches ---
batch_size = 4
all_preds = []
true_labels = []

collator = DataCollatorWithPadding(feature_extractor, return_tensors="pt")

print("Running inference in mini-batches...")
try:
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        collated = collator([{"input_values": ex["input_values"], "label": ex["label"]} for ex in batch])
        input_values = collated["input_values"].to("cuda")

        with torch.no_grad():
            logits = model(input_values).logits
            torch.cuda.synchronize()

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        true_labels.extend([ex["label"] for ex in batch])
        print(f"Batch {i // batch_size + 1} complete")

except KeyboardInterrupt:
    print("KeyboardInterrupt caught. Returning results collected so far.")

# --- Evaluation ---
print("Evaluating predictions...")
acc = accuracy_score(true_labels, all_preds)
print(f"\nOverall Accuracy: {acc:.4f}\n")

cm = confusion_matrix(true_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_labels, all_preds, target_names=["Control (0)", "AD (1)"]))

print(f"\nType I Errors (False Positives): {fp}")
print(f"Type II Errors (False Negatives): {fn}")



