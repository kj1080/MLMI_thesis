import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, classification_report

seed = 45
model_name = "ensemble_base_different_BERT-V1_RoBERTa-Original" # "ensemble_base_both_original" 

output_dir = "ensemble_outputs/nonmoe"

ids = np.load(os.path.join(output_dir, f"ids_{model_name}_seed{seed}.npy"))
labels = np.load(os.path.join(output_dir, f"labels_{model_name}_seed{seed}.npy"))
probs = np.load(os.path.join(output_dir, f"probs_{model_name}_seed{seed}.npy"))
preds = np.load(os.path.join(output_dir, f"preds_{model_name}_seed{seed}.npy"))
val_weights = np.load(os.path.join(output_dir, f"val_weights_{model_name}_seed{seed}.npy"))

with open(os.path.join(output_dir, f"meta_{model_name}_seed{seed}.json")) as f:
    meta = json.load(f)

print("Accuracy:", accuracy_score(labels, preds))
print(classification_report(labels, preds))

print("Validation weights:", val_weights)

print("First 10 samples, class probabilities:")
for i in range(10):
    print(ids[i], probs[i], "True:", labels[i], "Pred:", preds[i])
