import torch
import os
import pandas as pd
import numpy as np
import random
import wandb
from transformers import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader

# ===== Options =====
pause_encoding = True
question_first = "version_2" # 'late_fusion_different_prompt'
majority_voting = False
model_used = "ensemble"  # "bert", "roberta", or "ensemble"
top_n_epochs = 4  # NEW: Number of best epochs to ensemble
num_epochs = 20

bert_accuracies = []
roberta_accuracies = []


# ===== Paths =====
if pause_encoding:
    TRANSCRIPT_DIR = "pause_encoding/train"
    TRANSCRIPT_TEST_DIR = "pause_encoding/test"
else:
    TRANSCRIPT_DIR = "data/mfa_data/transcripts"
    TRANSCRIPT_TEST_DIR = "data/mfa_test_data/transcripts"

TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"
CONTROL_IDS = {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011",
               "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021",
               "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034",
               "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049",
               "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063",
               "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"}


# Temporary tokenizer for measuring truncation statistics before training
tmp_tokenizer = load_plm("bert", "bert-base-uncased")[1]  # only tokenizer is needed



# ===== Utilities =====
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
    return pd.DataFrame(rows).dropna(subset=["text", "label"])

def normalize_logits(logits):
    return (logits - logits.mean(axis=1, keepdims=True)) / (logits.std(axis=1, keepdims=True) + 1e-8)

def create_prompt_dataset(df, tokenizer, split_name="unspecified", max_len=512):
    dataset = []

    trunc_stats = {
        0: {"truncated": 0, "total": 0, "lengths": []},  # healthy
        1: {"truncated": 0, "total": 0, "lengths": []}   # AD
    }

    for i, row in df.iterrows():
        text = row["text"]
        label = int(row["label"])
        tokenized = tokenizer(text, truncation=False, add_special_tokens=False)
        input_len = len(tokenized["input_ids"])

        trunc_stats[label]["total"] += 1
        trunc_stats[label]["lengths"].append(input_len)

        if input_len > (max_len - 10):  # truncation threshold
            trunc_stats[label]["truncated"] += 1

        # Apply truncation for actual use
        tokenized = tokenizer(text, truncation=True, max_length=max_len - 10, add_special_tokens=False)
        truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        dataset.append(InputExample(guid=i, text_a=truncated_text, label=label))

    print(f"\nTruncation report for {split_name.upper()}:")
    for label in [0, 1]:
        label_name = "healthy" if label == 0 else "AD"
        count = trunc_stats[label]["total"]
        trunc_count = trunc_stats[label]["truncated"]
        mean_len = np.mean(trunc_stats[label]["lengths"])
        max_len_sample = np.max(trunc_stats[label]["lengths"])
        print(f"  {label_name}:")
        print(f"    Total           = {count}")
        print(f"    Truncated       = {trunc_count} ({(trunc_count / count * 100):.2f}%)")
        print(f"    Mean token len  = {mean_len:.1f}")
        print(f"    Max token len   = {max_len_sample}")

    return dataset


def run_pbft(seed, train_df, val_df, test_df):
    if model_used == "ensemble":
        model_configs = [("bert", "bert-base-uncased"), ("roberta", "roberta-base")]
    else:
        model_configs = [(model_used, {
            "bert": "bert-base-uncased",
            "roberta": "roberta-base"
        }[model_used])]

    run = wandb.init(
        entity="kj1080-university-of-cambridge",
        project="mlmi_thesis",
        name=f"pbft__{model_used.upper()}_ad_text_pause_encoded_seed_{seed}",
        config={"seed": seed, "learning_rate": 1e-5, "model": model_configs, "epochs": num_epochs, "prompt_style": question_first},
        settings=wandb.Settings(_disable_stats=True),
        resume=False
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if question_first == "late_fusion":
        templates = ['The diagnosis result is {"mask"}. {"placeholder":"text_a"}.', '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
    elif question_first == "late_fusion_version_1":
        templates = ['Based on their speech patterns within this transcript: {"placeholder":"text_a"} , this individual is {"mask"}.',
                     'This individual is {"mask"}. This is based on their speech patterns within this transcript: {"placeholder":"text_a"}.']
    elif question_first == "version_2":
        templates = ['''
                     You are a clinical speech assessment expert diagnosing whether a patient has Alzheimer's disease or is healthy, based solely on their verbal response to a picture description task.

                        Patients are shown a domestic cartoon scene and asked: "Describe what you see in this picture."

                        The speech responses include:
                        - Long pauses, represented as `...`
                        - Medium pauses, represented as `.`

                        Clinical research shows that:
                        - A high number of long (`...`) and medium (`.`) pauses is strongly correlated with Alzheimer's disease.
                        - Fewer pauses suggest healthy cognitive function.

                        Here is an example:

                        Response: "There is a woman ... uh ... doing dishes . the sink is overflowing ... and the child is ... on a stool . reaching for the cookie jar."

                        Diagnosis: Alzheimer's Disease

                        Now classify the following response:

                        Response: {'placeholder':'text_a'}.

                        Diagnosis: {"mask"}''']
    elif question_first == True:
        single_prompt = ('The diagnosis result is {"mask"}. {"placeholder":"text_a"}.')
    elif question_first == False:
        single_prompt = ('{"placeholder":"text_a"}. The diagnosis result is {"mask"}.')
    else:
        print('Error: invalid arguement for question_first')
        templates = [single_prompt]

    all_preds_models = []
    all_labels_final = None

    for plm_type, model_path in model_configs:
        for template_text in templates:
            plm, tokenizer, model_config, WrapperClass = load_plm(plm_type, model_path)
            template = ManualTemplate(text=template_text, tokenizer=tokenizer)

            verbalizer = ManualVerbalizer(
                classes=["alzheimer", "healthy"],
                label_words={"alzheimer": ["alzheimer"], "healthy": ["healthy"]},
                tokenizer=tokenizer
            )

            train_dataset = create_prompt_dataset(train_df, tokenizer, split_name="train")
            val_dataset = create_prompt_dataset(val_df, tokenizer, split_name="val")
            test_dataset = create_prompt_dataset(test_df, tokenizer, split_name="test")
            
            
train_df_full = load_transcripts(TRANSCRIPT_DIR).dropna()
test_df = load_test_set()
train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=44)

# Now safe to analyze truncation
_ = create_prompt_dataset(train_df, tmp_tokenizer, split_name="train")
_ = create_prompt_dataset(val_df, tmp_tokenizer, split_name="val")
_ = create_prompt_dataset(test_df, tmp_tokenizer, split_name="test")
