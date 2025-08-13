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
question_first = "late_fusion" # 'late_fusion_different_prompt'
majority_voting = True
model_used = "ensemble"  # "bert", "roberta", or "ensemble"
top_n_epochs = 3  # NEW: Number of best epochs to ensemble
num_epochs = 10

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

def create_prompt_dataset(df, tokenizer, max_len=512):
    dataset = []
    for i, row in df.iterrows():
        text = row["text"]
        tokenized = tokenizer(text, truncation=True, max_length=max_len - 10, add_special_tokens=False)
        truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        dataset.append(InputExample(guid=i, text_a=truncated_text, label=int(row["label"])))
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
    elif question_first == "late_fusion_different_prompt":
        templates = ['Based on their speech patterns within this transcript: {"placeholder":"text_a"} , this individual is {"mask"}.',
                     'This individual is {"mask"}. This is based on their speech patterns within this transcript: {"placeholder":"text_a"}.']
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

            train_dataset = create_prompt_dataset(train_df, tokenizer)
            val_dataset = create_prompt_dataset(val_df, tokenizer)
            test_dataset = create_prompt_dataset(test_df, tokenizer)

            train_loader = PromptDataLoader(dataset=train_dataset, template=template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                            batch_size=1, shuffle=True, num_workers=0)

            val_loader = PromptDataLoader(dataset=val_dataset, template=template, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                          batch_size=1, shuffle=False, num_workers=0)

            test_loader = PromptDataLoader(dataset=test_dataset, template=template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                           batch_size=1, shuffle=False, num_workers=0)

            model = PromptForClassification(template=template, plm=plm, verbalizer=verbalizer)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            optimizer = AdamW(model.parameters(), lr=1e-5)

            epoch_logits = []
            epoch_val_acc = []

            for epoch in tqdm(range(num_epochs), desc=f"Epochs ({model_path})", position=1, leave=False):
                model.train()
                running_loss = 0.0
                for i, batch in enumerate(train_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(batch)
                    loss = torch.nn.CrossEntropyLoss()(logits, batch['label']) / 4
                    loss.backward()
                    running_loss += loss.item()
                    if (i + 1) % 4 == 0 or (i + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                wandb.log({f"train_loss_{model_path}": running_loss, "epoch": epoch + 1})

                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        logits = model(batch)
                        val_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                        val_labels.extend(batch['label'].cpu().numpy())
                val_preds = np.concatenate(val_preds)
                val_acc = accuracy_score(val_labels, val_preds)
                epoch_val_acc.append((val_acc, epoch))

                # Save test logits for this epoch
                preds_this_epoch = []
                test_labels = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        logits = model(batch)
                        preds_this_epoch.append(torch.softmax(logits, dim=-1).cpu().numpy())
                        test_labels.extend(batch['label'].cpu().numpy())

                epoch_logits.append(np.vstack(preds_this_epoch))
                all_labels_final = test_labels

            # Select best n epochs
            top_epochs = sorted(epoch_val_acc, reverse=True)[:top_n_epochs]
            for acc, best_epoch in top_epochs:
                logits_stack = normalize_logits(epoch_logits[best_epoch])
                all_preds_models.append(logits_stack)

            wandb.log({f"top_eval_accuracy_{model_path}": top_epochs[0][0]})

            # Compute and log individual model test accuracy from its best epoch
            best_epoch = top_epochs[0][1]
            logits_stack = epoch_logits[best_epoch]
            preds = np.argmax(logits_stack, axis=1)
            acc_single = accuracy_score(all_labels_final, preds)

            if model_used == "ensemble":
                if plm_type == "bert":
                    bert_accuracies.append(acc_single)
                elif plm_type == "roberta":
                    roberta_accuracies.append(acc_single)


    # Ensemble
    all_preds_models_np = np.array(all_preds_models)
    if majority_voting:
        votes = np.argmax(all_preds_models_np, axis=2)
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
    else:
        avg_preds = np.mean(all_preds_models_np, axis=0)
        final_preds = np.argmax(avg_preds, axis=1)

    acc = accuracy_score(all_labels_final, final_preds)
    if model_used == "ensemble":
        if plm_type == "bert":
            bert_accuracies.append(acc)
        elif plm_type == "roberta":
            roberta_accuracies.append(acc)

    print(f"Test Accuracy for seed {seed}: {acc:.4f}")
    wandb.log({"test_accuracy": acc, "ensemble_test_accuracy": acc})
    if model_used == "ensemble":
        wandb.log({"ensemble_strategy": "majority" if majority_voting else "softmax"})

    run.finish()
    return acc

# --- Main Execution ---
train_df_full = load_transcripts(TRANSCRIPT_DIR).dropna()
test_df = load_test_set()

accuracies = []
seeds = range(44, 45)

with tqdm(seeds, desc="Seeds", position=0) as seed_bar:
    for seed in seed_bar:
        seed_bar.set_description(f"Seed {seed}")
        train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)
        acc = run_pbft(seed, train_df, val_df, test_df)
        accuracies.append(acc)

print("\n========== PBFT Summary Over 15 Seeds ==========")
print(f"Mean Accuracy     : {np.mean(accuracies):.4f}")
print(f"Std Deviation     : {np.std(accuracies):.4f}")
print(f"Max Accuracy      : {np.max(accuracies):.4f}")

if model_used == "ensemble":
    print("\n------ Individual Model Summary ------")
    if bert_accuracies:
        print(f"[BERT]    Mean: {np.mean(bert_accuracies):.4f}, Std: {np.std(bert_accuracies):.4f}, Max: {np.max(bert_accuracies):.4f}")
    else:
        print("[BERT]    No results recorded.")

    if roberta_accuracies:
        print(f"[RoBERTa] Mean: {np.mean(roberta_accuracies):.4f}, Std: {np.std(roberta_accuracies):.4f}, Max: {np.max(roberta_accuracies):.4f}")
    else:
        print("[RoBERTa] No results recorded.")
