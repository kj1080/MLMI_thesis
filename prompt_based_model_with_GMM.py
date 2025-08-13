import os
import re
import torch
import random
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import AdamW
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader

# ===== Path Settings =====
TRANSCRIPT_DIR = "pause_encoding/train"
TRANSCRIPT_TEST_DIR = "pause_encoding/test"
TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"

CONTROL_IDS = {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011", "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034", "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049", "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063", "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"}

model_used = "ensemble"
plm_size = "large"
question_first = "late_fusion"
num_epochs = 20
top_n_epochs = 3
majority_voting = False

bert_accuracies = []
roberta_accuracies = []

# ===== Pause Counting =====
def count_pauses(text):
    long = len(re.findall(r"\.\.\.", text))
    medium = len(re.findall(r"(?<!\.)\.(?!\.)", text))
    return medium, long

# ===== Load Transcripts =====
def load_transcripts_with_pauses(transcript_dir, is_test=False):
    rows = []
    for fname in os.listdir(transcript_dir):
        if not fname.endswith(".txt"): continue
        file_id = os.path.splitext(fname)[0].upper()
        label = None if is_test else (0 if file_id in CONTROL_IDS else 1)
        with open(os.path.join(transcript_dir, fname), "r") as f:
            text = f.read().strip()
        if text:
            medium, long = count_pauses(text)
            rows.append({"id": file_id, "text": text, "label": label, "medium": medium, "long": long})
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
            if label_row.empty: continue
            with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
                text = f.read().strip()
            medium, long = count_pauses(text)
            rows.append({"id": file_id, "text": text, "label": int(label_row["label"].values[0]), "medium": medium, "long": long})
    return pd.DataFrame(rows).dropna(subset=["text", "label"])

# ===== Fit Gaussian Models =====
def fit_gmms(df):
    control_data = df[df.label == 0][["medium", "long"]].values
    ad_data = df[df.label == 1][["medium", "long"]].values

    gmm_control = GaussianMixture(n_components=1, random_state=42).fit(control_data)
    gmm_ad = GaussianMixture(n_components=1, random_state=42).fit(ad_data)
    return gmm_control, gmm_ad

def compute_ad_prob(df, gmm_control, gmm_ad):
    feats = df[["medium", "long"]].values
    log_p_ad = gmm_ad.score_samples(feats)
    log_p_control = gmm_control.score_samples(feats)
    prob_ad = 1 / (1 + np.exp(log_p_control - log_p_ad))
    df["pause_prob_ad"] = prob_ad
    return df

# ===== Prompt Dataset Creation =====
def create_prompt_dataset(df, tokenizer, max_len=512):
    dataset = []
    for i, row in df.iterrows():
        text = row["text"]
        pause_score = row["pause_prob_ad"]
        pause_text = f"The probability of Alzheimer's Disease is {pause_score:.2f}"
        tokenized = tokenizer(text, truncation=True, max_length=max_len - 50, add_special_tokens=False)
        truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        dataset.append(InputExample(guid=i, text_a=truncated_text, text_b=pause_text, label=int(row["label"])))
    return dataset


# ===== Prompt-Based Fine-Tuning (PBFT) =====
def run_pbft(seed, train_df, val_df, test_df):
    if model_used == "ensemble":
        model_configs = [
            ("bert", "bert-large-uncased"),
            ("roberta", "roberta-large")
        ]
    else:
        model_configs = [(model_used, f"{model_used}-large")]

    run = wandb.init(
        entity="kj1080-university-of-cambridge",
        project="mlmi_thesis",
        name=f"pbft__{model_used}_pauseprob_seed_{seed}",
        config={"seed": seed, "learning_rate": 1e-5, "model": model_configs, "epochs": num_epochs},
        settings=wandb.Settings(_disable_stats=True),
        resume=False
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # templates = ['The diagnosis result is {"mask"}. Based on pauses features, {"placeholder":"text_b"}. Transcript: {"placeholder":"text_a"}.', 'Based on pauses features, {"placeholder":"text_b"}. Transcript: {"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
    templates = ['The diagnosis result is {"mask"}. {"placeholder":"text_b"}. {"placeholder":"text_a"}.', '{"placeholder":"text_b"}. {"placeholder":"text_a"}. The diagnosis result is {"mask"}.']

    all_preds_models = []
    all_labels_final = None

    for plm_type, model_path in model_configs:
        for template_text in templates:
            plm, tokenizer, model_config, WrapperClass = load_plm(plm_type, model_path)

            template = ManualTemplate(text=template_text, tokenizer=tokenizer)
            verbalizer = ManualVerbalizer(classes=["alzheimer", "healthy"],
                                          label_words={"alzheimer": ["alzheimer's"], "healthy": ["healthy"]},
                                          tokenizer=tokenizer)

            train_loader = PromptDataLoader(dataset=create_prompt_dataset(train_df, tokenizer), template=template,
                                            tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
                                            max_seq_length=512, batch_size=1, shuffle=True, num_workers=0)

            val_loader = PromptDataLoader(dataset=create_prompt_dataset(val_df, tokenizer), template=template,
                                          tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
                                          max_seq_length=512, batch_size=1, shuffle=False, num_workers=0)

            test_loader = PromptDataLoader(dataset=create_prompt_dataset(test_df, tokenizer), template=template,
                                           tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass,
                                           max_seq_length=512, batch_size=1, shuffle=False, num_workers=0)

            model = PromptForClassification(template=template, plm=plm, verbalizer=verbalizer).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer = AdamW(model.parameters(), lr=1e-5)

            epoch_logits, epoch_val_acc = [], []
            for epoch in tqdm(range(num_epochs), desc=f"{model_path}"):
                model.train()
                for i, batch in enumerate(train_loader):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    batch_inputs = {k: v for k, v in batch.items() if k != 'label'}
                    logits = model(batch_inputs)
                    loss = torch.nn.CrossEntropyLoss()(logits, batch['label']) / 4
                    loss.backward()
                    if (i + 1) % 4 == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        val_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                        val_labels.extend(batch['label'].cpu().numpy())
                val_acc = accuracy_score(val_labels, np.concatenate(val_preds))
                epoch_val_acc.append((val_acc, epoch))

                # Test
                preds_this_epoch = []
                all_labels_final = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        preds_this_epoch.append(torch.softmax(logits, dim=-1).cpu().numpy())
                        all_labels_final.extend(batch['label'].cpu().numpy())
                epoch_logits.append(np.vstack(preds_this_epoch))

            top_epochs = sorted(epoch_val_acc, reverse=True)[:top_n_epochs]
            for _, best_epoch in top_epochs:
                logits_stack = epoch_logits[best_epoch]
                preds = np.argmax(logits_stack, axis=1)
                acc_single = accuracy_score(all_labels_final, preds)

                if model_used == "ensemble":
                    if plm_type == "bert":
                        bert_accuracies.append(acc_single)
                    elif plm_type == "roberta":
                        roberta_accuracies.append(acc_single)

                    # Normalize and store logits
                logits_stack_norm = (logits_stack - logits_stack.mean(1, keepdims=True)) / (logits_stack.std(1, keepdims=True) + 1e-8)
                all_preds_models.append(logits_stack_norm)

    all_preds_models_np = np.array(all_preds_models)
    if majority_voting:
        votes = np.argmax(all_preds_models_np, axis=2)
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
    else:
        final_preds = np.argmax(np.mean(all_preds_models_np, axis=0), axis=1)

    acc = accuracy_score(all_labels_final, final_preds)
    wandb.log({"test_accuracy": acc})
    print(f"Seed {seed} Accuracy: {acc:.4f}")
    run.finish()
    return acc

# ===== Main =====
def main():
    train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR)
    test_df = load_test_set()

    gmm_control, gmm_ad = fit_gmms(train_df_full)
    train_df_full = compute_ad_prob(train_df_full, gmm_control, gmm_ad)
    test_df = compute_ad_prob(test_df, gmm_control, gmm_ad)

    seeds = range(42, 57)
    accuracies = []
    for seed in seeds:
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

if __name__ == "__main__":
    main()
