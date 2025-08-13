import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
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
from peft import get_peft_model, TaskType


# =========== Settings ==============

moe_in_attention = True
pause_encoding = True
transcript_type = "manual" # ASR_both_speakers, ASR_participant_only, manual, ASR_both_speakers_PE_medium
fine_tuned = False
prompt_type = "just_question_first_org" # both_original or both_v1 or different_BERT-V1_RoBERTa-Original, both_v1, just_question_first_org
model_used = "bert"
plm_size = "base"
question_first = "late_fusion"
num_epochs = 5
top_n_epochs = 1
weighted_majority_voting = True

# ========================================


def replace_attention_with_moe(model, d_model, n_heads, num_experts):
    for name, module in model.named_modules():
        if isinstance(module, (transformers.models.bert.modeling_bert.BertSelfAttention,
                               transformers.models.roberta.modeling_roberta.RobertaSelfAttention)):
            parent = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent, name.rsplit('.', 1)[1], MoEAttention(d_model, n_heads, num_experts))


class MoEAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_experts, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.k_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.v_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])

        # self.gate = nn.Sequential(
        #     nn.Linear(1280, 128),  # adjust if d_model = 1280
        #     nn.ReLU(),
        #     nn.Linear(128, num_experts)
        # )
        
        self.gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )


        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, pause_feats=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        x = hidden_states
        mask = attention_mask
        B, T, C = x.size()

        pooled = hidden_states.mean(dim=1)  # (B, d_model)
        gating_logits = self.gate(pooled)   # Only content used

        # if pause_feats is not None:
        #     pause_feats = pause_feats.view(B, -1)  # Ensure (B, 3)
        #     combined = torch.cat([pooled, pause_feats], dim=-1)  # (B, d_model+3)
        # else:
        #     combined = pooled

        # gating_logits = self.gate(combined)
        self.last_gating = torch.softmax(gating_logits, dim=-1).detach().cpu()
        gating_weights = torch.softmax(gating_logits, dim=-1)

        q = sum(w.unsqueeze(1).unsqueeze(2) * expert(x) for w, expert in zip(gating_weights.T, self.q_experts))
        k = sum(w.unsqueeze(1).unsqueeze(2) * expert(x) for w, expert in zip(gating_weights.T, self.k_experts))
        v = sum(w.unsqueeze(1).unsqueeze(2) * expert(x) for w, expert in zip(gating_weights.T, self.v_experts))

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), None

if pause_encoding == False:
    if transcript_type == "ASR_both_speakers":
        if fine_tuned == False:
            TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_non_ft/train"
            TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_non_ft/test"
    elif transcript_type == "ASR_participant_only":
            TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only/train"
            TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only/test"
    elif transcript_type == "ASR_participant_only_PE":
        TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only_PE/train"
        TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only_PE/test"

elif pause_encoding:
    if transcript_type == "manual":
        TRANSCRIPT_DIR = "pause_encoding/train"
        TRANSCRIPT_TEST_DIR = "pause_encoding/test"
    elif transcript_type == 'ASR_both_speakers':
        TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_only_PE/train"
        TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_only_PE/test"

TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"

# TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_only_PE/train"
# TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_only_PE/test"

CONTROL_IDS = {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011", "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034", "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049", "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063", "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"}

bert_accuracies = []
roberta_accuracies = []

print('Using Transcripts from:')
print(TRANSCRIPT_DIR)

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
        if plm_size == "large":
            model_configs = [
                ("bert", "bert-large-uncased"),
                ("roberta", "roberta-large")
            ]
        elif plm_size == "base":
            model_configs = [
                ("bert", "bert-base-uncased"),
                ("roberta", "roberta-base")
            ]
    elif model_used == "bert":
        if plm_size == "large":
            model_configs = [("bert", "bert-large-uncased")]
        elif plm_size == "base":
            model_configs = [("bert", "bert-base-uncased")]
    elif model_used == "roberta":
        if plm_size == "base":
            model_configs = [("roberta", "roberta-large")]
        elif plm_size == "base":
            model_configs = [("roberta", "roberta-base")]

    # run = wandb.init(
    #     entity="kj1080-university-of-cambridge",
    #     project="mlmi_thesis",
    #     name=f"pbft__{model_used}_pauseprob_seed_{seed}",
    #     config={"seed": seed, "learning_rate": 1e-5, "model": model_configs, "epochs": num_epochs},
    #     settings=wandb.Settings(_disable_stats=True),
    #     resume=False
    # )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if prompt_type == "different_BERT-V1_RoBERTa-Original":
        template_bert = ['Based on their speech patterns within this transcript: {"placeholder":"text_a"} , this individual is {"mask"}.', 'This individual is {"mask"}. This is based on their speech patterns within this transcript: {"placeholder":"text_a"}.']
        template_roberta = ['The diagnosis result is {"mask"}. {"placeholder":"text_a"}.', '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
    elif prompt_type == "both_original":
        template_bert = ['The diagnosis result is {"mask"}. {"placeholder":"text_a"}.', '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
        template_roberta = ['The diagnosis result is {"mask"}. {"placeholder":"text_a"}.', '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
    elif prompt_type == "both_v1":
        template_bert = ['Based on their speech patterns within this transcript: {"placeholder":"text_a"} , this individual is {"mask"}.', 'This individual is {"mask"}. This is based on their speech patterns within this transcript: {"placeholder":"text_a"}.']
        template_roberta = ['Based on their speech patterns within this transcript: {"placeholder":"text_a"} , this individual is {"mask"}.', 'This individual is {"mask"}. This is based on their speech patterns within this transcript: {"placeholder":"text_a"}.']
    elif prompt_type == "just_question_first_org":
        template_bert = ['{"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
        template_roberta = template_bert
    
    all_preds_models = []
    all_labels_final = None
    expert_gating_summary = []
    # all_labels_final = []
    val_weights = [] 

    for plm_type, model_path in model_configs:
        if plm_type == "bert":
            model_templates = template_bert
        elif plm_type == "roberta":
            model_templates = template_roberta
        else:
            raise ValueError(f"Unknown model type: {plm_type}")
        
        for template_text in model_templates:

            plm, tokenizer, model_config, WrapperClass = load_plm(plm_type, model_path)
                
            if 'bert' in model_path or 'roberta' in model_path:
                if plm_size == 'base':
                    replace_attention_with_moe(plm, d_model=768, n_heads=12, num_experts=4)  # Adjust for 'base'; for 'large', use d_model=1024, n_heads=16
                elif plm_size == 'large':
                    replace_attention_with_moe(plm, d_model=1024, n_heads=16, num_experts=4) 

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
            optimizer = AdamW(model.parameters(), lr=5e-5)

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
                expert_weights = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        preds_this_epoch.append(torch.softmax(logits, dim=-1).cpu().numpy())
                        all_labels_final.extend(batch['label'].cpu().numpy())
                        
                        # Access expert gating
                        moe_layers = [m for m in model.plm.modules() if hasattr(m, "last_gating")]
                        if moe_layers:
                            expert_weights.append(moe_layers[0].last_gating.numpy())
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
                val_weights.append(acc_single)


    all_preds_models_np = np.array(all_preds_models)
    val_weights_np = np.array(val_weights)  # Shape: [num_models]
    val_weights_np /= val_weights_np.sum()  # Normalize to sum to 1
    if weighted_majority_voting:
        assert val_weights_np.shape[0] == all_preds_models_np.shape[0], f"Mismatch: val_weights={val_weights_np.shape[0]}, preds={all_preds_models_np.shape[0]}"

        weighted_logits = np.tensordot(val_weights_np, all_preds_models_np, axes=([0], [0]))  # Shape: [num_samples, num_classes]
        final_preds = np.argmax(weighted_logits, axis=1)
        
    print("Ensemble prediction distribution:", np.mean(all_preds_models_np, axis=0).mean(axis=0))

    
    # Final ensemble prediction 
    acc = accuracy_score(all_labels_final, final_preds)

    # Get per-model best accuracy if ensemble
    if model_used == "ensemble":
        bert_acc = np.max(bert_accuracies[-top_n_epochs:]) if bert_accuracies else None
        roberta_acc = np.max(roberta_accuracies[-top_n_epochs:]) if roberta_accuracies else None

        print(f"Seed {seed} BERT Accuracy     : {bert_acc:.4f}" if bert_acc is not None else "No BERT accuracy available.")
        print(f"Seed {seed} RoBERTa Accuracy : {roberta_acc:.4f}" if roberta_acc is not None else "No RoBERTa accuracy available.")
        print(f"Seed {seed} Ensemble Accuracy: {acc:.4f}")
    else:
        print(f"Seed {seed} Accuracy: {acc:.4f}")

    # wandb.log({"test_accuracy": acc})
    
    # run.finish()
    return acc

# ===== Main =====
def main():
    train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR)
    test_df = load_test_set()

    gmm_control, gmm_ad = fit_gmms(train_df_full)
    train_df_full = compute_ad_prob(train_df_full, gmm_control, gmm_ad)
    test_df = compute_ad_prob(test_df, gmm_control, gmm_ad)

    seeds = range(42, 43)
    accuracies = []
    for seed in seeds:
        train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)
        acc = run_pbft(seed, train_df, val_df, test_df)
        accuracies.append(acc)

    print(f"\n========== PBFT Summary Over {len(seeds)} Seeds ==========")
    print(f"Mean Accuracy     : {np.mean(accuracies):.4f}")
    print(f"Std Deviation     : {np.std(accuracies):.4f}")
    print(f"Max Accuracy      : {np.max(accuracies):.4f}")
    
    if model_used == "ensemble":
        print("\n------ Individual Model Summary ------")
        if bert_accuracies:
            print(f"[BERT]    Mean: {np.mean(bert_accuracies):.4f}, Std: {np.std(bert_accuracies):.4f}, Max: {np.max(bert_accuracies):.4f}")
            print(bert_accuracies)
        else:
            print("[BERT]    No results recorded.")

        if roberta_accuracies:
            print(f"[RoBERTa] Mean: {np.mean(roberta_accuracies):.4f}, Std: {np.std(roberta_accuracies):.4f}, Max: {np.max(roberta_accuracies):.4f}")
            print(roberta_accuracies)
        else:
            print("[RoBERTa] No results recorded.")

if __name__ == "__main__":
    main()
