import os
import re
import torch
import random
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse
import shutil
from pathlib import Path
import hashlib
from collections import defaultdict
import collections
import matplotlib.pyplot as plt
from itertools import combinations


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from transformers import AdamW
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from peft import get_peft_model, LoraConfig, TaskType
from spanning import segment_text, average_logits
import torch.nn.functional as F


# =========== Path Settings ==============

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "sweep"], default="normal", help="Run mode: normal or sweep")
    parser.add_argument("--bert_lora", action="store_true", default="False")
    parser.add_argument("--roberta_lora", action="store_true", default="False")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--lambda_div", type=float, default=0.02)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--lambda_ortho", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=56)
    return parser.parse_args()



args = parse_args()

# Early stopping ON only for sweep mode
if args.mode == "sweep":
    early_stopping = True
    patience = 5   # Or set via wandb.config if you want to sweep this too
else:
    early_stopping = False
    patience = 20   # Can be ignored if not in sweep mode

all_mean_cos_sims = []
all_gating_entropies = []
all_diversity_indices = []


segment_expert_trigger = False#  True indicates 1 expert is triggered per segment False, and 1 expert per token
span = True
bert_lora = args.bert_lora
roberta_lora = args.roberta_lora
pause_encoding = True
transcript_type = "ASR_both_speakers_PE_medium" # ASR_both_speakers, ASR_participant_only, manual, ASR_both_speakers_PE_medium
fine_tuned = False
prompt_type = "both_v1" # both_original or both_v1 or different_BERT-V1_RoBERTa-Original, both_v1, just_question_first_org
model_used = "ensemble"
plm_size = "base"
num_epochs = 3
top_n_epochs = 3
weighted_majority_voting = True

# ========================================

if bert_lora or roberta_lora:
    from MOE_LORA import LoRALinear, MoEAttentionProjection, MoEBertSelfAttention, MoERobertaSelfAttention
else:
    from MOE import MoEAttentionProjection, MoEBertSelfAttention, MoERobertaSelfAttention


if pause_encoding == False:
    if transcript_type == "ASR_both_speakers":
        if fine_tuned == False:
            TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_non_ft/train"
            TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_non_ft/test"
    elif transcript_type == "ASR_participant_only":
            TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only/test"
    elif transcript_type == "ASR_participant_only_PE":
        TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only_PE/train"
        TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_participant_only_PE/test"

elif pause_encoding:
    if transcript_type == "manual":
        TRANSCRIPT_DIR = "pause_encoding/train"
        TRANSCRIPT_TEST_DIR = "pause_encoding/test"
    elif transcript_type == "ASR_both_speakers_PE_medium":
        TRANSCRIPT_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_PE_medium/train"
        TRANSCRIPT_TEST_DIR = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_PE_medium/test"
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

def apply_lora(plm, config, model_type):
    if model_type == "bert":
        if not config.bert_lora:
            return plm
    elif model_type == "roberta":
        if not config.roberta_lora:
            return plm
    else:
        # If you ever add more models, handle them here
        return plm
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(config.lora_r),
        lora_alpha=float(config.lora_alpha),
        lora_dropout=float(config.lora_dropout)
    )
    return get_peft_model(plm, lora_cfg)


def gating_entropy_loss(gate_weights):
    """
    Computes mean entropy over the gating softmax weights.
    gate_weights: [batch, seq_len, num_experts]
    """
    # Add a small epsilon for numerical stability
    probs = gate_weights.clamp(min=1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)  # sum over experts, shape: [batch, seq_len]
    return entropy.mean()  # scalar


def activation_entropy(expert_outputs):
    # expert_outputs: [seq_len, num_experts, hidden] or [batch, seq_len, num_experts, hidden]

    if expert_outputs.dim() == 4:
        expert_outputs = expert_outputs.squeeze(0)
    # Take mean across hidden dim, softmax over experts
    tokenwise_mean = expert_outputs.mean(dim=-1)  # [seq_len, num_experts]
    probs = F.softmax(tokenwise_mean, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [seq_len]
    return entropy.mean().item()


# ===== Pause Counting =====
def count_pauses(text):
    long = len(re.findall(r"\.\.\.", text))
    medium = len(re.findall(r"(?<!\.)\.(?!\.)", text))
    return medium, long

def log_token_heatmap(df, set_name, model_name, run_name):
    fig_height = max(6, 0.25 * len(df))  # 0.25 inches per token; adjust as needed
    fig, ax = plt.subplots(figsize=(8, fig_height))
    sns.heatmap(df, annot=True, fmt='d', ax=ax, cmap="YlGnBu")
    ax.set_title(f"{run_name}\nExpert Assignment for Top Differentiating Tokens ({set_name} Set)")
    plt.tight_layout()
    wandb.log({f"{set_name.lower()}_top_tokens_heatmap_{model_name}": wandb.Image(fig)})
    plt.close(fig)

def compute_mean_cosine_similarity(moe_modules):
    all_sims = []
    for moe_proj in moe_modules:
        # Assume moe_proj.expert_weights is shape [num_experts, hidden, ...]
        # You may need to adapt this depending on how weights are stored!
        weights = moe_proj.experts.weight.data.cpu().numpy()  # e.g. [num_experts, hidden, hidden] or [num_experts, hidden]
        num_experts = weights.shape[0]
        weights_flat = weights.reshape(num_experts, -1)  # Flatten for cosine
        # Pairwise cosine similarity
        for i, j in combinations(range(num_experts), 2):
            w1 = weights_flat[i]
            w2 = weights_flat[j]
            cos_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8)
            all_sims.append(cos_sim)
    return np.mean(all_sims)

def compute_gating_entropy_over_dataset(model, data_loader):
    entropies = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            _ = model({k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items() if k != 'label'})
            moe_modules = [m for m in model.modules() if isinstance(m, MoEAttentionProjection)]
            for moe_proj in moe_modules:
                if moe_proj._last_gate_weights is not None:
                    # Shape [batch, seq_len, num_experts]
                    gate_weights = moe_proj._last_gate_weights
                    ent = gating_entropy_loss(gate_weights)
                    entropies.append(ent.item())
    return np.mean(entropies)

def compute_expert_assignment_diversity(expert_usage_counts):
    # expert_usage_counts: shape [num_experts] (counts of assignments to each expert)
    proportions = np.array(expert_usage_counts) / (np.sum(expert_usage_counts) + 1e-8)
    diversity_index = 1.0 - np.max(proportions)  # Higher is more diverse
    return diversity_index

# ===== Load Transcripts =====
def load_transcripts_with_pauses(transcript_dir, is_test=False, span=False):
    rows = []
    for fname in os.listdir(transcript_dir):
        if not fname.endswith(".txt"):
            continue
        file_id = os.path.splitext(fname)[0].upper()
        label = None if is_test else (0 if file_id in CONTROL_IDS else 1)
        with open(os.path.join(transcript_dir, fname), "r") as f:
            text = f.read().strip()

        if span:
            segments = segment_text(text)
            for i, seg in enumerate(segments):
                seg_id = f"{file_id}_seg{i}"
                medium, long = count_pauses(seg)
                rows.append({"id": seg_id, "text": seg, "label": label, "medium": medium, "long": long})
                
        else:
            medium, long = count_pauses(text)
            rows.append({"id": file_id, "text": text, "label": label, "medium": medium, "long": long})
    # print(f"[DEBUG] {file_id} split into {len(segments)} segments.")
    return pd.DataFrame(rows)

def clean_token(tok):
    if isinstance(tok, str):
        # Remove BPE artifacts and make special tokens readable
        return tok.replace("Ġ", "").replace("Ċ", "").replace("<pad>", "[PAD]").replace("</s>", "[SEP]")
    return str(tok)

def hash_template(template_text):
    return hashlib.md5(template_text.encode()).hexdigest()[:6]


def load_test_set(span=False):
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
                print(f"[WARNING] No label for {file_id} in test set. Skipping.")
                continue

            with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
                text = f.read().strip()

            if span:
                segments = segment_text(text)
                for i, seg in enumerate(segments):
                    seg_id = f"{file_id}_seg{i}"
                    medium, long = count_pauses(seg)
                    rows.append({
                        "id": seg_id,
                        "text": seg,
                        "label": int(label_row["label"].values[0]),
                        "medium": medium,
                        "long": long
                    })
            else:
                medium, long = count_pauses(text)
                rows.append({
                    "id": file_id,
                    "text": text,
                    "label": int(label_row["label"].values[0]),
                    "medium": medium,
                    "long": long
                })
    return pd.DataFrame(rows).dropna(subset=["text", "label"])


# ===== Fit Gaussian Models =====
def fit_gmms(df):
    control_data = df[df.label == 0][["medium", "long"]].values
    ad_data = df[df.label == 1][["medium", "long"]].values

    gmm_control = GaussianMixture(n_components=1, random_state=47).fit(control_data)
    gmm_ad = GaussianMixture(n_components=1, random_state=47).fit(ad_data)
    return gmm_control, gmm_ad

def compute_ad_prob(df, gmm_control, gmm_ad):
    feats = df[["medium", "long"]].values
    log_p_ad = gmm_ad.score_samples(feats)
    log_p_control = gmm_control.score_samples(feats)
    prob_ad = 1 / (1 + np.exp(log_p_control - log_p_ad))
    df["pause_prob_ad"] = prob_ad
    return df

# ===== Prompt Dataset Creation =====
def create_prompt_dataset(df, tokenizer, max_len=512, span=False):
    dataset = []
    for i, row in df.iterrows():
        text = row["text"]
        pause_score = row["pause_prob_ad"]
        pause_text = f"The probability of Alzheimer's Disease is {pause_score:.2f}"
        tokenized = tokenizer(text, truncation=True, max_length=max_len - 50, add_special_tokens=False)
        truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)

        guid = row["id"] if span else i  # Use meaningful ID if spanning, else fallback to index
        dataset.append(InputExample(guid=guid, text_a=truncated_text, text_b=pause_text, label=int(row["label"])))
    return dataset

def get_token_df(token_expert_counts, tokenizer, num_experts, top_n=100):
    token_diff_scores = {}
    for tok_id, expert_counts in token_expert_counts.items():
        counts = [expert_counts[e] for e in range(num_experts)]
        total = sum(counts)
        if total < 20: continue
        probs = np.array(counts) / total
        score = probs.max() - probs.min()
        token_diff_scores[tok_id] = score
    top_tokens = sorted(token_diff_scores.items(), key=lambda x: -x[1])[:top_n]
    try:
        id2tok = {v: k for k, v in tokenizer.get_vocab().items()}
    except Exception:
        id2tok = {i: str(i) for i in range(50000)}
    token_labels = [clean_token(id2tok.get(tok_id, str(tok_id))) for tok_id, _ in top_tokens]
    data = []
    for tok_id, _ in top_tokens:
        row = [token_expert_counts[tok_id][e] for e in range(num_experts)]
        data.append(row)
    return token_labels, data

def make_dfs(token_labels, data, num_experts):
    df_all = pd.DataFrame(data, index=token_labels, columns=[f"Expert {i}" for i in range(num_experts)])
    exclude_tokens = {"<pad>", "[PAD]", "</s>", "[SEP]"}
    filtered = [(label, row) for label, row in zip(token_labels, data) if label not in exclude_tokens]
    if filtered:
        labels_no_pad, data_no_pad = zip(*filtered)
        df_no_pad = pd.DataFrame(data_no_pad, index=labels_no_pad, columns=[f"Expert {i}" for i in range(num_experts)])
    else:
        df_no_pad = pd.DataFrame([], columns=[f"Expert {i}" for i in range(num_experts)])
    return df_all, df_no_pad

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
        if plm_size == "large":
            model_configs = [("roberta", "roberta-large")]
        elif plm_size == "base":
            model_configs = [("roberta", "roberta-base")]

    run = wandb.init(
        entity="kj1080-university-of-cambridge",
        project="mlmi_thesis",
        name=f"bert_lora_{bert_lora}_roberta_lora_{roberta_lora}_{model_used}_seed_{seed}",
        config={
            "seed": seed,
            "bert_lora": bert_lora,
            "roberta_lora": roberta_lora,
            "plm": model_used,
            "plm_size": plm_size,
            "prompt_type": prompt_type,
            "epochs": num_epochs,
            "learning_rate": 1e-5, # 4e-5 if using LORA, 1e-5 if not
            "lora_r": 1,
            "lora_alpha": 512,
            "lora_dropout": 0.01,
            "num_experts": 4,
            "top_k": 1
        },
        settings=wandb.Settings(_disable_stats=True),
        resume=False
    )

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
    elif prompt_type == "minimal":
        template_bert = ['{"mask"}. {"placeholder":"text_a"}.', '{"placeholder":"text_a"}. {"mask"}.']
        template_roberta = ['{"mask"}. {"placeholder":"text_a"}.', '{"placeholder":"text_a"}. {"mask"}.']

    all_preds_models = []
    all_labels_final = []
    val_weights = [] 
    span_ids = []
    span_logits = []
    global_step = 0
    # Map: token string -> array of expert assignment counts, shape [num_experts]
    token_expert_counts = defaultdict(lambda: np.zeros(num_experts))
    token_label_counts = defaultdict(lambda: {0: np.zeros(num_experts), 1: np.zeros(num_experts)})  # If you want by diagnosis

    for plm_type, model_path in model_configs:
        if plm_type == "bert":
            model_templates = template_bert
        elif plm_type == "roberta":
            model_templates = template_roberta
        else:
            raise ValueError(f"Unknown model type: {plm_type}")

        for template_text in model_templates:
            epoch_logits, epoch_val_acc = [], []
            plm, tokenizer, model_config, WrapperClass = load_plm(plm_type, model_path)
            num_experts = wandb.config.num_experts
            top_k = wandb.config.top_k
            # lambda_ortho = wandb.config.lambda_ortho
            # lambda_div = wandb.config.lambda_div
            # lambda_entropy = wandb.config.lambda_entropy
            # learning_rate = wandb.config.learning_rate
            
            if (plm_type == "bert" and bert_lora) or (plm_type == "roberta" and roberta_lora):
            # if lora:
                if plm_type == "bert":
                    for name, module in plm.named_modules():
                        if isinstance(module, BertSelfAttention):
                            parent_name = name.rsplit(".", 1)[0]
                            attr_name = name.rsplit(".", 1)[-1]
                            parent_module = dict(plm.named_modules())[parent_name]
                            setattr(
                                parent_module,
                                attr_name,
                                MoEBertSelfAttention(
                                    config=model_config,
                                    num_experts=num_experts,
                                    k=top_k,
                                    use_lora=wandb.config.bert_lora,
                                    lora_r=wandb.config.lora_r,
                                    lora_alpha=wandb.config.lora_alpha,
                                    lora_dropout=wandb.config.lora_dropout,
                                    segment_expert_trigger=segment_expert_trigger
                                )
                            )
                elif plm_type == "roberta":
                    for name, module in plm.named_modules():
                        if isinstance(module, RobertaSelfAttention):
                            parent_name = name.rsplit(".", 1)[0]
                            attr_name = name.rsplit(".", 1)[-1]
                            parent_module = dict(plm.named_modules())[parent_name]
                            setattr(
                                parent_module,
                                attr_name,
                                MoERobertaSelfAttention(
                                    config=model_config,
                                    num_experts=num_experts,
                                    k=top_k,
                                    use_lora=wandb.config.roberta_lora,
                                    lora_r=wandb.config.lora_r,
                                    lora_alpha=wandb.config.lora_alpha,
                                    lora_dropout=wandb.config.lora_dropout,
                                    segment_expert_trigger=segment_expert_trigger
                                )
                            )

            else:
                
                if plm_type == "bert":
                    for name, module in plm.named_modules():
                        if isinstance(module, BertSelfAttention):
                            parent_name = name.rsplit(".", 1)[0]
                            attr_name = name.rsplit(".", 1)[-1]
                            parent_module = dict(plm.named_modules())[parent_name]
                            setattr(parent_module, attr_name, MoEBertSelfAttention(config=model_config, num_experts=num_experts, k=top_k, segment_expert_trigger=segment_expert_trigger))
                elif plm_type == "roberta":
                    for name, module in plm.named_modules():
                        if isinstance(module, RobertaSelfAttention):
                            parent_name = name.rsplit(".", 1)[0]
                            attr_name = name.rsplit(".", 1)[-1]
                            parent_module = dict(plm.named_modules())[parent_name]
                            setattr(parent_module, attr_name, MoERobertaSelfAttention(config=model_config, num_experts=num_experts, k=top_k, segment_expert_trigger=segment_expert_trigger))
            
            if (plm_type == "bert" and bert_lora) or (plm_type == "roberta" and roberta_lora):
                plm = apply_lora(plm, wandb.config, plm_type)
                
                for name, param in plm.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for param in plm.parameters():
                    param.requires_grad = True
                    
            total_params = sum(p.numel() for p in plm.parameters())
            trainable_params = sum(p.numel() for p in plm.parameters() if p.requires_grad)
            print(f"Trainable / Total params: {trainable_params} / {total_params}")

            # Define template and verbalizer
            template = ManualTemplate(text=template_text, tokenizer=tokenizer)
            verbalizer = ManualVerbalizer(
                classes=["alzheimer", "healthy"],
                label_words={"alzheimer": ["alzheimer's"], "healthy": ["healthy"]},
                tokenizer=tokenizer
            )

            # Prepare data loaders
            train_loader = PromptDataLoader(
                dataset=create_prompt_dataset(train_df, tokenizer, span=span),  # <-- Added span=span
                template=template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=512,
                batch_size=1,
                shuffle=True,
                num_workers=0
            )

            val_loader = PromptDataLoader(
                dataset=create_prompt_dataset(val_df, tokenizer, span=span),  # <-- Added span=span
                template=template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=512,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )

            test_loader = PromptDataLoader(
                dataset=create_prompt_dataset(test_df, tokenizer, span=span),  # <-- Added span=span
                template=template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=512,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )


            # Wrap PLM in a prompt model
            model = PromptForClassification(template=template, plm=plm, verbalizer=verbalizer).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            optimizer = AdamW(model.parameters(), lr=float(wandb.config.learning_rate))
            if early_stopping:
                best_val_loss = float('inf')
                epochs_without_improve = 0
            
            for epoch in tqdm(range(num_epochs), desc=f"{model_path}"):
                expert_usage_counts = np.zeros(num_experts, dtype=int)  # <- Reset each epoch

                start_temp = 2.0
                end_temp = 0.5
                current_temp = start_temp + (end_temp - start_temp) * (epoch / (num_epochs - 1))

                # Set temperature for all MoEAttentionProjection modules:
                for moe_proj in model.modules():
                    if isinstance(moe_proj, MoEAttentionProjection):
                        moe_proj.set_temperature(current_temp)
                print(f"[Epoch {epoch}] Set MoE temperature to {current_temp:.3f}")

                model.train()
                for i, batch in enumerate(train_loader):
                    batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                    batch_inputs = {k: v for k, v in batch.items() if k != 'label'}

                    # Forward pass and get logits
                    logits = model(batch_inputs)

                    # Base loss
                    loss = torch.nn.CrossEntropyLoss()(logits, batch['label']) / 4

                    # === REGULARIZATION LOSS (for MoE) ===
                    lambda_ortho = 0.8
                    lambda_div = 0.05
                    lambda_entropy = 0.08
                    moe_losses = 0.0
                    entropy_loss_total = 0.0

                    # Loop over all MoEAttentionProjection modules in the model
                    moe_modules = [m for m in model.modules() if isinstance(m, MoEAttentionProjection)]

                    for moe_proj in moe_modules:
                        moe_losses += lambda_ortho * moe_proj.orthogonality_loss()
                        moe_losses += lambda_div * moe_proj.output_diversity_loss()
                        if moe_proj._last_gate_weights is not None:
                            entropy_loss = gating_entropy_loss(moe_proj._last_gate_weights)
                            entropy_loss_total += entropy_loss
                            wandb.log({"gating_entropy_loss": entropy_loss.item()}, step=global_step)

                        # ==== Log expert assignments for this batch ====
                        if moe_proj._last_topk_indices is not None:
                            topk_indices = moe_proj._last_topk_indices.numpy()  # shape [batch, seq_len, k]
                            flat_experts = topk_indices.flatten()
                            for ex in flat_experts:
                                expert_usage_counts[ex] += 1

                    total_loss = loss + moe_losses + (lambda_entropy * entropy_loss_total)
                    x = lambda_entropy * entropy_loss_total
                    # print(f"total loss: {total_loss}")
                    # print(f"moe losses: {moe_losses}")
                    # print(f"lambda_entropy * entropy_loss_total: {x}")
                    total_loss.backward()

                    wandb.log({
                        "train_loss": total_loss.item(),
                        "moe_ortho_loss": sum([m.orthogonality_loss().item() for m in moe_modules]),
                        "moe_div_loss": sum([m.output_diversity_loss().item() for m in moe_modules])
                    }, step=global_step)

                    # Compute total gradient norm
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    wandb.log({"grad_norm": total_norm}, step=global_step)

                    if (i + 1) % 4 == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    global_step += 1

                # ===== Log expert distribution after epoch =====
                usage_total = expert_usage_counts.sum()
                proportions = expert_usage_counts / (usage_total + 1e-8)
                for ex_id, count in enumerate(expert_usage_counts):
                    wandb.log({f"expert_{ex_id}_count": count}, step=global_step)
                for ex_id, prop in enumerate(proportions):
                    wandb.log({f"expert_{ex_id}_proportion": prop}, step=global_step)
                print(f"[Epoch {epoch}] Expert assignment counts: {expert_usage_counts.tolist()}")
                print(f"[Epoch {epoch}] Expert assignment proportions: {proportions.tolist()}")


                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        # batch = {k: v.to(model.device) for k, v in batch.items()}
                        batch = {k: v.to(model.device) if hasattr(v, 'to') else v
                             for k, v in batch.items()}
                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        val_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                        val_labels.extend(batch['label'].cpu().numpy())
                val_acc = accuracy_score(val_labels, np.concatenate(val_preds))
                epoch_val_acc.append((val_acc, epoch))
                wandb.log({"val_accuracy": val_acc}, step=global_step)
                val_metric = -val_acc
                
                if early_stopping:
                    if val_metric < best_val_loss:
                        best_val_loss = val_metric
                        epochs_without_improve = 0
                    else:
                        epochs_without_improve += 1
                        if epochs_without_improve >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1} (no improvement in {patience} epochs)")
                            break

                # test
                preds_this_epoch = []
                labels_this_epoch = []
                with torch.no_grad():
                    for batch in test_loader:
                        # batch = {k: v.to(model.device) for k, v in batch.items()}
                        batch = {k: v.to(model.device) if hasattr(v, 'to') else v
                             for k, v in batch.items()}
                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        preds_this_epoch.append(probs)
                        # Assuming your test dataset rows have "id" column strings like "S108_seg0"
                        # Collect predictions and span-level metadata
                        batch_ids = batch.get("guid", batch.get("id"))  # Use "guid" if available, else fallback to "id"
                        span_ids.extend(batch_ids)
                        span_logits.extend(probs)
                        labels_this_epoch.extend(batch["label"].cpu().numpy())

                logits_epoch = np.vstack(preds_this_epoch)
                epoch_logits.append(logits_epoch)

                pred_classes = np.argmax(logits_epoch, axis=1)
                test_acc = accuracy_score(labels_this_epoch, pred_classes)

                wandb.log({"test_accuracy": test_acc}, step=global_step)

            top_epochs = sorted(epoch_val_acc, reverse=True)[:top_n_epochs]
            all_labels_final = labels_this_epoch 

            for _, best_epoch in top_epochs:
                print(f"Selected epoch {best_epoch} for model={plm_type}, template='{template_text[:60]}...' with val_acc={_:.4f}")

                logits_stack = epoch_logits[best_epoch]
                preds = np.argmax(logits_stack, axis=1)

                acc_single = accuracy_score(labels_this_epoch, preds)  # <-- now using local labels from this epoch


                if model_used == "ensemble":
                    if plm_type == "bert":
                        bert_accuracies.append(acc_single)
                    elif plm_type == "roberta":
                        roberta_accuracies.append(acc_single)

                    # Normalize and store logits
                # logits_stack_norm = (logits_stack - logits_stack.mean(1, keepdims=True)) / (logits_stack.std(1, keepdims=True) + 1e-8)
                # all_preds_models.append(logits_stack_norm)
                all_preds_models.append(logits_stack)
                val_weights.append(acc_single)
                
                
                
                                # =========== Expert Assignment Visualization (per token, per label) ===========
                token_expert_counts = collections.defaultdict(lambda: collections.Counter())
                label_expert_counts = collections.defaultdict(lambda: collections.Counter())
                
                
                

                print("Aggregating expert assignments by token and label (this may take a minute)...")

                data_loader = val_loader  # <-- Use validation set for analysis

                model.eval()
                with torch.no_grad():
                    for batch in data_loader:
                        input_ids = batch['input_ids'].cpu().numpy()
                        labels = batch['label'].cpu().numpy()
                        batch_size, seq_len = input_ids.shape
                        _ = model({k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items() if k != 'label'})
                        moe_modules = [m for m in model.modules() if isinstance(m, MoEAttentionProjection)]
                        for b in range(batch_size):
                            moe_proj = moe_modules[0]
                            if moe_proj._last_topk_indices is not None:
                                indices = moe_proj._last_topk_indices[b].cpu().numpy()
                                if indices.ndim == 2:
                                    indices = indices[:, 0]
                                for t in range(seq_len):
                                    tok_id = input_ids[b, t]
                                    if tok_id != 0:
                                        expert = indices[t]
                                        token_expert_counts[tok_id][expert] += 1
                                        label_expert_counts[labels[b]][expert] += 1

                
                # 1. Build DataFrames for validation and test expert assignments (w/ and w/o PAD)
                val_token_labels, val_token_data = get_token_df(token_expert_counts, tokenizer, num_experts)
                val_df_all, val_df_no_pad = make_dfs(val_token_labels, val_token_data, num_experts)
                
                run_name = wandb.run.name
                log_token_heatmap(val_df_all, "Val", plm_type, run_name)
                log_token_heatmap(val_df_no_pad, "Val_no_pad", plm_type, run_name)

                # -- Print for VAL (all tokens)
                print("\n[VAL] Top tokens (incl. PAD):")
                for label, row in zip(val_token_labels, val_token_data):
                    print(f"{label}: {row}")
                print("\n[VAL] Top tokens (excl. PAD):")
                if val_df_no_pad.index.size > 0:
                    for label, row in zip(val_df_no_pad.index, val_df_no_pad.values):
                        print(f"{label}: {row.tolist()}")
                else:
                    print("No non-PAD tokens with enough assignments.")

                # -- Wandb log for VAL (all tokens)
                wandb_table_val_all = wandb.Table(columns=["Token"] + [f"Expert {i}" for i in range(num_experts)])
                for label, row in zip(val_token_labels, val_token_data):
                    wandb_table_val_all.add_data(label, *row)
                wandb.log({"val_top_tokens_expert_assignments": wandb_table_val_all})

                # -- Wandb log for VAL (no PAD)
                wandb_table_val_no_pad = wandb.Table(columns=["Token"] + [f"Expert {i}" for i in range(num_experts)])
                if val_df_no_pad.index.size > 0:
                    for label, row in zip(val_df_no_pad.index, val_df_no_pad.values):
                        wandb_table_val_no_pad.add_data(label, *row)
                    wandb.log({"val_top_tokens_no_pad_expert_assignments": wandb_table_val_no_pad})
                    
                # and for test
                # =========== Expert Assignment Visualization for TEST ===========

                token_expert_counts_test = collections.defaultdict(lambda: collections.Counter())
                label_expert_counts_test = collections.defaultdict(lambda: collections.Counter())

                print("Aggregating expert assignments by token and label on TEST set...")

                data_loader_test = test_loader  # <-- Use test set for analysis

                model.eval()
                with torch.no_grad():
                    for batch in data_loader_test:
                        input_ids = batch['input_ids'].cpu().numpy()
                        labels = batch['label'].cpu().numpy()
                        batch_size, seq_len = input_ids.shape
                        _ = model({k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items() if k != 'label'})
                        moe_modules = [m for m in model.modules() if isinstance(m, MoEAttentionProjection)]
                        for b in range(batch_size):
                            moe_proj = moe_modules[0]
                            if moe_proj._last_topk_indices is not None:
                                indices = moe_proj._last_topk_indices[b].cpu().numpy()
                                if indices.ndim == 2:
                                    indices = indices[:, 0]
                                for t in range(seq_len):
                                    tok_id = input_ids[b, t]
                                    if tok_id != 0:
                                        expert = indices[t]
                                        token_expert_counts_test[tok_id][expert] += 1
                                        label_expert_counts_test[labels[b]][expert] += 1

                # Compute token-level expert stats for TEST
                test_token_labels, test_token_data = get_token_df(token_expert_counts_test, tokenizer, num_experts)
                test_df_all, test_df_no_pad = make_dfs(test_token_labels, test_token_data, num_experts)
                log_token_heatmap(test_df_all, "Test", plm_type, run_name)
                log_token_heatmap(test_df_no_pad, "Test_no_pad", plm_type, run_name)

                # Print for TEST (all tokens)
                print("\n[TEST] Top tokens (incl. PAD):")
                for label, row in zip(test_token_labels, test_token_data):
                    print(f"{label}: {row}")
                print("\n[TEST] Top tokens (excl. PAD):")
                if test_df_no_pad.index.size > 0:
                    for label, row in zip(test_df_no_pad.index, test_df_no_pad.values):
                        print(f"{label}: {row.tolist()}")
                else:
                    print("No non-PAD tokens with enough assignments.")

                # Wandb log for TEST (all tokens)
                wandb_table_test_all = wandb.Table(columns=["Token"] + [f"Expert {i}" for i in range(num_experts)])
                for label, row in zip(test_token_labels, test_token_data):
                    wandb_table_test_all.add_data(label, *row)
                wandb.log({"test_top_tokens_expert_assignments": wandb_table_test_all})

                # Wandb log for TEST (no PAD)
                wandb_table_test_no_pad = wandb.Table(columns=["Token"] + [f"Expert {i}" for i in range(num_experts)])
                if test_df_no_pad.index.size > 0:
                    for label, row in zip(test_df_no_pad.index, test_df_no_pad.values):
                        wandb_table_test_no_pad.add_data(label, *row)
                    wandb.log({"test_top_tokens_no_pad_expert_assignments": wandb_table_test_no_pad})


                
                # After validation + test evaluation in each epoch
    print("len(all_preds_models):", len(all_preds_models))
    print("val_weights:", val_weights)
    all_preds_models_np = np.array(all_preds_models)
    val_weights_np = np.array(val_weights)  # Shape: [num_models]
    val_weights_np /= val_weights_np.sum()  # Normalize to sum to 1
    if weighted_majority_voting:
        assert val_weights_np.shape[0] == all_preds_models_np.shape[0], f"Mismatch: val_weights={val_weights_np.shape[0]}, preds={all_preds_models_np.shape[0]}"
        if val_weights_np.sum() == 0:
            print("Warning: Sum of validation weights is zero. Using uniform weights.")
            val_weights_np = np.ones_like(val_weights_np) # Use uniform weights if all are zero
            val_weights_np /= val_weights_np.sum() # Re-normalize (will sum to 1 now)
            
        weighted_logits = (val_weights_np[:, np.newaxis, np.newaxis] * all_preds_models_np).sum(axis=0)
        print(f"weighted logits: {weighted_logits}")
        if span:
            # From M-weighted across models:
            combined_logits = weighted_logits  # shape [num_spans, num_classes]
            transcript_ids, final_preds, avg_logits = average_logits(span_ids, combined_logits)
            
            # Get true transcript-level labels
            seen = {}
            for sid, lbl in zip(span_ids, labels_this_epoch):
                tid = sid.split('_seg')[0]
                if tid not in seen:
                    seen[tid] = lbl
            true_labels = [seen[tid] for tid in transcript_ids]

            print(f"[DEBUG] Aggregated {len(span_ids)} spans into {len(transcript_ids)} transcripts.")
            acc = accuracy_score(true_labels, final_preds)
        else:
            final_preds = np.argmax(weighted_logits, axis=1)
            true_labels = labels_this_epoch
            acc = accuracy_score(true_labels, final_preds)

    
    print(f"[DEBUG] Final predictions shape: {final_preds.shape}")
    print(f"[DEBUG] Final probabilities shape: {probs.shape}")
    print(f"[DEBUG] Sample predictions: {final_preds[:5]}")


    # Get per-model best accuracy if ensemble
    if model_used == "ensemble":
        bert_acc = np.max(bert_accuracies[-top_n_epochs:]) if bert_accuracies else None
        roberta_acc = np.max(roberta_accuracies[-top_n_epochs:]) if roberta_accuracies else None

        print(f"Seed {seed} BERT Accuracy     : {bert_acc:.4f}" if bert_acc is not None else "No BERT accuracy available.")
        print(f"Seed {seed} RoBERTa Accuracy : {roberta_acc:.4f}" if roberta_acc is not None else "No RoBERTa accuracy available.")
        print(f"Seed {seed} Ensemble Accuracy: {acc:.4f}")
    else:
        print(f"Seed {seed} Accuracy: {acc:.4f}")

    if span:
        report = classification_report(true_labels, final_preds, output_dict=True)
    else:
        report = classification_report(all_labels_final, final_preds, output_dict=True)

    if weighted_majority_voting:
        probs = weighted_logits
    else:
        probs = np.mean(all_preds_models_np, axis=0)
    probs = np.clip(probs, 1e-9, 1.0)
    entropy = -np.sum(probs * np.log(probs)) / len(probs)


    wandb.log({
        "test_accuracy": acc,
        "test_precision_AD": report['1']['precision'],
        "test_recall_AD": report['1']['recall'],
        "test_f1_AD": report['1']['f1-score'],
        "mean_prob_alzheimer": np.mean(probs[:, 0]),
        "mean_prob_healthy": np.mean(probs[:, 1]),
        "output_entropy": entropy
    }, step=global_step)
    

    cm = confusion_matrix(true_labels if span else all_labels_final, final_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    
    # === COLLECT INTERPRETABILITY METRICS ===

    # 1. Mean Cosine Similarity of expert weights
    moe_modules = [m for m in model.modules() if isinstance(m, MoEAttentionProjection)]
    mean_cos_sim = compute_mean_cosine_similarity(moe_modules)
    print(f"Mean Expert Cosine Similarity: {mean_cos_sim:.4f}")
    wandb.log({"mean_expert_cosine_similarity": mean_cos_sim}, step=global_step)

    # 2. Mean gating entropy (on test set for reporting)
    mean_gating_entropy = compute_gating_entropy_over_dataset(model, test_loader)
    print(f"Mean Gating Entropy: {mean_gating_entropy:.4f}")
    wandb.log({"mean_gating_entropy": mean_gating_entropy}, step=global_step)

    # 3. Diversity Index (using final test expert assignment counts)
    # expert_usage_counts should be collected for the test set (see below)
    diversity_idx = compute_expert_assignment_diversity(expert_usage_counts_test)
    print(f"Expert Assignment Diversity Index: {diversity_idx:.4f}")
    wandb.log({"expert_diversity_index": diversity_idx}, step=global_step)

    # Save these values to global lists (make sure you declare these as global in run_pbft() if needed)
    all_mean_cos_sims.append(mean_cos_sim)
    all_gating_entropies.append(mean_gating_entropy)
    all_diversity_indices.append(diversity_idx)




    run.finish()
    return acc

# ===== Main =====
def main():
    train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR, span=span)
    test_df = load_test_set(span=span)

    gmm_control, gmm_ad = fit_gmms(train_df_full)
    train_df_full = compute_ad_prob(train_df_full, gmm_control, gmm_ad)
    test_df = compute_ad_prob(test_df, gmm_control, gmm_ad)

    seeds = range(42, 43)
    accuracies = []
    for seed in seeds:
        train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)
        acc = run_pbft(seed, train_df, val_df, test_df)
        accuracies.append(acc)

    print("\n========== PBFT Summary Over 15 Seeds ==========")
    print(f"Mean Accuracy     : {np.mean(accuracies):.4f}")
    print(f"Std Deviation     : {np.std(accuracies):.4f}")
    print(f"Max Accuracy      : {np.max(accuracies):.4f}")
    print(f"Mean Cosine Similarity       : {np.mean(all_mean_cos_sims):.4f} ± {np.std(all_mean_cos_sims):.4f}")
    print(f"Mean Gating Entropy          : {np.mean(all_gating_entropies):.4f} ± {np.std(all_gating_entropies):.4f}")
    print(f"Mean Diversity Index         : {np.mean(all_diversity_indices):.4f} ± {np.std(all_diversity_indices):.4f}")

    
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
    args = parse_args()
    if args.mode == "sweep":
        # Sweep version of main
        train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR)
        test_df = load_test_set()
        gmm_control, gmm_ad = fit_gmms(train_df_full)
        train_df_full = compute_ad_prob(train_df_full, gmm_control, gmm_ad)
        test_df = compute_ad_prob(test_df, gmm_control, gmm_ad)
        train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=47)
        run_pbft(47, train_df, val_df, test_df)
    else:
        main()