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
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType


# =========== Settings ==============

span = True
moe_in_attention = True
pause_encoding = True
transcript_type = "manual" # ASR_both_speakers, ASR_participant_only, manual, ASR_both_speakers_PE_medium
fine_tuned = False
prompt_type = "different_BERT-V1_RoBERTa-Original" # both_original or both_v1 or different_BERT-V1_RoBERTa-Original, both_v1, just_question_first_org
model_used = "ensemble"
plm_size = "base"
question_first = "late_fusion"
num_epochs = 20
top_n_epochs = 3
weighted_majority_voting = True
num_experts = 4


# ========================================

# revert back after sweep
# def replace_attention_with_moe(model, d_model, n_heads, num_experts):

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_to_sentence_spans(hidden_states, attention_mask=None, span_count=4):
    B, T, C = hidden_states.size()
    spans, masks = [], []

    for b in range(B):
        if attention_mask is not None:
            mask = attention_mask[b].bool()
            valid_len = mask.sum().item()
        else:
            valid_len = T

        tokens = hidden_states[b, :valid_len]
        span_len = valid_len // span_count
        remainder = valid_len % span_count

        idx = 0
        chunks, chunk_masks = [], []
        for i in range(span_count):
            extra = 1 if i < remainder else 0
            this_len = span_len + extra
            chunk = tokens[idx:idx+this_len]
            chunk_mask = torch.ones(this_len, dtype=torch.bool, device=hidden_states.device)
            idx += this_len

            chunks.append(chunk)
            chunk_masks.append(chunk_mask)

        max_span_len = max(chunk.size(0) for chunk in chunks)
        padded_chunks = [F.pad(chunk, (0, 0, 0, max_span_len - chunk.size(0))) for chunk in chunks]
        padded_masks = [F.pad(mask, (0, max_span_len - mask.size(0)), value=0) for mask in chunk_masks]

        spans.append(torch.stack(padded_chunks, dim=0))
        masks.append(torch.stack(padded_masks, dim=0))

    spans = torch.stack(spans, dim=0)  # (B, span_count, max_len, C)
    masks = torch.stack(masks, dim=0)  # (B, span_count, max_len)
    return spans, masks


class TopKMoEAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_experts, top_k=2, span_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = d_model // n_heads
        self.span_len = span_len

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.k_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.v_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])

        self.gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, pause_feats=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        B, T, C = hidden_states.size()
        span = False

        if span:  # USE SPAN SLICING
            S = self.span_len
            num_spans = (T + S - 1) // S

            pad_len = num_spans * S - T
            if pad_len > 0:
                pad_tensor = torch.zeros(B, pad_len, C, device=hidden_states.device)
                hidden_states = torch.cat([hidden_states, pad_tensor], dim=1)

            x, mask = split_to_sentence_spans(hidden_states, attention_mask, span_count=num_spans)
        else:  # USE FULL SEQUENCE AS A SINGLE SPAN
            x = hidden_states.unsqueeze(1)  # (B, 1, T, C)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1)  # (B, 1, T)
            else:
                mask = torch.ones(B, 1, T, device=hidden_states.device)

        num_spans = x.size(1)
        mask = mask.float()
        mask_sum = mask.sum(dim=2, keepdim=True).clamp(min=1)
        span_pooled = (x * mask.unsqueeze(-1)).sum(dim=2) / mask_sum

        gating_logits = self.gate(span_pooled)

        topk_vals, topk_indices = gating_logits.topk(self.top_k, dim=-1)
        gating_weights = torch.zeros_like(gating_logits)
        gating_weights.scatter_(-1, topk_indices, torch.softmax(topk_vals, dim=-1))
        self.last_gating = gating_weights.detach().cpu()
        
        # Log selected expert indices
        selected_experts = topk_indices.detach().cpu().numpy().flatten()
        counts = np.bincount(selected_experts, minlength=self.num_experts)
        for i, count in enumerate(counts):
            wandb.log({f"expert_{i}_selection_count": count})


        out_spans = []
        for i in range(num_spans):
            span = x[:, i, :, :]
            S_i = span.size(1)
            weights = gating_weights[:, i, :]

            q = sum(weights[:, j].unsqueeze(1).unsqueeze(2) * self.q_experts[j](span) for j in range(self.num_experts))
            k = sum(weights[:, j].unsqueeze(1).unsqueeze(2) * self.k_experts[j](span) for j in range(self.num_experts))
            v = sum(weights[:, j].unsqueeze(1).unsqueeze(2) * self.v_experts[j](span) for j in range(self.num_experts))

            q = q.view(B, S_i, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S_i, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S_i, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            span_mask = mask[:, i, :S_i].unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~span_mask.bool(), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, S_i, C)
            out_spans.append(out)

        out = torch.cat(out_spans, dim=1)[:, :T, :]
        return self.out_proj(out), None

def replace_attention_with_moe(model, d_model=768, n_heads=12, num_experts=None):

    for name, module in model.named_modules():
        if isinstance(module, (transformers.models.bert.modeling_bert.BertSelfAttention,
                               transformers.models.roberta.modeling_roberta.RobertaSelfAttention)):
            parent = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent, name.rsplit('.', 1)[1], TopKMoEAttention(d_model, n_heads, num_experts))


class MoEAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_experts, span_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.head_dim = d_model // n_heads
        self.span_len = span_len

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.k_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.v_experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])

        self.gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, pause_feats=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        B, T, C = hidden_states.size()
        S = self.span_len
        num_spans = (T + S - 1) // S

        # Pad sequence to full span length
        pad_len = num_spans * S - T
        if pad_len > 0:
            pad_tensor = torch.zeros(B, pad_len, C, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, pad_tensor], dim=1)

        x, mask = split_to_sentence_spans(hidden_states, attention_mask, span_count=self.num_experts)
        num_spans = x.size(1)
        mask = mask.float()
        mask_sum = mask.sum(dim=2, keepdim=True).clamp(min=1)
        span_pooled = (x * mask.unsqueeze(-1)).sum(dim=2) / mask_sum
        gating_logits = self.gate(span_pooled)     # (B, num_spans, num_experts)
        gating_weights = torch.softmax(gating_logits, dim=-1)  # (B, num_spans, num_experts)
        self.last_gating = gating_weights.detach().cpu()

        # Expert computation per span
        out_spans = []
        for i in range(num_spans):
            span = x[:, i, :, :]  # (B, S_i, C)
            S_i = span.size(1)
            weights = gating_weights[:, i, :]  # (B, num_experts)

            q = sum(w[:, None, None] * expert(span) for w, expert in zip(weights.T, self.q_experts))
            k = sum(w[:, None, None] * expert(span) for w, expert in zip(weights.T, self.k_experts))
            v = sum(w[:, None, None] * expert(span) for w, expert in zip(weights.T, self.v_experts))

            assert q.numel() == B * S_i * self.d_model, f"Unexpected size: got {q.numel()}, expected {B*S_i*self.d_model}"

            q = q.view(B, S_i, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S_i, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S_i, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            # mask: (B, S_i), expand to match attention score shape
            span_mask = mask[:, i, :S_i].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_i)
            scores = scores.masked_fill(~span_mask.bool(), float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, S_i, C)
            out_spans.append(out)

        out = torch.cat(out_spans, dim=1)[:, :T, :]  # Trim to original length
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
        if plm_size == "large":
            model_configs = [("roberta", "roberta-large")]
        elif plm_size == "base":
            model_configs = [("roberta", "roberta-base")]

    run = wandb.init(
        entity="kj1080-university-of-cambridge",
        project="mlmi_thesis",
        name=f"MOE_span_attention__{model_used}_testing_seed_{seed}",
        config={
            "seed": seed,
            "model": model_configs,
            "epochs": num_epochs
        },
        settings=wandb.Settings(_disable_stats=True),
        resume=False
    )

    config = wandb.config

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
                    replace_attention_with_moe(plm, d_model=768, n_heads=12, num_experts=num_experts)  # Adjust for 'base'; for 'large', use d_model=1024, n_heads=16
                elif plm_size == 'large':
                    replace_attention_with_moe(plm, d_model=1024, n_heads=16, num_experts=num_experts) 

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
            


            optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * len(train_loader) * num_epochs,
                num_training_steps=len(train_loader) * num_epochs
            )


            epoch_logits, epoch_val_acc = [], []
            for epoch in tqdm(range(num_epochs), desc=f"{model_path}"):
                model.train()
                for i, batch in enumerate(train_loader):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    batch_inputs = {k: v for k, v in batch.items() if k != 'label'}
                    logits = model(batch_inputs)
                    
                    grad_accum_steps = 8
                    loss = torch.nn.CrossEntropyLoss()(logits, batch['label']) / grad_accum_steps
                    
                    # Auxiliary loss to encourage diverse expert usage
                    aux_loss = 0.0
                    moe_layers = [m for m in model.plm.modules() if hasattr(m, "last_gating")]
                    if moe_layers:
                        gating = moe_layers[0].last_gating.to(logits.device)  # shape: [batch_size, num_spans, num_experts]
                        expert_usage = gating.mean(dim=(0, 1))  # avg over batch & spans: [num_experts]
                        aux_loss = (expert_usage ** 2).sum()  # encourages uniformity
                        λ = 0.2 # tune this
                        loss += λ * aux_loss
                        wandb.log({"aux_loss": aux_loss.item(), "epoch": epoch})
                    
                    loss.backward()

                    if (i + 1) % grad_accum_steps == 0:
                         wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": epoch * len(train_loader) + i})
                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Can tune1.0 as needed
                         optimizer.step()
                         lr_scheduler.step()

                        # Gradient norm logging
                         total_norm = 0
                         for p in model.parameters():
                             if p.grad is not None:
                                 param_norm = p.grad.data.norm(2)
                                 total_norm += param_norm.item() ** 2
                         total_norm = total_norm ** 0.5
                         wandb.log({"grad_norm": total_norm, "epoch": epoch, "step": epoch * len(train_loader) + i})

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
                wandb.log({"val_accuracy": val_acc, "epoch": epoch})

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

    wandb.log({"test_accuracy": acc})
    
    run.finish()
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
    
# def sweep_main():
#     print('Starting sweep with settings:')
#     print(f"pause_encoding: {pause_encoding}, transcript_type: {transcript_type}, model_used: {model_used}")

#     train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR)
#     test_df = load_test_set()

#     if pause_encoding:
#         gmm_control, gmm_ad = fit_gmms(train_df_full)
#         train_df_full = compute_ad_prob(train_df_full, gmm_control, gmm_ad)
#         test_df = compute_ad_prob(test_df, gmm_control, gmm_ad)

#     train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=42)
#     run_pbft(42, train_df, val_df, test_df)

# if __name__ == "__main__":
#     sweep_main()

