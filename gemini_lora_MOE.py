import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import Linear as LoRALinear
import wandb
import os
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from transformers import AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from openprompt.plms import load_plm

# Configuration
pause_encoding = True 
transcript_type = "manual"
model_used = "bert"
plm_size = "base"
lora = True 
num_experts = 2
num_epochs = 10
top_n_epochs = 1
weighted_majority_voting = True  # (if using ensemble voting by validation performance)
prompt_type = "both_v1"
grad_accum_steps = 4

# Paths for transcripts
if pause_encoding == False:
    if transcript_type == "ASR_both_speakers":
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
CONTROL_IDS = {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011", "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034", "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049", "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063", "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"}

print('Using Transcripts from:', TRANSCRIPT_DIR)

def split_to_sentence_spans(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, span_count: int = 4):
    """
    Split hidden states into `span_count` spans (chunks) and pad each to the same length.
    Returns (spans, spans_mask) with shapes:
      - spans: (B, span_count, max_span_len_across_batch, C)
      - spans_mask: (B, span_count, max_span_len_across_batch)
    """
    B, T, C = hidden_states.size()
    spans_list, masks_list = [], []
    for b in range(B):
        # Compute valid token length for this sequence
        valid_len = attention_mask[b].long().sum().item() if attention_mask is not None else T
        # If no valid tokens (all padding), create empty spans
        if valid_len == 0:
            chunks = [torch.zeros((0, C), device=hidden_states.device) for _ in range(span_count)]
            chunk_masks = [torch.zeros((0), device=hidden_states.device, dtype=torch.bool) for _ in range(span_count)]
            max_span_len = 0
        else:
            tokens = hidden_states[b, :valid_len]  # (valid_len, C)
            # Determine span lengths (distribute valid_len across span_count as evenly as possible)
            base_len = valid_len // span_count
            remainder = valid_len % span_count
            chunks, chunk_masks = [], []
            idx = 0
            for i in range(span_count):
                extra = 1 if i < remainder else 0
                this_len = base_len + extra
                chunk = tokens[idx: idx + this_len]
                idx += this_len
                chunks.append(chunk)
                # Mask for this chunk (1s for actual tokens)
                chunk_masks.append(torch.ones(chunk.size(0), device=hidden_states.device, dtype=torch.bool))
            max_span_len = max(chunk.size(0) for chunk in chunks)
        # Pad each chunk to max_span_len for this sequence
        padded_chunks = [F.pad(chunk, (0, 0, 0, max_span_len - chunk.size(0))) for chunk in chunks]
        padded_masks = [F.pad(mask, (0, max_span_len - mask.size(0)), value=False) for mask in chunk_masks]
        spans_list.append(torch.stack(padded_chunks, dim=0))   # (span_count, max_span_len, C)
        masks_list.append(torch.stack(padded_masks, dim=0))    # (span_count, max_span_len)
    # Convert list to tensors and pad span lengths across batch to uniform size
    span_count = len(spans_list[0]) if spans_list else span_count
    max_len_across_batch = max(span_tensor.size(1) for span_tensor in spans_list) if spans_list else 0
    for i in range(len(spans_list)):
        if spans_list[i].size(1) < max_len_across_batch:
            pad_len = max_len_across_batch - spans_list[i].size(1)
            spans_list[i] = F.pad(spans_list[i], (0, 0, 0, pad_len))
            masks_list[i] = F.pad(masks_list[i], (0, pad_len), value=False)
    spans = torch.stack(spans_list, dim=0)   # (B, span_count, max_len_across_batch, C)
    masks = torch.stack(masks_list, dim=0)   # (B, span_count, max_len_across_batch)
    return spans, masks

def apply_lora(plm, config):
    """Apply LoRA to the given model (for sequence classification task) and return the wrapped model and its LoRA config."""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.get("lora_r", 1),
        lora_alpha=config.get("lora_alpha", 512),
        lora_dropout=config.get("lora_dropout", 0.01)
    )
    plm = get_peft_model(plm, lora_config)
    return plm, lora_config

class TopKMoEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_experts: int, top_k: int = 2, span_len: int = 64, dropout: float = 0.1, lora_cfg: Optional[LoraConfig] = None):
        """
        Mixture-of-Experts multi-head self-attention.
        Replaces a standard self-attention mechanism with `num_experts` experts, selecting top_k experts per span of the sequence.
        If lora_cfg is provided, uses LoRA adapters in the expert linear layers.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = d_model // n_heads
        self.span_len = span_len  # target span length for splitting sequences
        # Define expert linear layers for Q, K, V projections
        if lora_cfg:
            def create_lora_linear():
                # Create a base linear layer and wrap it with LoRA adapter
                base_linear = nn.Linear(d_model, d_model, bias=False)
                return LoRALinear(
                    base_layer=base_linear,
                    adapter_name="default",
                    r=lora_cfg.r,
                    lora_alpha=lora_cfg.lora_alpha,
                    lora_dropout=lora_cfg.lora_dropout
                )
            self.q_experts = nn.ModuleList(create_lora_linear() for _ in range(num_experts))
            self.k_experts = nn.ModuleList(create_lora_linear() for _ in range(num_experts))
            self.v_experts = nn.ModuleList(create_lora_linear() for _ in range(num_experts))
        else:
            # Standard linear layers (no LoRA)
            self.q_experts = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(num_experts))
            self.k_experts = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(num_experts))
            self.v_experts = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(num_experts))
        # Gating network to select experts
        self.gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        self.attn_dropout = nn.Dropout(dropout)
        self._current_aux_loss = torch.tensor(0.0)  # stores gating entropy loss for this layer
        self.last_gating = None  # stores last computed gating weights (for analysis/logging)

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False,
                **kwargs):
        """
        Forward pass for MoE self-attention.
        Returns a tuple like (context_layer, attn_weights?) to mimic BertSelfAttention.
        """
        B, T, C = hidden_states.size()
        if T == 0:
            # Handle empty input edge case
            empty_out = torch.zeros_like(hidden_states)
            return (empty_out, None) if output_attentions else (empty_out,)
        # Split sequence into spans
        num_spans = (T + self.span_len - 1) // self.span_len  # number of spans for this sequence length
        x_spans, mask_spans = split_to_sentence_spans(hidden_states, attention_mask, span_count=num_spans)  # shapes: (B, num_spans, L_span, C) and (B, num_spans, L_span)
        # Pool each span to get gating inputs (using mean of token representations in the span)
        mask_for_pool = mask_spans.unsqueeze(-1).float()  # (B, num_spans, L_span, 1)
        span_sum = (x_spans * mask_for_pool).sum(dim=2)  # sum over tokens in span -> (B, num_spans, C)
        span_len = mask_for_pool.sum(dim=2).clamp(min=1e-9)  # count of valid tokens per span -> (B, num_spans, 1)
        span_pooled = span_sum / span_len  # mean pooling (avoid division by 0 via clamp)
        # Compute gating logits and weights
        gating_logits = self.gate(span_pooled)            # (B, num_spans, num_experts)
        topk_vals, topk_idx = gating_logits.topk(self.top_k, dim=-1)  # (B, num_spans, top_k)
        gating_weights = torch.zeros_like(gating_logits)  # initialize all-zero
        # Scatter normalized top-k weights into the full gating weight tensor
        gating_weights.scatter_(-1, topk_idx, torch.softmax(topk_vals, dim=-1))
        self.last_gating = gating_weights.detach().cpu()   # store gating distribution for logging
        # Calculate auxiliary loss = average entropy of gating distributions (encourage diversity)
        self._current_aux_loss = -(gating_weights * (gating_weights.clamp(min=1e-9).log())).sum(dim=-1).mean()
        # Perform attention for each span using the mixture of experts
        out_spans = []  # to collect output for each span
        B, num_spans, L_span, _ = x_spans.size()
        for i in range(num_spans):
            # Slice out the i-th span for all batch elements
            span_data = x_spans[:, i, :, :]   # (B, L_span, C)
            span_mask = mask_spans[:, i, :]   # (B, L_span)
            if span_data.size(1) == 0:
                continue  # skip empty span (if any)
            # Compute weighted sum of expert projections for this span
            weights = gating_weights[:, i, :]  # (B, num_experts)
            # Project span through each expert's Q, K, V and weight their outputs by gating weights
            Q_combined = sum(weights[:, j].unsqueeze(1).unsqueeze(2) * self.q_experts[j](span_data) for j in range(self.num_experts))
            K_combined = sum(weights[:, j].unsqueeze(1).unsqueeze(2) * self.k_experts[j](span_data) for j in range(self.num_experts))
            V_combined = sum(weights[:, j].unsqueeze(1).unsqueeze(2) * self.v_experts[j](span_data) for j in range(self.num_experts))
            # Reshape combined projections to (B, n_heads, L_span, head_dim)
            Q = Q_combined.view(B, L_span, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, L_span, head_dim)
            K = K_combined.view(B, L_span, self.n_heads, self.head_dim).transpose(1, 2)
            V = V_combined.view(B, L_span, self.n_heads, self.head_dim).transpose(1, 2)
            # Scaled dot-product attention within this span
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, L_span, L_span)
            if span_mask is not None:
                # Create an attention mask for padded tokens in the span
                # span_mask: (B, L_span) -> (B, 1, 1, L_span)
                span_mask_bool = span_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
                attn_scores = attn_scores.masked_fill(~span_mask_bool, float('-inf'))
            attn_probs = torch.softmax(attn_scores, dim=-1)  # attention softmax
            attn_probs = self.attn_dropout(attn_probs)
            # Compute weighted sum of values
            context = torch.matmul(attn_probs, V)  # (B, n_heads, L_span, head_dim)
            # Reshape back to (B, L_span, C)
            context = context.transpose(1, 2).contiguous().view(B, L_span, C)
            out_spans.append(context)
        # Concatenate span outputs to reconstruct the full sequence
        if out_spans:
            concatenated_out = torch.cat(out_spans, dim=1)  # (B, T_actual, C) where T_actual = sum of span lengths (should equal valid_len for each seq)
        else:
            concatenated_out = torch.zeros((B, 0, C), device=hidden_states.device)
        # If needed, pad or truncate to original sequence length T (including padding tokens)
        if concatenated_out.size(1) >= T:
            output_sequence = concatenated_out[:, :T, :]
        else:
            pad_len = T - concatenated_out.size(1)
            pad_tensor = torch.zeros((B, pad_len, C), device=hidden_states.device)
            output_sequence = torch.cat([concatenated_out, pad_tensor], dim=1)
        # Return in format compatible with BertSelfAttention: (context_layer, attn_weights?) 
        if output_attentions:
            # We do not compute global attentions across spans; return None or span-local attn if needed.
            return (output_sequence, None)
        else:
            return (output_sequence,)

def replace_attention_with_moe(model: nn.Module, d_model: Optional[int], n_heads: Optional[int], num_experts: int, lora_cfg: Optional[LoraConfig] = None):
    """
    Replace the self-attention submodule in each Transformer layer with TopKMoEAttention.
    Preserves existing weights for expert 0 and integrates LoRA if provided.
    """
    # Determine model hidden size and number of heads if not given
    if hasattr(model, 'config'):
        if d_model is None:
            d_model = model.config.hidden_size
        if n_heads is None:
            n_heads = model.config.num_attention_heads
    # Determine target module names for self-attention in this model architecture
    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 12
    target_names = [f"encoder.layer.{i}.attention.self" for i in range(num_layers)]
    # Prepare a container to track all MoE modules (for logging/aux loss retrieval)
    if not hasattr(model, '_moe_attention_modules'):
        model._moe_attention_modules = nn.ModuleList()
    for name, module in model.named_modules():
        if name in target_names:
            parent_name = name.rsplit('.', 1)[0]   # e.g., "encoder.layer.0.attention"
            submodule_name = name.rsplit('.', 1)[1]  # "self"
            parent_module = model.get_submodule(parent_name)
            print(f"[INFO] Replacing attention: {name}")
            # Save original self-attention module and its weights
            orig_attn = module  # BertSelfAttention or RobertaSelfAttention instance
            # Instantiate MoE attention and move it to the same device as original module
            device = next(parent_module.parameters()).device if any(parent_module.parameters()) else torch.device('cpu')
            moe_attn = TopKMoEAttention(d_model, n_heads, num_experts, top_k=2, span_len=64, dropout=0.1, lora_cfg=lora_cfg).to(device)
            # Initialize expert 0 weights from original attention (query, key, value)
            try:
                if isinstance(moe_attn.q_experts[0], LoRALinear):
                    # Copy into base layers of LoRA adapters
                    moe_attn.q_experts[0].base_layer.weight.data.copy_(orig_attn.query.weight.data)
                    moe_attn.k_experts[0].base_layer.weight.data.copy_(orig_attn.key.weight.data)
                    moe_attn.v_experts[0].base_layer.weight.data.copy_(orig_attn.value.weight.data)
                else:
                    moe_attn.q_experts[0].weight.data.copy_(orig_attn.query.weight.data)
                    moe_attn.k_experts[0].weight.data.copy_(orig_attn.key.weight.data)
                    moe_attn.v_experts[0].weight.data.copy_(orig_attn.value.weight.data)
                # Note: biases from original attention are not used (our experts use bias=False for stability)
            except Exception as e:
                print(f"Warning: could not copy pre-trained weights for experts: {e}")
            # Set gating biases to favor expert 0 at initialization (so initial behavior resembles original model)
            if hasattr(moe_attn.gate, '__iter__'):  # if gate is a Sequential
                final_linear = moe_attn.gate[-1]
                if final_linear.bias is not None:
                    final_linear.bias.data[0] = 5.0
                    if moe_attn.num_experts > 1:
                        final_linear.bias.data[1:] = 0.0
            # Replace the module
            setattr(parent_module, submodule_name, moe_attn)
            model._moe_attention_modules.append(moe_attn)
            # If using LoRA, freeze expert 0 base weights to preserve pre-trained behavior (train only LoRA for expert 0)
            if lora_cfg is not None:
                for j in range(moe_attn.num_experts):
                    if isinstance(moe_attn.q_experts[j], LoRALinear) and j == 0:
                        moe_attn.q_experts[j].base_layer.weight.requires_grad = False
                    if isinstance(moe_attn.k_experts[j], LoRALinear) and j == 0:
                        moe_attn.k_experts[j].base_layer.weight.requires_grad = False
                    if isinstance(moe_attn.v_experts[j], LoRALinear) and j == 0:
                        moe_attn.v_experts[j].base_layer.weight.requires_grad = False

# ===== Pause Counting (feature extraction) =====
def count_pauses(text: str):
    # Count medium pauses (single periods not part of "...")
    medium_pauses = len(re.findall(r"(?<!\.)\.(?!\.)", text))
    # Count long pauses ("...")
    long_pauses = len(re.findall(r"\.\.\.", text))
    return medium_pauses, long_pauses

# ===== Load Transcripts =====
def load_transcripts_with_pauses(directory: str, is_test: bool = False):
    rows = []
    for fname in os.listdir(directory):
        if not fname.endswith(".txt"):
            continue
        file_id = os.path.splitext(fname)[0].upper()
        label = None if is_test else (0 if file_id in CONTROL_IDS else 1)
        with open(os.path.join(directory, fname), "r") as f:
            text = f.read().strip()
        if text:
            medium, long = count_pauses(text)
            rows.append({"id": file_id, "text": text, "label": label, "medium": medium, "long": long})
    return pd.DataFrame(rows)

def load_test_set():
    # Load official test labels
    df_labels = pd.read_csv(TEST_LABELS_PATH, sep=";")
    df_labels.columns = df_labels.columns.str.strip().str.lower()
    df_labels["id"] = df_labels["id"].str.strip().str.upper()
    df_labels["label"] = df_labels["label"].astype(int)
    rows = []
    for fname in os.listdir(TRANSCRIPT_TEST_DIR):
        if fname.endswith(".txt"):
            file_id = os.path.splitext(fname)[0].upper()
            row = df_labels[df_labels["id"] == file_id]
            if row.empty:
                continue  # skip if no label found for this ID
            with open(os.path.join(TRANSCRIPT_TEST_DIR, fname), "r") as f:
                text = f.read().strip()
            if text:
                medium, long = count_pauses(text)
                label = int(row["label"].values[0])
                rows.append({"id": file_id, "text": text, "label": label, "medium": medium, "long": long})
    return pd.DataFrame(rows).dropna(subset=["text", "label"])

# ===== GMM for Pause Features =====
def fit_gmms(df: pd.DataFrame):
    """Fit Gaussian Mixture Models on pause count features for control vs AD classes."""
    control_data = df[df.label == 0][["medium", "long"]].values
    ad_data = df[df.label == 1][["medium", "long"]].values
    try:
        gmm_ctrl = GaussianMixture(n_components=1, random_state=42).fit(control_data)
        gmm_ad = GaussianMixture(n_components=1, random_state=42).fit(ad_data)
    except ValueError as e:
        print(f"Warning: GMM fitting failed due to insufficient data: {e}")
        return None, None
    return gmm_ctrl, gmm_ad

def compute_ad_prob(df: pd.DataFrame, gmm_ctrl, gmm_ad):
    """Add a column 'pause_prob_ad' with probability of AD computed from GMM scores."""
    if gmm_ctrl is None or gmm_ad is None:
        # If GMM models are not available, assign neutral probability 0.5
        df["pause_prob_ad"] = 0.5
        return df
    feats = df[["medium", "long"]].values
    log_p_ad = gmm_ad.score_samples(feats)
    log_p_ctrl = gmm_ctrl.score_samples(feats)
    # Avoid overflow in exp by clamping difference
    prob_ad = 1.0 / (1.0 + np.exp(np.clip(log_p_ctrl - log_p_ad, -50, 50)))
    df["pause_prob_ad"] = prob_ad
    return df

# ===== Create Prompt Dataset =====
def create_prompt_dataset(df: pd.DataFrame, tokenizer, max_len: int = 512):
    """
    Convert DataFrame to list of InputExamples for OpenPrompt.
    Each example's `text_a` is the transcript (possibly truncated) and `text_b` is a templated sentence with the pause-based probability.
    """
    dataset = []
    for i, row in df.iterrows():
        text = str(row["text"])
        pause_score = float(row.get("pause_prob_ad", 0.5))
        pause_text = f"The probability of Alzheimer's Disease is {pause_score:.2f}."
        tokenized = tokenizer(text, truncation=True, max_length=max_len - 50, add_special_tokens=False)
        truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        
        # NEW: enforce label is present and an integer
        if "label" not in row or pd.isna(row["label"]):
            raise ValueError(f"[DATA ERROR] Missing label in row {i}: {row.to_dict()}")
        try:
            label = int(row["label"])
        except Exception as e:
            raise ValueError(f"[DATA ERROR] Could not cast label at row {i} to int: {row.to_dict()}") from e

        dataset.append(InputExample(guid=str(i), text_a=truncated_text, text_b=pause_text, label=label))
    return dataset


# Accuracy trackers for ensemble components
bert_accuracies = []
roberta_accuracies = []

# ===== Prompt-Based Fine-Tuning (PBFT) =====
def run_pbft(seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Run prompt-based fine-tuning for a given random seed. Supports BERT, RoBERTa, or an ensemble of both.
    Returns the final test accuracy (or ensemble accuracy).
    """
    global bert_accuracies, roberta_accuracies
    # Model configurations to run (depending on model_used and plm_size)
    if model_used == "ensemble":
        model_configs = [
            ("bert", "bert-large-uncased" if plm_size == "large" else "bert-base-uncased"),
            ("roberta", "roberta-large" if plm_size == "large" else "roberta-base")
        ]
    elif model_used == "bert":
        model_configs = [("bert", "bert-large-uncased" if plm_size == "large" else "bert-base-uncased")]
    elif model_used == "roberta":
        model_configs = [("roberta", "roberta-large" if plm_size == "large" else "roberta-base")]
    else:
        raise ValueError(f"Unknown model type: {model_used}")
    # Initialize Weights & Biases logging
    run = wandb.init(
        project="mlmi_thesis",
        name=f"MOExLORA_{model_used}_seed_{seed}",
        config={
            "seed": seed, "model": model_configs, "epochs": num_epochs,
            "lora": lora, "plm": model_used, "plm_size": plm_size,
            "prompt_type": prompt_type, "lr": 5e-6,
            "lora_r": 1, "lora_alpha": 512, "lora_dropout": 0.01
        },
        reinit=True
    )
    config = wandb.config
    # Set random seeds for reproducibility
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    # Define prompt templates for BERT and RoBERTa
    if prompt_type == "different_BERT-V1_RoBERTa-Original":
        template_texts = {
            "bert": [
                'Based on the speech patterns in the transcript: {"placeholder":"text_a"}, the individual is {"mask"}.',
                'This individual is {"mask"}. This is based on the speech patterns in the transcript: {"placeholder":"text_a"}.'
            ],
            "roberta": [
                'The diagnosis result is {"mask"}. {"placeholder":"text_a"}.',
                '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.'
            ]
        }
    elif prompt_type == "both_original":
        template_texts = {
            "bert": [
                'The diagnosis result is {"mask"}. {"placeholder":"text_a"}.',
                '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.'
            ],
            "roberta": [
                'The diagnosis result is {"mask"}. {"placeholder":"text_a"}.',
                '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.'
            ]
        }
    elif prompt_type == "both_v1":
        template_texts = {
            "bert": [
                'Based on the speech patterns in the transcript: {"placeholder":"text_a"}, the individual is {"mask"}.',
                'This individual is {"mask"}. This is based on their speech patterns in the transcript: {"placeholder":"text_a"}.'
            ],
            "roberta": [
                'Based on the speech patterns in the transcript: {"placeholder":"text_a"}, the individual is {"mask"}.',
                'This individual is {"mask"}. This is based on their speech patterns in the transcript: {"placeholder":"text_a"}.'
            ]
        }
    else:
        template_texts = {
            "bert": ['{"placeholder":"text_a"} {"mask"}.'],  # default fallback
            "roberta": ['{"placeholder":"text_a"} {"mask"}.']
        }
    all_model_logits = []  # to store test logits from each model/template for ensembling
    ensemble_weights = []  # to store validation accuracy for weighting ensemble
    all_test_labels = None
    # Iterate through each PLM (and each prompt template variation, if multiple)
    for plm_type, model_name in model_configs:
        templates = template_texts["bert"] if plm_type == "bert" else template_texts["roberta"]
        for template_str in templates:
            # Load pre-trained model and tokenizer
            plm, tokenizer, model_config, WrapperClass = load_plm(plm_type, model_name)
            # Apply LoRA adapters if enabled
            if lora:
                plm, lora_cfg = apply_lora(plm, wandb.config)
            else:
                lora_cfg = None
                # If not using LoRA, ensure all model parameters are trainable
                for param in plm.parameters():
                    param.requires_grad = True
            # Initialize Manual Template and Verbalizer for prompting
            template = ManualTemplate(text=template_str, tokenizer=tokenizer)
            verbalizer = ManualVerbalizer(
                classes=["alzheimer", "healthy"],
                label_words={"alzheimer": ["Alzheimer's"], "healthy": ["healthy"]},
                tokenizer=tokenizer
            )
            # Wrap the model with PromptForClassification (combines template, plm, verbalizer)
            model = PromptForClassification(template=template, plm=plm, verbalizer=verbalizer)
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # Replace BERT/RoBERTa self-attention with MoE-attention in the PLM
            if plm_type in ["bert", "roberta"]:
                replace_attention_with_moe(model.plm, d_model=None, n_heads=None, num_experts=num_experts, lora_cfg=lora_cfg)
            else:
                raise ValueError(f"Unsupported model type for MoE: {plm_type}")
            # Log confirmation of MoE insertion
            for name, module in model.plm.named_modules():
                if isinstance(module, TopKMoEAttention):
                    print(f"[DEBUG] Inserted MoE layer at: {name}")
            # Prepare data loaders for training/validation/testing
            train_data = create_prompt_dataset(train_df, tokenizer)
            val_data = create_prompt_dataset(val_df, tokenizer)
            test_data = create_prompt_dataset(test_df, tokenizer)
            train_loader = PromptDataLoader(dataset=train_data, template=template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                            batch_size=8, shuffle=True, num_workers=0)
            val_loader = PromptDataLoader(dataset=val_data, template=template, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                          batch_size=8, shuffle=False, num_workers=0)
            test_loader = PromptDataLoader(dataset=test_data, template=template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                           batch_size=8, shuffle=False, num_workers=0)
            # Optimizer (filter out frozen params) and learning rate scheduler
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01)
            total_steps = len(train_loader) * num_epochs
            lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
            best_val_acc = -1.0
            best_epoch = -1
            best_test_logits = None
            
            # Training loop
            for epoch in tqdm(range(num_epochs)):
                model.train()
                running_loss = 0.0
                for step, batch in enumerate(train_loader):
                    # Move batch to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(model.device)
                    for step, batch in enumerate(train_loader):
                        print(f"[DEBUG] Batch keys: {batch.keys()}")  # <- Add this line to verify
                        if "label" not in batch:
                            raise KeyError(f"[FATAL] 'label' missing in batch at step {step}. Batch keys: {batch.keys()}")
                    labels = batch["label"]
                    device = model.device
                    batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "label"}
                    labels = batch["label"].to(device)

                    outputs = model(batch_inputs)
                    # Separate logits and auxiliary loss if returned as tuple
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, aux_loss = outputs
                    else:
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        # Sum aux loss from all MoE layers
                        moe_layers = model.plm._moe_attention_modules if hasattr(model.plm, "_moe_attention_modules") else []
                        aux_loss = sum(layer._current_aux_loss for layer in moe_layers) if moe_layers else torch.tensor(0.0, device=model.device)
                    # Compute classification loss
                    cls_loss = nn.CrossEntropyLoss()(logits, labels)
                    total_loss = cls_loss + 0.1 * aux_loss  # include MoE entropy loss (weighted by 0.1)
                    total_loss.backward()
                    running_loss += total_loss.item()
                    # Gradient accumulation handling
                    if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                # Validation after each epoch
                model.eval()
                all_val_preds, all_val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                batch[key] = value.to(model.device)
                        labels = batch["label"]
                        outputs = model(batch)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        preds = torch.argmax(logits, dim=-1)
                        all_val_preds.extend(preds.cpu().numpy().tolist())
                        all_val_labels.extend(labels.cpu().numpy().tolist())
                val_acc = accuracy_score(all_val_labels, all_val_preds)
                wandb.log({f"val_accuracy/{plm_type}": val_acc}, step=epoch)
                # Track best validation performance
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    # Evaluate on test set at this epoch
                    all_test_labels = []
                    test_probabilities = []
                    with torch.no_grad():
                        for batch in test_loader:
                            for key, value in batch.items():
                                if isinstance(value, torch.Tensor):
                                    batch[key] = value.to(model.device)
                            labels = batch.pop("label")
                            outputs = model(batch)
                            logits = outputs[0] if isinstance(outputs, tuple) else outputs
                            probs = torch.softmax(logits, dim=-1)
                            test_probabilities.append(probs.cpu().numpy())
                            all_test_labels.extend(labels.cpu().numpy().tolist())
                    best_test_logits = np.vstack(test_probabilities)  # shape (num_samples, 2)
                # Log training loss for the epoch
                wandb.log({f"train_loss/{plm_type}": running_loss / len(train_loader)}, step=epoch)
            # After training, use the best epoch's test logits
            if best_test_logits is None:
                # In case no validation (though we always have val set), default to last epoch
                all_test_labels = []
                test_probabilities = []
                with torch.no_grad():
                    for batch in test_loader:
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                batch[key] = value.to(model.device)
                                
                                
                        for step, batch in enumerate(train_loader):
                            print(f"[DEBUG] Batch keys: {batch.keys()}")  # <- Add this line to verify
                            if "label" not in batch:
                                raise KeyError(f"[FATAL] 'label' missing in batch at step {step}. Batch keys: {batch.keys()}")
                            labels = batch.pop("label")

                        
                        
                        
                        outputs = model(batch)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        probs = torch.softmax(logits, dim=-1)
                        test_probabilities.append(probs.cpu().numpy())
                        all_test_labels.extend(labels.cpu().numpy().tolist())
                best_test_logits = np.vstack(test_probabilities)
                best_val_acc = accuracy_score(all_val_labels, all_val_preds) if all_val_labels else 0.0
            # Log best validation accuracy and corresponding test accuracy
            test_preds = np.argmax(best_test_logits, axis=1)
            test_acc = accuracy_score(all_test_labels, test_preds)
            wandb.log({f"best_val_accuracy/{plm_type}": best_val_acc, f"test_accuracy/{plm_type}": test_acc})
            print(f"{plm_type.upper()} (prompt: {template_str[:30]}...) Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
            # Store for ensembling
            # Normalize logits for stable ensembling (z-score per sample)
            logits_mean = best_test_logits.mean(axis=1, keepdims=True)
            logits_std = best_test_logits.std(axis=1, keepdims=True) + 1e-8
            norm_logits = (best_test_logits - logits_mean) / logits_std
            all_model_logits.append(norm_logits)
            ensemble_weights.append(best_val_acc)
            if model_used == "ensemble":
                if plm_type == "bert":
                    bert_accuracies.append(test_acc)
                elif plm_type == "roberta":
                    roberta_accuracies.append(test_acc)
    # Ensemble predictions (weighted by validation accuracy if multiple models/templates)
    all_model_logits = np.array(all_model_logits)  # shape (M, n_samples, num_classes)
    ensemble_weights = np.array(ensemble_weights)
    if ensemble_weights.sum() == 0:
        # If all weights are zero (should not happen unless val set missing), use equal weight
        ensemble_probs = all_model_logits.mean(axis=0)
    else:
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        ensemble_probs = np.tensordot(ensemble_weights, all_model_logits, axes=([0], [0]))  # weighted sum over models
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    final_acc = accuracy_score(all_test_labels, ensemble_preds)
    wandb.log({"final_ensemble_accuracy": final_acc})
    print(f"Final {'Ensemble' if len(all_model_logits) > 1 else model_used.capitalize()} Accuracy: {final_acc:.4f}")
    run.finish()
    return final_acc

# ===== Main Execution =====
def main():
    # Load and preprocess data
    train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR, is_test=False)
    test_df = load_test_set()
    # Compute pause-based features and probabilities
    gmm_ctrl, gmm_ad = fit_gmms(train_df_full)
    train_df_full = compute_ad_prob(train_df_full, gmm_ctrl, gmm_ad)
    test_df = compute_ad_prob(test_df, gmm_ctrl, gmm_ad)
    # Split train into train/val
    seeds = [42]  # can run multiple seeds for robustness
    accuracies = []
    # Reset global accuracy lists
    global bert_accuracies, roberta_accuracies
    bert_accuracies = []
    roberta_accuracies = []
    for seed in seeds:
        print(f"\n===== Running training for seed {seed} =====")
        train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=seed)
        acc = run_pbft(seed, train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))
        accuracies.append(acc)
    # Summary
    print("\n========== Summary ==========")
    print(f"Seeds: {seeds}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
    if model_used == "ensemble":
        if bert_accuracies:
            print(f"BERT Accuracies: {bert_accuracies} | Best: {np.max(bert_accuracies):.4f}")
        if roberta_accuracies:
            print(f"RoBERTa Accuracies: {roberta_accuracies} | Best: {np.max(roberta_accuracies):.4f}")

if __name__ == "__main__":
    main()
