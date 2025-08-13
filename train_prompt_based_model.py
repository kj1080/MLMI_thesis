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
from sklearn.metrics import classification_report
import argparse

from transformers import AdamW
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from peft import get_peft_model, LoraConfig, TaskType
from spanning import segment_text, average_logits
import json

# =========== Path Settings ==============

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "sweep"], default="normal", help="Run mode: normal or sweep")
    return parser.parse_args()

span = True
lora = False
pause_encoding = True
transcript_type = "manual" # ASR_both_speakers, ASR_participant_only, manual, ASR_both_speakers_PE_medium
fine_tuned = False
prompt_type = "different_BERT-V1_RoBERTa-Original" # both_original or both_v1 or different_BERT-V1_RoBERTa-Original, both_v1
model_used = "ensemble"
plm_size = "base"
question_first = "late_fusion"
num_epochs = 20
top_n_epochs = 3
weighted_majority_voting = False
output_dir = "final_8th_august_ensemble_outputs/nonmoe"

# ========================================


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


# ===== Lora SWEEP=====
def apply_lora(plm, config):
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(config.lora_r),
        lora_alpha=float(config.lora_alpha),
        lora_dropout=float(config.lora_dropout)
    )
    return get_peft_model(plm, lora_cfg)

def apply_lora(plm, config):
    if lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(config.lora_r),
            lora_alpha=float(config.lora_alpha),
            lora_dropout=float(config.lora_dropout)
        )
            
    else:
        lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=1,
        lora_alpha=512,
        lora_dropout=0.01
    )
        
    return get_peft_model(plm, lora_cfg)


# ===== Pause Counting =====
def count_pauses(text):
    long = len(re.findall(r"\.\.\.", text))
    medium = len(re.findall(r"(?<!\.)\.(?!\.)", text))
    return medium, long

# ===== Load Transcripts =====
def load_transcripts_with_pauses(transcript_dir, is_test=False, span=False):
    rows = []
    for fname in os.listdir(transcript_dir):
        if not fname.endswith(".txt"): continue
        file_id = os.path.splitext(fname)[0].upper()
        label = None if is_test else (0 if file_id in CONTROL_IDS else 1)
        with open(os.path.join(transcript_dir, fname), "r") as f:
            text = f.read().strip()
        if not text:
            continue
        if span:
            segments = segment_text(text)
            for i, seg in enumerate(segments):
                seg_id = f"{file_id}_seg{i}"
                medium, long = count_pauses(seg)
                rows.append({"id": seg_id, "text": seg, "label": label, "medium": medium, "long": long, "orig_id": file_id})
        else:
            medium, long = count_pauses(text)
            rows.append({"id": file_id, "text": text, "label": label, "medium": medium, "long": long, "orig_id": file_id})
    return pd.DataFrame(rows)

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
            if label_row.empty: continue
            with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
                text = f.read().strip()
            if not text:
                continue
            if span:
                segments = segment_text(text)
                for i, seg in enumerate(segments):
                    seg_id = f"{file_id}_seg{i}"
                    medium, long = count_pauses(seg)
                    rows.append({"id": seg_id, "text": seg, "label": int(label_row["label"].values[0]), "medium": medium, "long": long, "orig_id": file_id})
            else:
                medium, long = count_pauses(text)
                rows.append({"id": file_id, "text": text, "label": int(label_row["label"].values[0]), "medium": medium, "long": long, "orig_id": file_id})
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
        dataset.append(InputExample(guid=row["id"], text_a=truncated_text, text_b=pause_text, label=int(row["label"])))
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

    run = wandb.init(
        entity="kj1080-university-of-cambridge",
        project="mlmi_thesis",
        name=f"lora__{model_used}_testing_seed_{seed}",
        config={
            "seed": seed,
            "lora": lora,
            "plm": model_used,
            "plm_size": plm_size,
            "prompt_type": prompt_type,
            "epochs": num_epochs,
            "lr": 4e-5, # 4e-5 if using LORA, otherwise using 1e-5
            "lora_r": 1,
            "lora_alpha": 512,
            "lora_dropout": 0.01
        },
        settings=wandb.Settings(_disable_stats=True),
        resume=False
    )


    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # templates = ['The diagnosis result is {"mask"}. Based on pauses features, {"placeholder":"text_b"}. Transcript: {"placeholder":"text_a"}.', 'Based on pauses features, {"placeholder":"text_b"}. Transcript: {"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
    # templates = ['The diagnosis result is {"mask"}. {"placeholder":"text_b"}. {"placeholder":"text_a"}.', '{"placeholder":"text_b"}. {"placeholder":"text_a"}. The diagnosis result is {"mask"}.']


    # templates = [
    # # Prompt 1: pause-aware
    # 'The diagnosis result is {"mask"}. {"placeholder":"text_b"}. {"placeholder":"text_a"}.',
    # '{"placeholder":"text_b"}. {"placeholder":"text_a"}. The diagnosis result is {"mask"}.',
    
    # # Prompt 2: plain transcript only
    # 'The diagnosis result is {"mask"}. {"placeholder":"text_a"}.',
    # '{"placeholder":"text_a"}. The diagnosis result is {"mask"}.']
    
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
    all_labels_final = []
    val_weights = [] 
    global_step = 0

    for plm_type, model_path in model_configs:
        if plm_type == "bert":
            model_templates = template_bert
        elif plm_type == "roberta":
            model_templates = template_roberta
        else:
            raise ValueError(f"Unknown model type: {plm_type}")
        
        for template_text in model_templates:

            plm, tokenizer, model_config, WrapperClass = load_plm(plm_type, model_path)
            if lora:
                plm = apply_lora(plm, wandb.config)

                # print("Trainable parameters:")
                for name, param in plm.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                total_params = sum(p.numel() for p in plm.parameters())
                trainable_params = sum(p.numel() for p in plm.parameters() if p.requires_grad)
                print(f"Trainable / Total params: {trainable_params} / {total_params}")

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
            optimizer = AdamW(model.parameters(), lr=float(wandb.config.lr))



            epoch_logits, epoch_val_acc = [], []
            
            for epoch in tqdm(range(num_epochs), desc=f"{model_path}"):
                model.train()
                for i, batch in enumerate(train_loader):
                    batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}

                    batch_inputs = {k: v for k, v in batch.items() if k != 'label'}
                    logits = model(batch_inputs)
                    loss = torch.nn.CrossEntropyLoss()(logits, batch['label']) / 4
                    loss.backward()
                    wandb.log({"train_loss": loss.item()}, step=global_step)
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

                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}

                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        val_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                        val_labels.extend(batch['label'].cpu().numpy())
                val_acc = accuracy_score(val_labels, np.concatenate(val_preds))
                epoch_val_acc.append((val_acc, epoch))
                wandb.log({"val_accuracy": val_acc}, step=global_step)

                # test
                preds_this_epoch = []
                labels_this_epoch = []
                span_ids_this_epoch = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}

                        logits = model({k: v for k, v in batch.items() if k != 'label'})
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        preds_this_epoch.append(probs)
                        labels_this_epoch.extend(batch['label'].cpu().numpy())
                        # Get guid from batch, supports batch_size=1 or >1
                        if "guid" in batch:
                            batch_ids = batch["guid"]
                            # batch_ids could be a list/str/tensor depending on loader
                            if isinstance(batch_ids, (list, tuple)):
                                span_ids_this_epoch.extend([str(x) for x in batch_ids])
                            elif isinstance(batch_ids, torch.Tensor):
                                span_ids_this_epoch.extend([str(x.item()) for x in batch_ids])
                            else:
                                span_ids_this_epoch.append(str(batch_ids))
                        else:
                            # fallback to index
                            span_ids_this_epoch.append(str(len(span_ids_this_epoch)))
                logits_epoch = np.vstack(preds_this_epoch)
                epoch_logits.append(logits_epoch)

                # For single-span, old logic:
                if not span:
                    pred_classes = np.argmax(logits_epoch, axis=1)
                    test_acc = accuracy_score(labels_this_epoch, pred_classes)
                else:
                    # For span mode: aggregate per transcript
                    transcript_ids, final_preds_span, _ = average_logits(span_ids_this_epoch, logits_epoch)
                    # Get true label for each transcript
                    # Map transcript_id to label using test_df
                    tid2label = dict(zip(test_df["orig_id"], test_df["label"]))
                    true_labels = [tid2label[tid] for tid in transcript_ids]
                    test_acc = accuracy_score(true_labels, final_preds_span)

                wandb.log({"test_accuracy": test_acc}, step=global_step)

            top_epochs = sorted(epoch_val_acc, reverse=True)[:top_n_epochs]
            all_labels_final = labels_this_epoch 

            all_raw_logits_per_model = []  # NEW
            all_probs_per_model = []       # NEW
            
            for _, best_epoch in top_epochs:
                print(f"Selected epoch {best_epoch} for model={plm_type}, template='{template_text[:60]}...' with val_acc={_:.4f}")

                logits_stack = epoch_logits[best_epoch]
                preds = np.argmax(logits_stack, axis=1)
                
                # --- Save raw logits and probabilities for meta-ensemble ---
                all_raw_logits_per_model.append(logits_stack)       # shape: [num_samples, num_classes]
                all_probs_per_model.append(torch.softmax(torch.tensor(logits_stack), dim=-1).numpy())


                acc_single = accuracy_score(labels_this_epoch, preds)  # <-- now using local labels from this epoch


                if model_used == "ensemble":
                    if plm_type == "bert":
                        bert_accuracies.append(acc_single)
                    elif plm_type == "roberta":
                        roberta_accuracies.append(acc_single)

                    # Normalize and store logits
                logits_stack_norm = (logits_stack - logits_stack.mean(1, keepdims=True)) / (logits_stack.std(1, keepdims=True) + 1e-8)
                all_preds_models.append(logits_stack_norm)
                val_weights.append(acc_single)
                
                # After validation + test evaluation in each epoch

    # all_preds_models_np = np.array(all_preds_models)
    # val_weights_np = np.array(val_weights)  # Shape: [num_models]
    # val_weights_np /= val_weights_np.sum()  # Normalize to sum to 1
    # if weighted_majority_voting:
    #     weighted_logits = np.tensordot(val_weights_np, all_preds_models_np, axes=([0], [0]))  # Shape: [num_samples, num_classes]
    #     final_preds = np.argmax(weighted_logits, axis=1)

    # else:
    #     votes = np.argmax(all_preds_models_np, axis=2)
    #     final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
    
    # # Final ensemble prediction 
    # acc = accuracy_score(all_labels_final, final_preds)
        
    # ---- Final ensemble/voting/aggregation block ----
    model_name = f"{model_used}_{plm_size}_{prompt_type}"
    
    # --- Save per-model/template logits and probs for meta-ensembling ---
    all_raw_logits_per_model = np.array(all_raw_logits_per_model) # [num_models, num_samples, num_classes]
    all_probs_per_model = np.array(all_probs_per_model)           # same shape
    np.save(os.path.join(output_dir, f"all_logits_{model_name}_seed{seed}.npy"), all_raw_logits_per_model)
    np.save(os.path.join(output_dir, f"all_probs_{model_name}_seed{seed}.npy"), all_probs_per_model)

    if not span:
        all_preds_models_np = np.array(all_preds_models)
        val_weights_np = np.array(val_weights)
        val_weights_np /= val_weights_np.sum()
        if weighted_majority_voting:
            weighted_logits = np.tensordot(val_weights_np, all_preds_models_np, axes=([0], [0]))
            final_preds = np.argmax(weighted_logits, axis=1)
        else:
            votes = np.argmax(all_preds_models_np, axis=2)
            final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
        acc = accuracy_score(all_labels_final, final_preds)
    else:
        all_preds_models_np = np.array(all_preds_models)  # [num_models, num_spans, num_classes]
        val_weights_np = np.array(val_weights)
        val_weights_np /= val_weights_np.sum()
        weighted_span_logits = np.tensordot(val_weights_np, all_preds_models_np, axes=([0], [0]))  # [num_spans, num_classes]
        span_ids = list(test_df["id"])  # Test set span ids (in order)
        transcript_ids, final_preds, probs = average_logits(span_ids, weighted_span_logits)
        tid2label = dict(zip(test_df["orig_id"], test_df["label"]))
        true_labels = [tid2label[tid] for tid in transcript_ids]
        acc = accuracy_score(true_labels, final_preds)
        all_labels_final = true_labels




    # Get per-model best accuracy if ensemble
    if model_used == "ensemble":
        bert_acc = np.max(bert_accuracies[-top_n_epochs:]) if bert_accuracies else None
        roberta_acc = np.max(roberta_accuracies[-top_n_epochs:]) if roberta_accuracies else None

        print(f"Seed {seed} BERT Accuracy     : {bert_acc:.4f}" if bert_acc is not None else "No BERT accuracy available.")
        print(f"Seed {seed} RoBERTa Accuracy : {roberta_acc:.4f}" if roberta_acc is not None else "No RoBERTa accuracy available.")
        print(f"Seed {seed} Ensemble Accuracy: {acc:.4f}")
    else:
        print(f"Seed {seed} Accuracy: {acc:.4f}")


    report = classification_report(all_labels_final, final_preds, output_dict=True)

    if not span:
        if weighted_majority_voting:
            probs = weighted_logits
        else:
            probs = np.mean(all_preds_models_np, axis=0)
    else:
        # For span mode, probs is already set by average_logits above (in the aggregation block)
        # and refers to per-transcript probabilities after aggregation
        pass  # probs already set

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
    }, step=global_step + 1)
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels_final, final_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    
    def save_ensemble_outputs(
        model_name,              # e.g. "nonmoe" or "moe"
        span_ids,                # list of unique ids (spans or samples, order matches probs/preds)
        probs,                   # numpy array, shape [num_samples, num_classes]
        preds,                   # final predictions, shape [num_samples]
        true_labels,             # ground truth, shape [num_samples]
        val_weights,             # list or numpy array of validation accuracies, shape [num_models_in_ensemble]
        seed,                    # the seed used for this run
        output_dir="ensemble_outputs/nonmoe"   # where to save
    ):
        model_name = f"{model_used}_{plm_size}_{prompt_type}"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"probs_{model_name}_seed{seed}.npy"), probs)
        np.save(os.path.join(output_dir, f"preds_{model_name}_seed{seed}.npy"), preds)
        np.save(os.path.join(output_dir, f"labels_{model_name}_seed{seed}.npy"), true_labels)
        np.save(os.path.join(output_dir, f"ids_{model_name}_seed{seed}.npy"), np.array(span_ids))
        np.save(os.path.join(output_dir, f"val_weights_{model_name}_seed{seed}.npy"), np.array(val_weights))

    if span:
        # transcript_ids, final_preds, probs already computed after aggregation
        output_span_ids = transcript_ids
        output_preds = final_preds
        output_probs = probs
        output_labels = true_labels
    else:
        output_span_ids = list(test_df["id"])
        output_preds = final_preds
        output_probs = probs
        output_labels = all_labels_final



    save_ensemble_outputs(model_name, output_span_ids, output_probs, output_preds, output_labels, val_weights, seed)

    meta = {
        "model_configs": model_configs,
        "prompt_type": prompt_type,
        "plm_size": plm_size,
        "span": span,
        "num_epochs": num_epochs,
        "top_n_epochs": top_n_epochs,
        "weighted_majority_voting": weighted_majority_voting,
        "seed": seed,
        "all_logits_file": f"all_logits_{model_name}_seed{seed}.npy",
        "all_probs_file": f"all_probs_{model_name}_seed{seed}.npy",
        # Add MoE info if relevant
        # "expert_ids_file": f"expert_ids_{model_name}_seed{seed}.npy",
        # "routing_weights_file": f"routing_weights_{model_name}_seed{seed}.npy",
    }

    with open(os.path.join(output_dir, f"meta_{model_name}_seed{seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    run.finish()
    return acc

# ===== Main =====
def main():
    train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR, span = span)
    test_df = load_test_set(span = span)

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
        train_df_full = load_transcripts_with_pauses(TRANSCRIPT_DIR, span = span)
        test_df = load_test_set(span = span)
        gmm_control, gmm_ad = fit_gmms(train_df_full)
        train_df_full = compute_ad_prob(train_df_full, gmm_control, gmm_ad)
        test_df = compute_ad_prob(test_df, gmm_control, gmm_ad)
        train_df, val_df = train_test_split(train_df_full, test_size=0.2, stratify=train_df_full["label"], random_state=42)
        run_pbft(42, train_df, val_df, test_df)
    else:
        main()
