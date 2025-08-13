import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms.mlm import MLMTokenizerWrapper
from peft import PeftModel, PeftConfig
from MOE_LORA import MoEBertSelfAttention, MoERobertaSelfAttention
from spanning import average_logits
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from openprompt.plms import load_plm

MODEL_ROOT = "/home/kej48/rds/hpc-work/Thesis/models_saved"
TRANSCRIPT_TEST_DIR = "pause_encoding/test"
TEST_LABELS_PATH = "data/ADReSS/ADReSS-IS2020-data/test/test_results.txt"
OUTPUT_ROOT = "./inference_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
CONTROL_IDS = {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011", "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034", "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049", "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063", "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077"}
span = False

bert_accuracies = []
roberta_accuracies = []

def detect_model_type(model_dir):
    parent = os.path.basename(os.path.dirname(model_dir.rstrip("/"))).lower()
    if parent.startswith("bert"):
        return "bert"
    elif parent.startswith("roberta"):
        return "roberta"
    else:
        raise ValueError(f"Unknown model type: {parent}")

def segment_and_load_test():
    df_labels = pd.read_csv(TEST_LABELS_PATH, sep=";")
    df_labels.columns = df_labels.columns.str.strip().str.lower()
    id_to_label = dict(zip(df_labels['id'].str.strip().str.upper(), df_labels['label']))
    examples, span_ids = [], []

    for fname in sorted(os.listdir(TRANSCRIPT_TEST_DIR)):
        if not fname.endswith(".txt"):
            continue
        file_id = fname.replace(".txt", "").upper()
        label = id_to_label.get(file_id)
        if label is None:
            continue

        with open(os.path.join(TRANSCRIPT_TEST_DIR, fname)) as f:
            text = f.read().strip()

        pause_score = 0.00
        pause_text = f"The probability of Alzheimer's Disease is {pause_score:.2f}"

        if span:
            from spanning import segment_text
            segments = segment_text(text)
            for i, seg in enumerate(segments):
                seg_id = f"{file_id}_seg{i}"
                examples.append(InputExample(
                    guid=seg_id,
                    text_a=seg,
                    text_b=pause_text,
                    label=int(label)
                ))
                span_ids.append(seg_id)
        else:
            seg_id = file_id
            examples.append(InputExample(
                guid=seg_id,
                text_a=text,
                text_b=pause_text,
                label=int(label)
            ))
            span_ids.append(seg_id)

    return examples, span_ids




def patch_expert_logging(module, expert_logs):
    original_forward = module.forward
    def forward_with_logging(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if hasattr(module, 'last_expert_indices') and module.last_expert_indices is not None:
            expert_logs.append(module.last_expert_indices.detach().cpu().numpy())
        else:
            expert_logs.append(None)
        return output
    module.forward = forward_with_logging

def patch_all_moe_projections(model, expert_logs):
    for name, module in model.named_modules():
        if type(module).__name__ == "MoEAttentionProjection":
            patch_expert_logging(module, expert_logs)

def load_and_patch_plm(peft_dir, expert_logs):
    # 1. Load PEFT config to get base model
    peft_config = PeftConfig.from_pretrained(peft_dir)
    base_model_name = peft_config.base_model_name_or_path
    model_type = detect_model_type(peft_dir)

    # 2. Load PLM using OpenPrompt's util (for tokenizer etc)
    plm, tokenizer, model_config, WrapperClass = load_plm(model_type, base_model_name)

    # 3. Patch backbone with LoRA/PEFT
    model = PeftModel.from_pretrained(plm, peft_dir)

    # 4. Patch with MoE if used (MUST be after PeftModel.from_pretrained)
    if model_type == "bert":
        from transformers.models.bert.modeling_bert import BertSelfAttention
        for name, module in model.named_modules():
            if isinstance(module, BertSelfAttention):
                parent_name, attr = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, attr, MoEBertSelfAttention(model.config))
    elif model_type == "roberta":
        from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
        for name, module in model.named_modules():
            if isinstance(module, RobertaSelfAttention):
                parent_name, attr = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, attr, MoERobertaSelfAttention(model.config))
    patch_all_moe_projections(model, expert_logs)
    return model, tokenizer, model_type

def evaluate_model(model, tokenizer, test_data, template_text, expert_logs, model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    template = ManualTemplate(text=template_text, tokenizer=tokenizer)
    verbalizer = ManualVerbalizer(
        classes=["alzheimer", "healthy"],
        label_words={"alzheimer": ["alzheimer's"], "healthy": ["healthy"]},
        tokenizer=tokenizer
    )
    dataloader = PromptDataLoader(
        dataset=test_data,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=MLMTokenizerWrapper,
        max_seq_length=512,
        batch_size=1,
        shuffle=False
    )
    prompt_model = PromptForClassification(template=template, plm=model, verbalizer=verbalizer).to(device)
    pt_path = os.path.join(model_dir, "prompt_model.pt")
    state = torch.load(pt_path, map_location=device)
    missing, unexpected = prompt_model.load_state_dict(state, strict=False)

    prompt_model.eval()
    all_logits, all_labels, all_ids, experts, texts, logit_0s, logit_1s = [], [], [], [], [], [], []
    guid_to_text = {ex.guid: ex.text_a for ex in test_data}
    for batch in dataloader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.no_grad():
            logits = prompt_model({k: v for k, v in batch.items() if k != "label"})
            # === MoE logging here ===
            if expert_logs:
                expert_entry = expert_logs.pop(0)
                # Save as string for CSV/debugging
                if isinstance(expert_entry, np.ndarray):
                    experts.append(str(expert_entry.tolist()))
                else:
                    experts.append(str(expert_entry))
            else:
                experts.append(None)
            probs = softmax(logits.cpu().numpy().squeeze())
            logit_0s.append(probs[0])
            logit_1s.append(probs[1])
            all_logits.append(logits.cpu().numpy().squeeze())
            all_labels.extend(batch["label"].cpu().numpy())
            all_ids.extend(batch["guid"])
            texts.extend([guid_to_text[i] for i in batch["guid"]])
    return np.vstack(all_logits), all_labels, all_ids, experts, texts, logit_0s, logit_1s


def main():
    test_data, span_ids = segment_and_load_test()
    # ENSEMBLE LISTS
    all_logits_by_model = []
    all_labels_by_model = []
    model_types_list = []
    span_logits_list = []
    span_ids_list = []
    # === Build group mapping (MODEL_GROUP/epochX) ===
    # model_dirs = sorted(glob(os.path.join(MODEL_ROOT, '*_seed42/epoch*/')))
    # model_group_map = {}
    # for d in model_dirs:
    #     group = os.path.basename(os.path.dirname(d.rstrip("/")))  # e.g., bert_11cbed_seed42
    #     model_group_map.setdefault(group, []).append(d)
    model_group_map = {"roberta_bf1bc2_seed56": ["/home/kej48/rds/hpc-work/Thesis/models_saved/roberta_bf1bc2_seed56/epoch16"]}


    # === Evaluation + Saving per model and group ===
    all_logits_by_model = []
    all_labels_by_model = []
    model_types_list = []
    span_logits_list = []
    span_ids_list = []

    for group, paths in model_group_map.items():
        for model_dir in tqdm(paths, desc=f"Evaluating {group}"):
            expert_logs = []
            model, tokenizer, model_type = load_and_patch_plm(model_dir, expert_logs)
            meta_path = os.path.join(model_dir, "meta.txt")
            with open(meta_path) as f:
                lines = f.readlines()
            template_text = next((l.replace("template:", "").strip() for l in lines if l.startswith("template:")), None)
            if template_text is None:
                raise ValueError(f"No template found in {meta_path}")

            logits, labels, ids, experts, texts, logit_0s, logit_1s = evaluate_model(
                model, tokenizer, test_data, template_text, expert_logs, model_dir)
            probs = softmax(logits, axis=1)
            transcript_ids, preds, avg_probs = average_logits(span_ids, probs)
            seen = {}
            for sid, lbl in zip(span_ids, labels):
                tid = sid.split("_seg")[0]
                if tid not in seen:
                    seen[tid] = lbl
            true_labels = [seen[tid] for tid in transcript_ids]

            acc = accuracy_score(true_labels, preds)
            print(f"[{model_type}] {model_dir} Accuracy: {acc:.4f}")

            # Store for ensemble
            all_logits_by_model.append(avg_probs)
            all_labels_by_model.append(true_labels)
            model_types_list.append(model_type)
            span_logits_list.append(probs)
            span_ids_list.append(transcript_ids)

            # Save CSVs, broken out by group/epoch
            out_base = os.path.join(OUTPUT_ROOT, group)
            os.makedirs(out_base, exist_ok=True)
            epoch_name = os.path.basename(model_dir.rstrip("/"))
            pd.DataFrame({
                "transcript_id": transcript_ids,
                "true_label": true_labels,
                "pred_label": preds,
                "prob_0": avg_probs[:, 0],
                "prob_1": avg_probs[:, 1],
                "model_type": model_type,
                "template_version": template_text
            }).to_csv(os.path.join(out_base, f"results_{epoch_name}.csv"), index=False)
            pd.DataFrame({
                "segment_id": ids,
                "segment_text": texts,
                "true_label": labels,
                "expert": experts,
                "logit_0": logit_0s,
                "logit_1": logit_1s
            }).to_csv(os.path.join(out_base, f"expert_routing_{epoch_name}.csv"), index=False)

            # Track accuracy by model type
            if model_type == "bert":
                bert_accuracies.append(acc)
            elif model_type == "roberta":
                roberta_accuracies.append(acc)

    # === ENSEMBLE (Weighted Majority Voting) ===
    if all_logits_by_model:
        logits_stack = np.stack(all_logits_by_model)
        weights = np.ones(len(all_logits_by_model)) / len(all_logits_by_model)
        ensemble_logits = np.average(logits_stack, axis=0, weights=weights)
        final_preds = np.argmax(ensemble_logits, axis=1)
        true_labels = all_labels_by_model[0]
        acc = accuracy_score(true_labels, final_preds)
        print(f"[Ensemble] Accuracy: {acc:.4f}")
        out_base = OUTPUT_ROOT
        pd.DataFrame({
            "transcript_id": span_ids_list[0],
            "true_label": true_labels,
            "ensemble_pred": final_preds,
            "prob_0": ensemble_logits[:, 0],
            "prob_1": ensemble_logits[:, 1]
        }).to_csv(os.path.join(out_base, "ensemble_results.csv"), index=False)

    print("\n========== PBFT Inference Summary ==========")
    if bert_accuracies:
        print(f"[BERT]    Mean: {np.mean(bert_accuracies):.4f}, Std: {np.std(bert_accuracies):.4f}, Max: {np.max(bert_accuracies):.4f}")
    else:
        print("[BERT]    No results recorded.")
    if roberta_accuracies:
        print(f"[RoBERTa] Mean: {np.mean(roberta_accuracies):.4f}, Std: {np.std(roberta_accuracies):.4f}, Max: {np.max(roberta_accuracies):.4f}")
    else:
        print("[RoBERTa] No results recorded.")
    if all_logits_by_model:
        print(f"[Ensemble] Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

