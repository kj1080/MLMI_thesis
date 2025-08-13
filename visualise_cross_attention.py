import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModel
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader

# === SETTINGS ===
EXPERTS_DIR = "/home/kej48/rds/hpc-work/Thesis/experts"
OUTPUT_DIR = os.path.join(EXPERTS_DIR, "attention_visualise2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === SAMPLE INPUT ===
example = InputExample(
    guid=0,
    # text_a="Tell me everything that you see going on in that picture, everything that you see happening. Well, there's a mother standing there washing the dishes and the sink is over spilling. And the window is open and outside the window there's a walk, a curved walk with a garden. And you can see another building there, looks like a garage or something with curtains and grass in the garden. And there are dishes, two cups and a saucer on the sink. And she's getting her feet wet from the overflow of the water from the sink. She seems to be oblivious to the fact that the sink is overflowing. She's also oblivious to the fact that her kids are stealing cookies out of a cookie jar. And the kid on the stool is going to fall off the stool. He's standing up there in the cupboard, taking cookies out of the jar, handing them to his girl about the same age. The kids are somewhere around seven or eight years old or nine. And the mother is going to get shocked when he tumbles and the cookie jar comes down. And I think that's about all.",
    text_a= "What's going on in the shop? ... We have a lot of women that are in the shop, but I don't think she's that kind of person. She's just a kid, and she hasn't seen you, and you know, there's a big turn on the back of the list. Then go back to the dishes. ... She's deciding it. If she did see that, she's deciding that she's going to let you go home. But I don't think she'll let you go home if you decide you're going to go to the farm. Maybe not her. But she will. She's got a lot of things. She's got a lot of things. And when it's night, she falls. She falls. And with the water on the floor, she's down on the floor. And then, three seconds. And she's going to be on the floor. ... Okay, great.",
    label=0,
)

TEMPLATE_TEXT = 'Based on their speech patterns within this transcript: {"placeholder":"text_a"} , this individual is {"mask"}.'
LABEL_WORDS = {"alzheimer": ["alzheimer's"], "healthy": ["healthy"]}


# === PROCESS EACH MODEL ===
for dirname in sorted(os.listdir(EXPERTS_DIR)):
    expert_path = os.path.join(EXPERTS_DIR, dirname)
    if not os.path.isdir(expert_path):
        continue

    print(f"Processing {dirname}...")
    attention_storage = []

    # Load components
    try:
        from openprompt.plms import load_plm

        # Detect model type from dir name (e.g., "bert_bert-base-uncased_seed42_epoch10")
        if dirname.startswith("bert"):
            model_type = "bert"
        elif dirname.startswith("roberta"):
            model_type = "roberta"
        else:
            print(f"Unknown model type for {dirname}, skipping...")
            continue

        plm, tokenizer, _, WrapperClass = load_plm(model_type, expert_path)

    except Exception as e:
        print(f"Failed to load {dirname}: {e}")
        continue


    # Prompt setup
    template = ManualTemplate(text=TEMPLATE_TEXT, tokenizer=tokenizer)
    verbalizer = ManualVerbalizer(classes=["alzheimer", "healthy"], label_words=LABEL_WORDS, tokenizer=tokenizer)
    model = PromptForClassification(template=template, plm=plm, verbalizer=verbalizer).to(device)
    model.eval()

    dataloader = PromptDataLoader(
        dataset=[example],
        tokenizer=tokenizer,
        template=template,
        tokenizer_wrapper_class=WrapperClass,  # use this, not model.tokenizer_wrapper_class
        max_seq_length=512,
        batch_size=1,
        shuffle=False,
    )


    # Get input_ids and attention_mask
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Run raw PLM directly to get attentions
    with torch.no_grad():
        outputs = plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        all_attentions = outputs.attentions  # List of tensors, one per layer


    # === Plot attention ===
    if all_attentions:

        attn = all_attentions[-1].squeeze(0)  # shape: (heads, seq, seq)
        mask_idx = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
        avg_attn = attn[:, mask_idx, :].mean(0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].cpu().tolist())

        plt.figure(figsize=(15, 3))
        # Filter tokens with attention above a threshold (e.g., 0.01)
        threshold = 0.01
        # Filter tokens with attention above a threshold

        significant_tokens = [tok if score > threshold else "" for tok, score in zip(tokens, avg_attn)]

        plt.figure(figsize=(20, 4))
        ax = sns.heatmap(avg_attn[None, :], xticklabels=significant_tokens, yticklabels=["[MASK]"], cmap="viridis")

        # Improve spacing of overlapping labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.title(f"{dirname}: Attention from [MASK]")
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"{dirname}_attn.png")
        plt.savefig(save_path)
        plt.close()
    else:
        print(f"No attention weights captured for {dirname}.")
    
    import csv

    # --- Filter & extract top-K tokens ---
    topk = 30
    top_indices = np.argsort(avg_attn)[-topk:][::-1]
    top_tokens = [tokens[i] for i in top_indices]
    top_scores = [avg_attn[i] for i in top_indices]

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{dirname}_top_tokens.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Token", "AttentionScore"])
        writer.writerows(zip(top_tokens, top_scores))

    # Plot heatmap (highlight top tokens only)
    significant_tokens = [tok if i in top_indices else "" for i, tok in enumerate(tokens)]
    plt.figure(figsize=(20, 4))
    ax = sns.heatmap(avg_attn[None, :], xticklabels=significant_tokens, yticklabels=["[MASK]"], cmap="viridis")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.title(f"{dirname}: Attention from [MASK]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dirname}_attn.png"))
    plt.close()

    # --- Store for comparison ---
    comparison_dict = {
        "model": dirname,
        "tokens": top_tokens,
        "scores": top_scores,
    }

