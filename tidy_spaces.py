import os
import re

def fix_spacing_after_periods(text):
    # Add a space after periods that are not followed by a space or pause pattern
    return re.sub(r'(?<!\s)\.(?=[A-Za-z])', '. ', text)

def process_transcripts(input_dir, output_dir=None):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        in_path = os.path.join(input_dir, fname)
        with open(in_path, "r") as f:
            raw_text = f.read()

        fixed_text = fix_spacing_after_periods(raw_text)

        out_path = os.path.join(output_dir, fname) if output_dir else in_path
        with open(out_path, "w") as f:
            f.write(fixed_text)

        print(f"Processed: {fname}")

# === Example usage ===
input_folder = "/home/kej48/rds/hpc-work/Thesis/transcript_asr_whisper_not_ft_both_speakers_PE_medium/train"  
output_folder = None  # set to None to overwrite in place, or give a new path to save elsewhere

process_transcripts(input_folder, output_folder)
