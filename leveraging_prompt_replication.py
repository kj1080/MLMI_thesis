import os
import re
import shutil
import pylangacq

# Input folders
base_transcript_dir = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/transcription"
base_audio_dir = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio"
test_dir = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/transcription"

control_transcript_dir = os.path.join(base_transcript_dir, "cc")
ad_transcript_dir = os.path.join(base_transcript_dir, "cd")

control_audio_dir = os.path.join(base_audio_dir, "cc")
ad_audio_dir = os.path.join(base_audio_dir, "cd")

# Output folders for MFA
mfa_test_data_dir = "data/mfa_test_data/transcripts"
mfa_test_audio_dir = "data/mfa_test_data/wavs"

mfa_data_dir = "data/mfa_data"
mfa_transcript_dir = os.path.join(mfa_data_dir, "transcripts")
mfa_audio_dir = os.path.join(mfa_data_dir, "wavs")

os.makedirs(mfa_transcript_dir, exist_ok=True)
os.makedirs(mfa_audio_dir, exist_ok=True)

def clean_transcript(text):
    # Remove behavioral markers (e.g. &=laughs)
    text = re.sub(r'&=\w+', '', text)
    text = re.sub(r'POSTCLITIC', '', text)
    # Remove symbols and fillers
    text = re.sub(r'\bxxx\b', '', text)
    text = re.sub(r'[&@<>/]', '', text)
    text = re.sub(r'\(\.\.\.\)', '', text)
    text = re.sub(r'\(\.\.\)', '', text)
    text = re.sub(r'\(\.\)', '', text)

    # Expand repetitions like: word [x 3]
    def expand_repeats(match):
        word = match.group(1)
        count = int(match.group(2))
        return ' '.join([word] * count)

    text = re.sub(r'(\w+)\s+\[x (\d+)\]', expand_repeats, text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def cha_to_txt(cha_path, txt_out_path):
    try:
        utts = pylangacq.read_chat(cha_path).utterances()
        par_utts = [u for u in utts if u.participant == "PAR"]

        lines = []
        for u in par_utts:
            raw = " ".join(t.word for t in u.tokens if t.word)
            cleaned = clean_transcript(raw)
            if cleaned:
                lines.append(cleaned)

        with open(txt_out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved: {os.path.basename(txt_out_path)}")

    except Exception as e:
        print(f"Error processing {cha_path}: {e}")

os.makedirs(mfa_test_data_dir, exist_ok=True)

def process_transcripts(source_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(source_folder):
        if fname.endswith(".cha"):
            cha_path = os.path.join(source_folder, fname)
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(output_folder, f"{base}.txt")
            cha_to_txt(cha_path, out_path)

def collect_wavs(source_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(source_folder):
        if fname.endswith(".wav"):
            src = os.path.join(source_folder, fname)
            dst = os.path.join(mfa_audio_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)  # faster and more efficient on HPC
                    # shutil.copy(src, dst)  # Uncomment this line instead if symlinks cause issues

# Process transcripts
# Process training transcripts
process_transcripts(control_transcript_dir, mfa_transcript_dir)
process_transcripts(ad_transcript_dir, mfa_transcript_dir)

# Process test transcripts
process_transcripts(test_dir, mfa_test_data_dir)

# Gather WAVs
# collect_wavs(control_audio_dir)
# collect_wavs(ad_audio_dir)
# collect_wavs(test_dir)




