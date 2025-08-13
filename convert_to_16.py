import os
import subprocess

WAV_DIR = "/home/kej48/rds/hpc-work/Thesis/data/mfa_data/wavs"
OUT_DIR = "/home/kej48/rds/hpc-work/Thesis/data/mfa_data/wavs_16k"
os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(WAV_DIR):
    if fname.endswith(".wav"):
        in_path = os.path.join(WAV_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname)
        subprocess.run(["sox", in_path, "-r", "16000", "-c", "1", "-b", "16", out_path])
