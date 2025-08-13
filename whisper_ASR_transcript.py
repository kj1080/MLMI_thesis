from pyannote.audio import Pipeline

# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
# import os
# import imageio_ffmpeg
# import whisperx

# # set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = whisperx.load_model(whisper_model, device=device)



# ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
# print("ImageIO ffmpeg path:", ffmpeg_path)
# print("Exists:", os.path.exists(ffmpeg_path))

# os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]
# os.environ["FFMPEG_BINARY"] = ffmpeg_path
# os.environ["WHISPER_FFMPEG_PATH"] = ffmpeg_path  # <-- Add this new line

# import whisper
# from tqdm import tqdm
# import subprocess
# import torch
# import tempfile
# import soundfile as sf
# from pyannote.audio import Pipeline
# from pydub import AudioSegment

# fine_tune = False
# participant_only = False
# pause_encoding = True

# # === Settings ===
# whisper_model = "x"  # or "medium" if resources limited
# if fine_tune:
#     if participant_only:
#         output_base = "transcript_asr_whisper_ft_participant_only"
#     output_base = "transcript_asr_whisper_ft"
    
# else:
#     if participant_only:
#         if pause_encoding == False:
#             output_base = 'transcript_asr_whisper_not_ft_participant_only'
#         else:
#             output_base = 'transcript_asr_whisper_not_ft_participant_only_PE'

#     else:
#         if pause_encoding == False:
#             output_base = "transcript_asr_whisper_non_ft"
#         else:
#             if whisper_model == 'medium':
#                 output_base = 'transcript_asr_whisper_not_ft_both_speakers_PE_medium'
#             elif whisper_model == 'x':
#                 output_base = 'transcript_asr_both_speakers_PE_WhisperX'
#             else:
#                 output_base = 'transcript_asr_whisper_not_ft_both_speakers_only_PE'
        

# audio_base_train = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio"
# audio_base_test = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/Full_wave_enhanced_audio"

# # === Load Whisper ===
# # model = whisper.load_model(whisper_model)

# # === Helper ===
# def transcribe_and_save(audio_path, output_dir):
#     filename = os.path.splitext(os.path.basename(audio_path))[0].upper()
#     out_path = os.path.join(output_dir, f"{filename}.txt")
#     if os.path.exists(out_path):  # skip if already done
#         return
#     result = model.transcribe(audio_path, language="en", fp16=True)
#     with open(out_path, "w") as f:
#         f.write(result["text"].strip())
        
# def transcribe_with_pause_encoding(audio_path, output_dir, medium_pause=0.5, long_pause=2.0):
#     filename = os.path.splitext(os.path.basename(audio_path))[0].upper()
#     out_path = os.path.join(output_dir, f"{filename}.txt")
#     if os.path.exists(out_path):
#         return

#     result = model.transcribe(audio_path, language="en", fp16=torch.cuda.is_available(), verbose=False)
#     segments = result.get("segments", [])
#     final_text = ""
#     last_end = 0.0

#     for seg in segments:
#         start = seg["start"]
#         gap = start - last_end
#         if gap >= long_pause:
#             final_text += " ... "
#         elif gap >= medium_pause:
#             final_text += " . "

#         final_text += seg["text"].strip()
#         last_end = seg["end"]

#     with open(out_path, "w") as f:
#         f.write(final_text.strip())


###

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
import os
import imageio_ffmpeg
import torch
import subprocess
import tempfile
import soundfile as sf
from tqdm import tqdm
from pyannote.audio import Pipeline
from pydub import AudioSegment

fine_tune = False
participant_only = False
pause_encoding = True

# === Settings ===
whisper_model = "x"  # options: 'x' for WhisperX, 'medium', 'large'

# Set ffmpeg path
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print("ImageIO ffmpeg path:", ffmpeg_path)
print("Exists:", os.path.exists(ffmpeg_path))
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]
os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["WHISPER_FFMPEG_PATH"] = ffmpeg_path

# Load Whisper or WhisperX
if whisper_model == "x":
    import whisperx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(whisper_model, device=device)
else:
    import whisper
    model = whisper.load_model(whisper_model)

# Output directory naming
if fine_tune:
    if participant_only:
        output_base = "transcript_asr_whisper_ft_participant_only"
    output_base = "transcript_asr_whisper_ft"
else:
    if participant_only:
        if not pause_encoding:
            output_base = 'transcript_asr_whisper_not_ft_participant_only'
        else:
            output_base = 'transcript_asr_whisper_not_ft_participant_only_PE'
    else:
        if not pause_encoding:
            output_base = "transcript_asr_whisper_non_ft"
        else:
            if whisper_model == 'medium':
                output_base = 'transcript_asr_whisper_not_ft_both_speakers_PE_medium'
            elif whisper_model == 'x':
                output_base = 'transcript_asr_both_speakers_PE_WhisperX'
            else:
                output_base = 'transcript_asr_whisper_not_ft_both_speakers_only_PE'

# Audio directories
audio_base_train = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio"
audio_base_test = "/home/kej48/rds/hpc-work/Thesis/data/ADReSS/ADReSS-IS2020-data/test/Full_wave_enhanced_audio"

# === Helper Functions ===
def transcribe_and_save(audio_path, output_dir):
    filename = os.path.splitext(os.path.basename(audio_path))[0].upper()
    out_path = os.path.join(output_dir, f"{filename}.txt")
    if os.path.exists(out_path):
        return
    if whisper_model == "x":
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
    else:
        result = model.transcribe(audio_path, language="en", fp16=True)
    with open(out_path, "w") as f:
        f.write(result["text"].strip())

def transcribe_with_pause_encoding(audio_path, output_dir, medium_pause=0.5, long_pause=2.0):
    filename = os.path.splitext(os.path.basename(audio_path))[0].upper()
    out_path = os.path.join(output_dir, f"{filename}.txt")
    if os.path.exists(out_path):
        return
    if whisper_model == "x":
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
    else:
        result = model.transcribe(audio_path, language="en", fp16=torch.cuda.is_available(), verbose=False)
    segments = result.get("segments", [])
    final_text = ""
    last_end = 0.0
    for seg in segments:
        start = seg["start"]
        gap = start - last_end
        if gap >= long_pause:
            final_text += " ... "
        elif gap >= medium_pause:
            final_text += " . "
        final_text += seg["text"].strip()
        last_end = seg["end"]
    with open(out_path, "w") as f:
        f.write(final_text.strip())
###



if fine_tune:
    # insert ft scripts here later
    x = 1
    
    
# Transcripts - Not Fine Tuned
else:
    
    if participant_only:
        if pause_encoding == False:
            def transcribe_and_save_patient_only(audio_path, output_dir):
                filename = os.path.splitext(os.path.basename(audio_path))[0].upper()
                out_path = os.path.join(output_dir, f"{filename}.txt")
                if os.path.exists(out_path):
                    return
                patient_audio = diarize_and_filter(audio_path)
                if patient_audio is None:
                    print(f"No patient audio found in {audio_path}")
                    return
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                    patient_audio.export(temp_wav.name, format="wav")
                    result = model.transcribe(temp_wav.name, language="en", fp16=torch.cuda.is_available())
                    with open(out_path, "w") as f:
                        f.write(result["text"].strip())
        elif pause_encoding:
            def transcribe_and_save_patient_only(audio_path, output_dir, medium_pause=0.5, long_pause=2):
                filename = os.path.splitext(os.path.basename(audio_path))[0].upper()
                out_path = os.path.join(output_dir, f"{filename}.txt")
                if os.path.exists(out_path):
                    return

                patient_audio = diarize_and_filter(audio_path)
                if patient_audio is None:
                    print(f"No patient audio found in {audio_path}")
                    return

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                    patient_audio.export(temp_wav.name, format="wav")
                    result = model.transcribe(temp_wav.name, language="en", fp16=torch.cuda.is_available(), verbose=False)

                    segments = result.get("segments", [])
                    final_text = ""
                    last_end = 0.0

                    for seg in segments:
                        start = seg["start"]
                        gap = start - last_end

                        if gap >= long_pause:
                            final_text += " ... "
                        elif gap >= medium_pause:
                            final_text += " . "

                        final_text += seg["text"].strip()
                        last_end = seg["end"]

                    with open(out_path, "w") as f:
                        f.write(final_text.strip())


        # Load Models
        model = whisper.load_model(whisper_model)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

        # Helper
        def diarize_and_filter(audio_path, speaker="SPEAKER_00"):
            diarization = pipeline(audio_path)
            audio = AudioSegment.from_wav(audio_path)
            segments = [audio[int(1000 * turn.start):int(1000 * turn.end)]
                        for turn, _, spk in diarization.itertracks(yield_label=True)
                        if spk == speaker]
            return sum(segments) if segments else None

        

        # Main
        def process_folder(audio_dir, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
            print(f"Processing {len(audio_files)} files in {audio_dir}...")
            for fname in tqdm(audio_files):
                audio_path = os.path.join(audio_dir, fname)
                transcribe_and_save_patient_only(audio_path, output_dir)

        for subset in ["cc", "cd"]:
            process_folder(os.path.join(audio_base_train, subset), os.path.join(output_base, "train"))

        process_folder(audio_base_test, os.path.join(output_base, "test"))

        print(f"Whisper model: {whisper_model}")
        print(f"Fine-tuned: {fine_tune}")
        print(f"Participant-only: {participant_only}")
        print(f"Pause encoding: {pause_encoding}")
        print("\n Process Complete. Patient-only transcripts saved to `transcript_asr_whisper_patient_only/`")

    else:
        
        # === Process Train ===
        for subset in ["cc", "cd"]:
            audio_dir = os.path.join(audio_base_train, subset)
            output_dir = os.path.join(output_base, "train")
            os.makedirs(output_dir, exist_ok=True)
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
            print(f"Processing {subset} with {len(audio_files)} files...")
                
            for fname in tqdm(audio_files, desc=f"{subset}"):
                audio_path = os.path.join(audio_dir, fname)
                if pause_encoding:
                    transcribe_with_pause_encoding(audio_path, output_dir)
                else:
                    transcribe_and_save(audio_path, output_dir)


        # === Process Test ===
        test_output_dir = os.path.join(output_base, "test")
        os.makedirs(test_output_dir, exist_ok=True)
        test_files = [f for f in os.listdir(audio_base_test) if f.endswith(".wav")]
        print(f"Processing TEST with {len(test_files)} files...")
        for fname in tqdm(test_files, desc="test"):
            audio_path = os.path.join(audio_base_test, fname)
            transcribe_and_save(audio_path, test_output_dir)

        print(f"Whisper model: {whisper_model}")
        print(f"Fine-tuned: {fine_tune}")
        print(f"Participant-only: {participant_only}")
        print(f"Pause encoding: {pause_encoding}")

        print("\nAll done. Transcripts saved to `transcript_asr_whisper_non_ft/`")


