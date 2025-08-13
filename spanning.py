
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
import os
from collections import defaultdict, Counter
import numpy as np

DEFAULT_MAX_SENTENCES_PER_SEGMENT = 5
MAX_SEGMENTS = 8  # cap on total segments

def segment_text(text, max_sentences=DEFAULT_MAX_SENTENCES_PER_SEGMENT, max_segments=MAX_SEGMENTS):
    sentences = sent_tokenize(text)
    raw_segments = [
        " ".join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]
    
    if len(raw_segments) <= max_segments:

        return raw_segments
    else:
        # Merge extra segments into the final segment
        allowed = raw_segments[:max_segments - 1]
        merged_last = " ".join(raw_segments[max_segments - 1:])

        return allowed + [merged_last]
    
    
def average_logits(span_ids, logits):
    logits_by_transcript = defaultdict(list)
    for logit, span_id in zip(logits, span_ids):
        base_id = span_id.split("_seg")[0]
        logits_by_transcript[base_id].append(logit)

    transcript_ids = []
    averaged_logits = []
    for tid in sorted(logits_by_transcript.keys()):
        logit_stack = np.vstack(logits_by_transcript[tid])
        avg_logit = logit_stack.mean(axis=0)
        transcript_ids.append(tid)
        averaged_logits.append(avg_logit)

    averaged_logits = np.vstack(averaged_logits)
    final_preds = np.argmax(averaged_logits, axis=1)
    return transcript_ids, final_preds, averaged_logits

# TRANSCRIPT_DIR = "well there's a mother standing there washing the dishes and the sink is overflowing . and the window's open . and outside the window there's a curved walk with a garden . , and you can see another building there . looks like a garage or something with curtains and the grass in the garden . , and there are two cups and a saucer on the sink . and she's getting her feet wet from the overflow of the water from the sink . she seems to be oblivious to the fact that the sink is overflowing . she's also oblivious to the fact that her kids are stealing cookies out of the cookie jar . and the kid on the stool is gonna fall off the stool . , he's standing up there in the cupboard taking cookies out of the jar , handing them to a girl about the same age . the kids are somewhere around seven or eight years old or nine . and the mother is gonna get shocked when he tumbles and the cookie jar comes down . and I think that's about all ." 


# # TRANSCRIPT_DIR = "are you ready ? . well the sink is overflowing . mother is standing in the water like a jerk . she's wiping the dishes also like a jerk . the boy is trying to get a cookie out_of the cookie jar but boy he's about to fall off the stool . his sister has her finger up to her mouth like she's saying . shh . to be quiet . don't let mother know what you're doing . and he's about to hand her a cookie . but in a few moments it's going to be like total catastrophe . the reason the water's flowing out over the sink is because the water is running furiously . and I'm looking out through the window . and I don't see anything going on out there . I don't . that's just a bush I'm presuming or a plum pudding . , I told you the stools about to go over . in a moment there's going to be real chaos which will make what's going on in the picture look like nothing . . the cookie jar is full . the lid is off the cookie jar . and the . do you want me to tell you all of those things ? . the cabinet door has just swung open . stool is about to fall . I guess I've just told you that . with a terrible crash . mother is daydreaming . she doesn't even know what's going on behind her . I think that's very important and sometimes typical . seems to be all I can see ."
# TRANSCRIPT_DIR = "Okay, here's the picture. ... I'm drying the dishes with water going out of the sink under the floor. ... The previous day I found a lot of flowers. . There were three dishes left to wash. . I was getting the water. . It was dangerous. . The children are getting into the cookie jar. The boys are sharing the food. . I saw most of them over there. . One cookie and a canning. The little girl and the canning mother and the little girl. ... Did you say asking? ... The little girl was just coming out of the canning. . It looks like she's going to be in the closet. ... The water is splashing under the floor. . Perfect. Good."
# segment_text(TRANSCRIPT_DIR, 5, 8)



# === Configuration for testing ===
# import os
# TRANSCRIPT_DIR = "pause_encoding/train"  # or your desired path
# MAX_SENTENCES_PER_SEGMENT = 5  # Change this to test different granularity

# def segment_text(text, max_sentences=MAX_SENTENCES_PER_SEGMENT):
#     sentences = sent_tokenize(text)
#     segments = [
#         " ".join(sentences[i:i + max_sentences])
#         for i in range(0, len(sentences), max_sentences)
#     ]
#     return segments

# def test_transcript_segmentation(transcript_dir):
#     for fname in sorted(os.listdir(transcript_dir)):
#         if not fname.endswith(".txt"):
#             continue
#         file_path = os.path.join(transcript_dir, fname)
#         with open(file_path, "r") as f:
#             text = f.read().strip()
#         segments = segment_text(text)
#         print(f"\n=== {fname} ===")
#         print(f"Original Length: {len(sent_tokenize(text))} sentences")
#         print(f"Segmented into {len(segments)} segments\n")
#         for i, seg in enumerate(segments):
#             print(f"[Segment {i+1}]: {seg[:]}")
#         print("="*60)

# if __name__ == "__main__":
#     test_transcript_segmentation(TRANSCRIPT_DIR)
