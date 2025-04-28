#!/usr/bin/env python3
# sample_submission.py

import os
import csv
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchaudio.functional import resample
from tqdm import tqdm

from src.my_beam_search import tokens_to_number_string
from src.models import ASRLightningConformer

# ─────────── SETTINGS ───────────
#CHECKPOINT_PATH = "./mlruns/156786290563127125/72c47862908346c8a8dbcde91ea3bebf/checkpoints/epoch=29-step=47100.ckpt"   # 21.746
CHECKPOINT_PATH = './mlruns/156786290563127125/d2137feeef3949b7b82bf092866cfe01/checkpoints/epoch=56-step=89490.ckpt'
TEST_CSV        = "./data/test.csv"          # CSV со столбцом "filename"
AUDIO_DIR       = "./data"                   # папка, где лежит test/…
OUTPUT_CSV      = "sample_submission.csv"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR       = 16000
BLANK_IDX       = 0

# ─────────── Feature transforms ───────────
mel_transform = MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=400, hop_length=160, n_mels=80
)
amp2db = AmplitudeToDB()

# ─────────── Load model ───────────
model = ASRLightningConformer.load_from_checkpoint(CHECKPOINT_PATH)
model.eval().to(DEVICE)

# ─────────── Helpers ───────────
def load_and_preprocess(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav[:1]
    if sr != TARGET_SR:
        wav = resample(wav, sr, TARGET_SR)
    mel = mel_transform(wav)
    log_mel = amp2db(mel)
    norm = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    spec = norm.squeeze(0).transpose(0, 1)  # [time, n_mels]
    return spec

def greedy_decode(logits_ts: torch.Tensor):
    pred_ids = logits_ts.argmax(dim=-1).cpu().tolist()
    toks, prev = [], BLANK_IDX
    for idx in pred_ids:
        if idx != prev and idx != BLANK_IDX:
            toks.append(model.idx_to_vocab[idx])
        prev = idx
    if toks and toks[0] == "|":
        toks = ["<1>", "|"] + toks[1:]
    return toks

# ─────────── Inference & write submission ───────────
with open(TEST_CSV, newline="", encoding="utf-8") as f_in, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:

    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    writer.writerow(["filename", "transcription"])

    for row in tqdm(reader, desc="Generating submission"):
        fn      = row["filename"]
        wavpath = os.path.join(AUDIO_DIR, fn)

        # 1) preprocess
        spec = load_and_preprocess(wavpath)
        T    = spec.size(0)
        x    = spec.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([T], device=DEVICE)

        # 2) forward
        with torch.no_grad():
            logits, _ = model(x, lengths)
        logits_ts = logits[:,0,:]

        # 3) greedy decode → tokens → number string
        toks = greedy_decode(logits_ts)
        num_str = tokens_to_number_string(toks)

        # 4) write
        writer.writerow([fn, num_str])

print(f"Done! Submission saved to {OUTPUT_CSV}")
