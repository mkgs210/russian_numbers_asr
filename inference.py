# inference.py

import os
import csv
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.functional import char_error_rate
from torchaudio.functional import resample

from src.my_beam_search import ctc_beam_search_fsa, tokens_to_number_string
from src.models import ASRLightningConformer
from src.datasets import CustomNumbersASRDataset

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ─────────── SETTINGS ───────────
CHECKPOINT_PATH = "./mlruns/156786290563127125/72c47862908346c8a8dbcde91ea3bebf/checkpoints/epoch=29-step=47100.ckpt"
DEV_CSV         = "./data/dev.csv"
AUDIO_DIR       = "./data"
OUT_DIR         = "./inference_outputs"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR       = 16000
BEAM_WIDTH      = 5
BLANK_IDX       = 0

# Сколько примеров сохранить
MAX_SAVED_EXAMPLES = 2

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────── Feature transforms ───────────
mel_transform = MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=400,
    hop_length=160,
    n_mels=80
)
amp2db = AmplitudeToDB()

# ─────────── Load model ───────────
model = ASRLightningConformer.load_from_checkpoint(CHECKPOINT_PATH)
model.eval().to(DEVICE)

# ─────────── Inference helpers ───────────
def load_and_preprocess(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav[:1]
    if sr != TARGET_SR:
        wav = resample(wav, sr, TARGET_SR)
    mel = mel_transform(wav)          # [1, n_mels, T]
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
        toks = ["<1>","|"] + toks[1:]
    return toks

# ─────────── Run inference ───────────
total_chars = 0
total_err_g = 0.0
total_err_b = 0.0
saved = 0

# прочитаем все строки, чтобы tqdm знал длину
with open(DEV_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in tqdm(rows, desc="Inference"):
    fn       = row["filename"]
    ref_str  = str(row["transcription"]).strip()
    wav_path = os.path.join(AUDIO_DIR, fn)

    spec = load_and_preprocess(wav_path)      # [time, n_mels]
    T    = spec.size(0)

    # forward
    x       = spec.unsqueeze(0).to(DEVICE)    # [1, T, n_mels]
    lengths = torch.tensor([T], device=DEVICE)
    with torch.no_grad():
        logits, _ = model(x, lengths)         # [time, 1, C]
    logits_ts = logits[:,0,:]                 # [time, C]

    # GREEDY
    toks_g = greedy_decode(logits_ts)
    num_g  = tokens_to_number_string(toks_g)

    # BEAM
    logp   = torch.log_softmax(logits_ts, dim=-1).cpu()
    beams  = ctc_beam_search_fsa(logp, beam_width=BEAM_WIDTH, blank=BLANK_IDX)
    seq_b, _ = beams[0]
    toks_b = [model.idx_to_vocab[i] for i in seq_b]
    if toks_b and toks_b[0] == "|":
        toks_b = ["<1>","|"] + toks_b[1:]
    num_b  = tokens_to_number_string(toks_b)

    # CER (по цифрам)
    ce_g = char_error_rate([num_g], [ref_str]).item()
    ce_b = char_error_rate([num_b], [ref_str]).item()
    total_chars += len(ref_str)
    total_err_g  += ce_g * len(ref_str)
    total_err_b  += ce_b * len(ref_str)

    # сохранить пару спектрограмм с подписями
    if saved < MAX_SAVED_EXAMPLES:
        plt.figure(figsize=(8,4))
        plt.imshow(spec.transpose(0,1).cpu(), aspect='auto', origin='lower', cmap='inferno')
        plt.title(
            f"REF: {ref_str}\n"
            f"GREEDY → {num_g} (CER={ce_g:.3f})\n"
            f"BEAM   → {num_b} (CER={ce_b:.3f})",
            fontsize=8
        )
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        base, _ = os.path.splitext(fn)
        out_png = os.path.join(OUT_DIR, f"{base}.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        plt.savefig(out_png, dpi=150)
        plt.close()
        saved += 1

# ─────────── Print final CER ───────────
print("\n=== Overall CER ===")
print(f"Greedy CER: {total_err_g/total_chars:.4f}")
print(f" Beam  CER: {total_err_b/total_chars:.4f}")