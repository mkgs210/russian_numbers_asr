import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from torchaudio.models import Conformer

from num2words import num2words

# =========================
# 1. Словарь, кодирование и декодирование
# =========================
vocab = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(vocab)}
idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(vocab)}
num_classes = len(vocab) + 1  # 0 зарезервирован для blank

def encode_transcription(text):
    text = str(text).strip()
    # Если текст состоит только из цифр, преобразуем его в слово на русском
    if text.isdigit():
        words = num2words(int(text), lang='ru')
    else:
        words = text
    words = words.lower().strip()
    return [vocab_to_idx[c] for c in words if c in vocab_to_idx]


def decode_ctc(indices):
    decoded = []
    prev = None
    for idx in indices:
        if idx != prev and idx != 0:
            decoded.append(idx)
        prev = idx
    return "".join([idx_to_vocab[i] for i in decoded if i in idx_to_vocab])

def levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

def compute_cer(ref, hyp):
    if len(ref) == 0:
        return 0 if len(hyp)==0 else 1
    return levenshtein_distance(ref, hyp) / len(ref)

# =========================
# 2. Датасет и collate_fn
# =========================
class ASRDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sample_rate=16000, augment=False, noise_std=0.05):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.augment = augment
        self.noise_std = noise_std
        self.mel_transform = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        self.amplitude_to_db = AmplitudeToDB()
        self.normal_specaug = nn.Sequential(
            FrequencyMasking(freq_mask_param=30),
            TimeMasking(time_mask_param=70)
        )
        self.aggressive_specaug = nn.Sequential(
            FrequencyMasking(freq_mask_param=25),
            *[TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
        )

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
        target = torch.tensor(encode_transcription(row["transcription"]), dtype=torch.long)
        if self.augment:
            original = mel_spec.clone()
            normal_aug = self.normal_specaug(mel_spec.clone())
            aggressive_aug = self.aggressive_specaug(mel_spec.clone())
            noise_aug = mel_spec + torch.randn_like(mel_spec) * self.noise_std
            variants = [original, normal_aug, aggressive_aug, noise_aug]
        else:
            variants = [mel_spec]
        return {'variants': variants}, target

def collate_fn(batch):
    specs = []
    targets = []
    spec_lengths = []
    target_lengths = []
    for sample, target in batch:
        variant = random.choice(sample['variants'])
        spec = variant.squeeze(0).transpose(0, 1)  # [time, 80]
        specs.append(spec)
        spec_lengths.append(spec.shape[0])
        targets.append(target)
        target_lengths.append(len(target))
    max_spec_len = max(spec_lengths)
    batch_size = len(specs)
    n_mels = specs[0].shape[1]
    padded_specs = torch.zeros(batch_size, max_spec_len, n_mels)
    for i, s in enumerate(specs):
        padded_specs[i, :s.shape[0], :] = s
    targets_concat = torch.cat(targets)
    return padded_specs, torch.tensor(spec_lengths, dtype=torch.long), targets_concat, torch.tensor(target_lengths, dtype=torch.long)

# =========================
# 3. LightningModule с Conformer, Gradient Checkpointing, BF16 и подсчётом общего CER
# =========================
class ASRLightningConformer(pl.LightningModule):
    def __init__(self,
                 input_dim=80,
                 num_classes=num_classes,
                 num_heads=4,
                 ffn_dim=1024,
                 num_layers=8,
                 depthwise_conv_kernel_size=31,
                 dropout=0.15,
                 learning_rate=0.01,
                 weight_decay=1e-4,
                 subsampling_factor=4):
        super(ASRLightningConformer, self).__init__()
        self.save_hyperparameters()
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False
        )
        self.fc = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.learning_rate = learning_rate
        self.subsampling_factor = subsampling_factor

        # Для подсчёта общего CER по всей валидации
        self.total_edit = 0
        self.total_ref_len = 0

    def forward(self, x, lengths):
        def checkpointed_conformer(inp):
            out, _ = self.conformer(inp, lengths)
            return out
        out = torch.utils.checkpoint.checkpoint(checkpointed_conformer, x, use_reentrant=False)
        out = self.fc(out)
        return out.transpose(0, 1)  # [time_subsampled, batch, num_classes]

    def training_step(self, batch, batch_idx):
        specs, spec_lengths, targets, target_lengths = batch
        output = self.forward(specs, spec_lengths)
        effective_spec_lengths = spec_lengths // self.subsampling_factor
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.criterion(output.float(), targets, effective_spec_lengths, target_lengths)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, spec_lengths, targets, target_lengths = batch
        output = self.forward(specs, spec_lengths)
        effective_spec_lengths = spec_lengths // self.subsampling_factor
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.criterion(output.float(), targets, effective_spec_lengths, target_lengths)
        # Декодирование прогнозов для CER
        output_cpu = output.detach().cpu()
        preds = torch.argmax(output_cpu, dim=2).transpose(0, 1)  # [batch, time_subsampled]
        predictions = [decode_ctc(pred.tolist()) for pred in preds]
        targets_cpu = targets.detach().cpu().tolist()
        refs = []
        idx = 0
        for l in target_lengths.tolist():
            ref = decode_ctc(targets_cpu[idx: idx+l])
            refs.append(ref)
            idx += l
        # Вычисление CER для каждого примера
        for ref, pred in zip(refs, predictions):
            self.total_edit += levenshtein_distance(ref, pred)
            self.total_ref_len += len(ref)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        overall_cer = self.total_edit / self.total_ref_len if self.total_ref_len > 0 else 0.0
        self.log("CER_avg", overall_cer, prog_bar=True, logger=True)
        print(f"Validation CER: {overall_cer:.4f}")
        # Обнуляем накопленные значения
        self.total_edit = 0
        self.total_ref_len = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "CER_avg"}}

# =========================
# 4. Функция main
# =========================
def main():
    train_csv = "./train.csv"
    dev_csv = "./dev.csv"
    audio_dir = "./"
    
    batch_size = 1
    num_epochs = 100
    early_stop_patience = 20
    
    train_dataset = ASRDataset(train_csv, audio_dir, target_sample_rate=16000, augment=True)
    dev_dataset = ASRDataset(dev_csv, audio_dir, target_sample_rate=16000, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    mlflow_logger = MLFlowLogger(experiment_name="ASR_Conformer_Experiment", tracking_uri="file:./mlruns")
    
    asr_module = ASRLightningConformer(
        input_dim=80,
        num_classes=num_classes,
        num_heads=8,
        ffn_dim=768,
        num_layers=12,
        depthwise_conv_kernel_size=31,
        dropout=0.15,
        learning_rate=0.01,
        weight_decay=1e-4,
        subsampling_factor=4
    )
    
    # EarlyStopping и ModelCheckpoint теперь мониторят CER_avg
    early_stop_callback = EarlyStopping(monitor="CER_avg", patience=early_stop_patience, mode="min", verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="CER_avg", mode="min", save_top_k=1, verbose=True)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=mlflow_logger,
        precision="bf16",
        accumulate_grad_batches=4
    )
    
    trainer.fit(asr_module, train_loader, dev_loader)

if __name__ == "__main__":
    main()
