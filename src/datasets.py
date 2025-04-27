# src/datasets.py
import os
import random
from abc import ABC, abstractmethod

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from num2words import num2words
from torch.utils.data import Dataset
from torchaudio.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    Resample,
    FrequencyMasking,
    TimeMasking,
)


class BaseASRDataset(Dataset, ABC):
    def __init__(self, csv_path, audio_dir, target_sample_rate=16000):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate

        self.mel_transform = MelSpectrogram(
            sample_rate=target_sample_rate, n_fft=400, hop_length=160, n_mels=80
        )
        self.amplitude_to_db = AmplitudeToDB()

        self._vocab = self.get_vocab()
        self.vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(self._vocab)}
        self.idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(self._vocab)}
        self.num_classes = len(self._vocab) + 1  # +1 for blank token

    @abstractmethod
    def get_vocab(self):
        pass

    @abstractmethod
    def process_text(self, text: str):
        pass

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])

        waveform, sample_rate = torchaudio.load(file_path)
        mel_spec = self._preprocess_audio(waveform, sample_rate)
        target = self._preprocess_target(str(row["transcription"]))
        return mel_spec, target

    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
        return mel_spec

    def _preprocess_target(self, target: str) -> torch.Tensor:
        tokens = self.process_text(target)
        encoded = [self.vocab_to_idx[c] for c in tokens if c in self.vocab_to_idx]
        return torch.tensor(encoded, dtype=torch.long)


class CustomNumbersASRDataset(BaseASRDataset):
    def __init__(
        self,
        csv_path,
        audio_dir,
        target_sample_rate=16000,
        noise_std: float = 0.05,
        augment: bool = False,
    ):
        super().__init__(csv_path, audio_dir, target_sample_rate)
        self.noise_std = noise_std
        self.augment = augment

        # Normal SpecAugment
        self.normal_specaugment = nn.Sequential(
            FrequencyMasking(freq_mask_param=30),
            TimeMasking(time_mask_param=70),
        )
        # Aggressive SpecAugment
        self.aggressive_specaugment = nn.Sequential(
            FrequencyMasking(freq_mask_param=25),
            *[TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)],
        )

    def get_vocab(self):
        return [
            "<1>",
            "<2>",
            "<3>",
            "<4>",
            "<5>",
            "<6>",
            "<7>",
            "<8>",
            "<9>",
            "<10>",
            "<20>",
            "<30>",
            "<40>",
            "<50>",
            "<60>",
            "<70>",
            "<80>",
            "<90>",
            "<100>",
            "<200>",
            "<300>",
            "<400>",
            "<500>",
            "<600>",
            "<700>",
            "<800>",
            "<900>",
            "|",
        ]

    def process_text(self, text: str):
        text = text.strip()
        thousands = text[:-3]
        remainder = text[-3:]

        tokens = []
        for place, digit in enumerate(thousands):
            if digit != "0":
                value = int(digit) * (10 ** (len(thousands) - 1 - place))
                tokens.append(f"<{value}>")
        tokens.append("|")
        for place, digit in enumerate(remainder):
            if digit != "0":
                value = int(digit) * (10 ** (2 - place))
                tokens.append(f"<{value}>")
        return tokens

    def __getitem__(self, idx):
        mel_spec, target = super().__getitem__(idx)

        if self.augment:
            variants = [
                mel_spec,  # оригинал
                self.normal_specaugment(mel_spec.clone()),
                self.aggressive_specaugment(mel_spec.clone()),
                mel_spec + torch.randn_like(mel_spec) * self.noise_std,
            ]
            mel_spec = random.choice(variants)

        return mel_spec, target


def collate_fn(batch):
    specs, targets = zip(*batch)
    specs = [s.squeeze(0).transpose(0, 1) for s in specs]
    spec_lengths = [s.shape[0] for s in specs]

    max_len = max(spec_lengths)
    B = len(specs)
    n_mels = specs[0].shape[1]
    padded = torch.zeros(B, max_len, n_mels)
    for i, s in enumerate(specs):
        padded[i, : spec_lengths[i], :] = s

    target_lengths = [t.numel() for t in targets]
    targets_concat = torch.cat(targets)

    return (
        padded,                                          # [B, T_max, n_mels]
        torch.tensor(spec_lengths, dtype=torch.long),    # [B]
        targets_concat,                                  # [sum(target_lengths)]
        torch.tensor(target_lengths, dtype=torch.long),  # [B]
    )
