import os
import random

import pandas as pd
import torch
import torchaudio
from num2words import num2words
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample


class NumericASRDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sample_rate=16000, augment=False):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.augment = augment
        self.mel_transform = MelSpectrogram(
            sample_rate=target_sample_rate, n_fft=400, hop_length=160, n_mels=80
        )
        self.amplitude_to_db = AmplitudeToDB()

        self._vocab = "0123456789"
        self.vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(self._vocab)}
        self.idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(self._vocab)}
        self.num_classes = len(self._vocab) + 1  # +1 для blank

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])

        waveform, sample_rate = torchaudio.load(file_path)
        mel_spec = self._preprocess_audio(waveform, sample_rate)

        target = self._preprocess_target(row["transcription"])
        return mel_spec, target

    def _preprocess_audio(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        if sample_rate != self.target_sample_rate:
            resampler = Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
        if self.augment:
            waveform = waveform * random.uniform(0.8, 1.2)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        return mel_spec

    def _preprocess_target(self, target: str) -> torch.Tensor:
        return torch.tensor(self._encode_transcription(target), dtype=torch.long)

    def _encode_transcription(self, text):
        text = str(text).strip()
        return [self.vocab_to_idx[c] for c in text if c in self.vocab_to_idx]


class RussianWordsASRDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sample_rate=16000, augment=False):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.augment = augment
        self.mel_transform = MelSpectrogram(
            sample_rate=target_sample_rate, n_fft=400, hop_length=160, n_mels=80
        )
        self.amplitude_to_db = AmplitudeToDB()

        self._vocab = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя -"
        self.vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(self._vocab)}
        self.idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(self._vocab)}
        self.num_classes = len(self._vocab) + 1  # +1 для blank

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])

        waveform, sample_rate = torchaudio.load(file_path)
        mel_spec = self._preprocess_audio(waveform, sample_rate)

        target = self._preprocess_target(row["transcription"])
        return mel_spec, target

    def _preprocess_audio(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        if sample_rate != self.target_sample_rate:
            resampler = Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
        if self.augment:
            waveform = waveform * random.uniform(0.8, 1.2)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        return mel_spec

    def _preprocess_target(self, target: str) -> torch.Tensor:
        target = num2words(target, lang="ru")
        return torch.tensor(self._encode_transcription(target), dtype=torch.long)

    def _encode_transcription(self, text):
        text = str(text).strip()
        return [self.vocab_to_idx[c] for c in text if c in self.vocab_to_idx]


def collate_fn(batch):
    specs, targets = zip(*batch)
    specs = [s.squeeze(0).transpose(0, 1) for s in specs]  # [time, 80]
    spec_lengths = [s.shape[0] for s in specs]
    max_spec_len = max(spec_lengths)
    batch_size = len(specs)
    n_mels = specs[0].shape[1]
    padded_specs = torch.zeros(batch_size, max_spec_len, n_mels)
    for i, s in enumerate(specs):
        padded_specs[i, : s.shape[0], :] = s
    target_lengths = [len(t) for t in targets]
    targets_concat = torch.cat(targets)
    return (
        padded_specs,
        torch.tensor(spec_lengths, dtype=torch.long),
        targets_concat,
        torch.tensor(target_lengths, dtype=torch.long),
    )
