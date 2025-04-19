import os
import random
from abc import ABC, abstractmethod
import pandas as pd
import torch
import torchaudio
from num2words import num2words
from torch.utils.data import Dataset
from torchaudio.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    Resample,
)


class BaseASRDataset(Dataset, ABC):
    def __init__(self, csv_path, audio_dir, target_sample_rate=16000, transforms=None):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.transforms = transforms

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
        """Return the dataset's vocabulary as a list or string of tokens."""
        pass

    @abstractmethod
    def process_text(self, text: str):
        """Process raw text into a sequence of tokens (characters or custom tokens)."""
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

        if self.transforms:
            waveform = waveform * random.uniform(0.8, 1.2)

        mel_spec = self.mel_transform(waveform)

        if self.transforms:
            mel_spec = self.transforms(mel_spec)

        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
        return mel_spec

    def _preprocess_target(self, target: str) -> torch.Tensor:
        processed_text = self.process_text(target)
        encoded = [
            self.vocab_to_idx[c] for c in processed_text if c in self.vocab_to_idx
        ]
        return torch.tensor(encoded, dtype=torch.long)


class NumericASRDataset(BaseASRDataset):
    def get_vocab(self):
        return "0123456789"

    def process_text(self, text: str):
        return text.strip()


class RussianWordsASRDataset(BaseASRDataset):
    def get_vocab(self):
        return "абвгдеёжзийклмнопрстуфхцчшщъыьэюя -"

    def process_text(self, text: str):
        return num2words(int(text), lang="ru").strip()


class CustomNumbersASRDataset(BaseASRDataset):
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
        assert 1000 <= int(text) <= 999999

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


def collate_fn(batch):
    specs, targets = zip(*batch)
    specs = [s.squeeze(0).transpose(0, 1) for s in specs]
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
