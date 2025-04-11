import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_lightning.loggers import MLFlowLogger

# Определяем устройство (Lightning сам определяет, если используете Trainer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Определение датасета
# =========================

# Словарь символов
vocab = "0123456789"
vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(vocab)}
idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(vocab)}
num_classes = len(vocab) + 1  # +1 для blank

def encode_transcription(text):
    """Преобразует строку (например, '139473') в последовательность индексов."""
    text = str(text).strip()
    return [vocab_to_idx[c] for c in text if c in vocab_to_idx]

class ASRDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sample_rate=16000, augment=False):
        """
        csv_path: путь к CSV-файлу с метаданными
        audio_dir: директория с аудиофайлами
        target_sample_rate: требуемая частота дискретизации (16 кГц)
        augment: применять ли аугментации
        """
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.augment = augment

        # Преобразования: ресемплирование, получение мел-спектрограммы и перевод в dB
        self.mel_transform = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=400,         # окно ~25 мс при 16 кГц
            hop_length=160,    # шаг ~10 мс
            n_mels=80
        )
        self.amplitude_to_db = AmplitudeToDB()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])
        waveform, sample_rate = torchaudio.load(file_path)

        # Если аудио многоканальное, выбираем первый канал
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)

        # Ресемплирование, если требуется
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Применение аугментации, если указано (например, изменение громкости)
        if self.augment:
            waveform = self.apply_augmentation(waveform)

        # Вычисление мел-спектрограммы и перевод в логарифмическую шкалу
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        # Кодирование транскрипции
        target = torch.tensor(encode_transcription(row["transcription"]), dtype=torch.long)
        return mel_spec, target

    def apply_augmentation(self, waveform):
        """Простая аугментация: случайное изменение громкости."""
        gain = random.uniform(0.8, 1.2)
        return waveform * gain

def collate_fn(batch):
    """
    Формирование батча: паддинг мел-спектрограмм до максимальной длины и объединение меток.
    """
    specs, targets = zip(*batch)
    spec_lengths = [spec.shape[-1] for spec in specs]
    max_spec_len = max(spec_lengths)

    batch_size = len(specs)
    n_mels = specs[0].shape[1]
    padded_specs = torch.zeros(batch_size, 1, n_mels, max_spec_len)

    for i, spec in enumerate(specs):
        length = spec.shape[-1]
        padded_specs[i, :, :, :length] = spec

    target_lengths = [len(t) for t in targets]
    targets_concat = torch.cat(targets)

    return padded_specs, torch.tensor(spec_lengths, dtype=torch.long), targets_concat, torch.tensor(target_lengths, dtype=torch.long)

# ========================
# 2. Определение модели ASR
# ========================

class ASRModel(nn.Module):
    def __init__(self, n_mels=80, num_classes=11, rnn_hidden=128, num_rnn_layers=1, dropout_rate=0.1):
        """
        n_mels: число мел-фильтров
        num_classes: число классов (с blank)
        rnn_hidden: размер скрытого состояния LSTM
        num_rnn_layers: число слоёв LSTM
        """
        super(ASRModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        # Вычисление выходной размерности после свёрточной части (по частоте деление на 4)
        conv_output_size = 64 * (n_mels // 4)
        self.lstm = nn.LSTM(input_size=conv_output_size,
                            hidden_size=rnn_hidden,
                            num_layers=num_rnn_layers,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        """
        x: Tensor [batch, 1, n_mels, time]
        Возвращает: log_probs [time, batch, num_classes] для CTC loss
        """
        x = self.conv(x)  # [batch, channels, freq, time]
        batch_size, channels, freq, time = x.size()
        x = x.permute(3, 0, 1, 2)  # [time, batch, channels, freq]
        x = x.contiguous().view(time, batch_size, channels * freq)
        x, _ = self.lstm(x)
        x = self.fc(x)
        log_probs = nn.functional.log_softmax(x, dim=2)
        return log_probs

# ==========================================
# 3. Обёртка в LightningModule (ASRLightning)
# ==========================================

class ASRLightningModule(pl.LightningModule):
    def __init__(self, n_mels=80, num_classes=num_classes, rnn_hidden=128, 
                num_rnn_layers=1, learning_rate=0.001, dropout_rate=0.1, weight_decay=1e-4):
        super(ASRLightningModule, self).__init__()
        self.save_hyperparameters()  # сохраняем гиперпараметры
        self.model = ASRModel(n_mels=n_mels, num_classes=num_classes,
                              rnn_hidden=rnn_hidden, num_rnn_layers=num_rnn_layers,
                              dropout_rate=dropout_rate)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        specs, spec_lengths, targets, target_lengths = batch
        # Прямой проход
        output = self.model(specs)  # [time, batch, num_classes]
        effective_spec_lengths = spec_lengths // 4  # корректировка из-за maxpool
        loss = self.criterion(output, targets, effective_spec_lengths, target_lengths)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, spec_lengths, targets, target_lengths = batch
        output = self.model(specs)
        effective_spec_lengths = spec_lengths // 4
        loss = self.criterion(output, targets, effective_spec_lengths, target_lengths)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.weight_decay)
        # Используем scheduler ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ======================
# 4. Обучение с Lightning
# ======================

def main():
    # Пути к данным (укажите свои корректные пути)
    train_csv = "./train.csv"
    dev_csv = "./dev.csv"
    audio_dir = "./"

    # Гиперпараметры
    batch_size = 56
    learning_rate = 0.001
    num_epochs = 50
    early_stop_patience = 5  # количество эпох без улучшения для ранней остановки

    # Создаём датасеты и DataLoader’ы
    train_dataset = ASRDataset(train_csv, audio_dir, target_sample_rate=16000, augment=True)
    dev_dataset = ASRDataset(dev_csv, audio_dir, target_sample_rate=16000, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # Инициализируем Lightning-модель
    asr_module = ASRLightningModule(n_mels=80, num_classes=num_classes, rnn_hidden=256,
                                    num_rnn_layers=1, dropout_rate=0.1, learning_rate=learning_rate,
                                    weight_decay=1e-4)

    # Callback для ранней остановки
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=early_stop_patience, mode="min", verbose=True)
    # Callback для сохранения лучшей модели
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, verbose=True)

    mlflow_logger = MLFlowLogger(experiment_name="ASR_Experiment", tracking_uri="file:./mlruns")
    
    # Инициализируем Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=mlflow_logger
    )

    # Запуск обучения
    trainer.fit(asr_module, train_loader, dev_loader)

if __name__ == "__main__":
    main()
