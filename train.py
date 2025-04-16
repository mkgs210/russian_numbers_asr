import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from src.datasets import RussianWordsASRDataset, collate_fn
from src.models import ASRLightningConformer

torch.set_float32_matmul_precision("medium")


def main():
    train_csv = "./data/train/train.csv"  # Замените путь на реальный
    dev_csv = "./data/dev/dev.csv"  # Замените путь на реальный
    train_audio_dir = "./data/train"  # Замените путь на реальный
    dev_audio_dir = "./data/dev"

    batch_size = 4
    num_epochs = 100
    early_stop_patience = 25

    train_dataset = RussianWordsASRDataset(
        train_csv, train_audio_dir, target_sample_rate=16000, augment=True
    )
    dev_dataset = RussianWordsASRDataset(
        dev_csv, dev_audio_dir, target_sample_rate=16000, augment=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    num_classes = train_dataset.num_classes

    mlflow_logger = MLFlowLogger(
        experiment_name="ASR_Conformer_Experiment", tracking_uri="file:./mlruns"
    )

    asr_module = ASRLightningConformer(
        input_dim=80,  # число мел-банов
        num_classes=num_classes,  # число классов (включая blank)
        num_heads=4,  # увеличено с 4 до 8
        ffn_dim=512,  # увеличено с 576 до 2048
        num_layers=4,  # увеличено с 3 до 8 слоёв
        depthwise_conv_kernel_size=31,  # оставляем как есть
        dropout=0.1,  # стандартное значение dropout
        learning_rate=0.0005,
        weight_decay=1e-4,
        subsampling_factor=4,  # предполагаемая субдискретизация,
        idx_to_vocab=train_dataset.idx_to_vocab,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=early_stop_patience, mode="min", verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=mlflow_logger,
    )

    trainer.fit(asr_module, train_loader, dev_loader)


if __name__ == "__main__":
    main()
