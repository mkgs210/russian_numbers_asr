import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import io
import tempfile
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchaudio.models import Conformer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.functional import char_error_rate
import matplotlib.pyplot as plt
from PIL import Image
import mlflow

from src.my_beam_search import tokens_to_number_string  # только утилита


class ASRLightningConformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        subsampling_factor: int,
        scheduler_t_0: int,
        scheduler_t_mult: int,
        scheduler_min_lr: float,
        cer_monitor: str,
        optimizer_type: str,
        augmentation_start_epoch: int,
        idx_to_vocab: dict[int, str],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.idx_to_vocab = idx_to_vocab
        self.augment_from = augmentation_start_epoch
        self.cer_monitor = cer_monitor
        #self.val_sample_counter = 0

        # Для логирования валид. спектрограмм
        self.validation_samples = []

        # Модель
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )
        self.fc = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x, lengths):
        x, out_lens = self.conformer(x, lengths)
        x = self.fc(x)
        return x.transpose(0, 1), out_lens

    def on_train_epoch_start(self):
        # Включаем аугментацию только после нужной эпохи
        dl = self.trainer.train_dataloader
        ds = getattr(dl, "dataset", None) or dl[0].dataset
        ds.augment = (self.current_epoch >= self.augment_from)

    def training_step(self, batch, batch_idx):
        specs, lens, targets, t_lens = batch
        logits, out_lens = self(specs, lens)

        eff = out_lens // self.hparams.subsampling_factor
        eff = torch.clamp(eff, min=1)
        loss = self.criterion(logits, targets, eff, t_lens)
        self.log("train_loss", loss, prog_bar=True, batch_size=specs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        specs, lens, targets, t_lens = batch
        logits, out_lens = self(specs, lens)

        eff = out_lens // self.hparams.subsampling_factor
        eff = torch.clamp(eff, min=1)
        loss = self.criterion(logits, targets, eff, t_lens)
        self.log("val_loss", loss, prog_bar=True, batch_size=specs.size(0))

        # Greedy decode для CER (каждый 10-й пример)
        bs = specs.size(0)
        offset = 0
        for i in range(bs):
            L = t_lens[i].item()
            ref_idx = targets[offset : offset + L].cpu().tolist()
            offset += L

            # self.val_sample_counter += 1
            # if self.val_sample_counter % 10 != 0:
            #     continue

            probs = torch.softmax(logits[:, i, :], dim=-1)
            pred = torch.argmax(probs, dim=-1).tolist()
            toks, prev = [], 0
            for idx in pred:
                if idx != prev and idx != 0:
                    toks.append(self.idx_to_vocab[idx])
                prev = idx

            # Если первый токен '|', вставляем '<1>'
            if toks and toks[0] == "|":
                toks = ["<1>", "|"] + toks[1:]

            pred_str_tok = "".join(toks)
            pred_str_num = tokens_to_number_string(toks)

            ref_toks    = [self.idx_to_vocab[x] for x in ref_idx]
            ref_str_tok = "".join(ref_toks)
            ref_str_num = tokens_to_number_string(ref_toks)

            cer_t = char_error_rate([pred_str_tok], [ref_str_tok]).item()
            cer_n = char_error_rate([pred_str_num], [ref_str_num]).item()
            self.log("CER_tokens_step", cer_t, batch_size=1)
            self.log("CER_numeric_step", cer_n, batch_size=1)

            # Собираем примеры для логирования спектрограмм (первые 3 примера)
            if batch_idx == 0 and len(self.validation_samples) < 3:
                # Спектрограмма: [n_mels, time]
                mel = specs[i, :lens[i].item(), :].cpu().numpy().T
                self.validation_samples.append({
                    "mel_spec": mel,
                    "pred_tok": pred_str_tok,
                    "ref_tok":  ref_str_tok,
                    "pred_num": pred_str_num,
                    "ref_num":  ref_str_num,
                })

        return loss

    def on_validation_epoch_end(self):
        # если нет ни одного примера — сразу выходим
        if not self.validation_samples:
            return

        for idx, s in enumerate(self.validation_samples):
            # 1) Рисуем спектрограмму
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(s["mel_spec"], aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(
                f"Pred TOK: {s['pred_tok']}\n"
                f"Ref  TOK: {s['ref_tok']}\n"
                f"Pred NUM: {s['pred_num']}\n"
                f"Ref  NUM: {s['ref_num']}"
            )
            plt.tight_layout()

            # 2) Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, format="png")
                tmp_path = tmp.name
            plt.close(fig)

            # 3) Логируем через MlflowClient, доступный в self.logger.experiment
            #    Первым аргументом — run_id, вторым — локальный путь файла, третьим — папка внутри артефактов.
            self.logger.experiment.log_artifact(
                self.logger.run_id,
                local_path=tmp_path,
                artifact_path=f"val_spectrograms/epoch_{self.current_epoch}"
            )

            # 4) Удаляем временный файл
            os.remove(tmp_path)

        # 5) Очищаем накопленные примеры, чтобы не залогировать дважды
        self.validation_samples.clear()

    def configure_optimizers(self):
        lr  = self.hparams.learning_rate
        wd  = self.hparams.weight_decay
        opt = self.hparams.optimizer_type

        if opt == "AdamW":
            from torch.optim import AdamW
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif opt == "NovoGrad":
            from torch_optimizer import NovoGrad
            optimizer = NovoGrad(self.parameters(), lr=lr, weight_decay=wd)
        elif opt == "Lion":
            from lion_pytorch import Lion
            optimizer = Lion(self.parameters(), lr=lr, weight_decay=wd)
        elif opt == "Ranger":
            from torch_optimizer import Ranger
            optimizer = Ranger(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.scheduler_t_0,
            T_mult=self.hparams.scheduler_t_mult,
            eta_min=self.hparams.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.cer_monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }