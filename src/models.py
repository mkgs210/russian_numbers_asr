import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchaudio.models import Conformer


class ASRLightningConformer(pl.LightningModule):
    def __init__(
        self,
        input_dim=80,
        num_classes=31,
        num_heads=4,
        ffn_dim=576,
        num_layers=3,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=1e-4,
        subsampling_factor=4,
    ):
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
            convolution_first=False,
        )
        self.fc = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.learning_rate = learning_rate
        self.subsampling_factor = subsampling_factor

    def forward(self, x, lengths):
        # Передаем lengths в Conformer и распаковываем выход
        x, lengths_out = self.conformer(x, lengths)
        x = self.fc(x)
        return x.transpose(0, 1)

    def training_step(self, batch, batch_idx):
        specs, spec_lengths, targets, target_lengths = batch
        output = self.forward(specs, spec_lengths)
        effective_spec_lengths = spec_lengths // self.subsampling_factor
        loss = self.criterion(output, targets, effective_spec_lengths, target_lengths)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, spec_lengths, targets, target_lengths = batch
        output = self.forward(specs, spec_lengths)
        effective_spec_lengths = spec_lengths // self.subsampling_factor
        loss = self.criterion(output, targets, effective_spec_lengths, target_lengths)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
