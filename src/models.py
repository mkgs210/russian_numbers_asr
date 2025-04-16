import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchaudio.models import Conformer
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io


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
        idx_to_vocab=None,
    ):
        super(ASRLightningConformer, self).__init__()
        self.save_hyperparameters()
        self.idx_to_vocab = idx_to_vocab or {}
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

        self.validation_samples = []

    def forward(self, x, lengths):
        # Передаем lengths в Conformer и распаковываем выход
        x, lengths_out = self.conformer(x, lengths)
        x = self.fc(x)
        return x.transpose(0, 1)

    def decode_output(self, output):
        pred_indices = torch.argmax(output, dim=1).tolist()
        decoded = []
        previous = None
        for idx in pred_indices:
            if idx != previous:
                if idx != 0:  # Skip blank
                    decoded.append(self.idx_to_vocab.get(idx, ""))
                previous = idx
        return "".join(decoded)

    def decode_target(self, target_indices):
        return "".join([self.idx_to_vocab.get(idx, "") for idx in target_indices])

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
        specs, spec_lengths, targets_concat, target_lengths = batch
        output = self.forward(specs, spec_lengths)
        effective_spec_lengths = spec_lengths // self.subsampling_factor
        loss = self.criterion(
            output, targets_concat, effective_spec_lengths, target_lengths
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:  # Log samples from the first validation batch
            batch_size = specs.size(0)
            n_samples = min(batch_size, 5)  # Log up to 5 samples
            for i in range(n_samples):
                # Process each sample
                spec_len = spec_lengths[i].item()
                mel_spec = specs[i, :spec_len, :].cpu().numpy().T  # (n_mels, time)

                effective_len = effective_spec_lengths[i].item()
                sample_output = output[:effective_len, i, :].cpu()

                predicted_text = self.decode_output(sample_output)

                # Extract target indices
                target_start = sum(target_lengths[:i])
                target_end = target_start + target_lengths[i]
                target_indices = targets_concat[target_start:target_end].cpu().tolist()
                ground_truth_text = self.decode_target(target_indices)

                self.validation_samples.append(
                    {
                        "mel_spec": mel_spec,
                        "predicted_text": predicted_text,
                        "ground_truth_text": ground_truth_text,
                    }
                )
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_samples or not self.logger:
            return

        def _plt_image_to_pillow_image(plt_image) -> Image:
            img_buf = io.BytesIO()
            plt_image.savefig(img_buf, format="png")
            pillow_image = Image.open(img_buf)
            return pillow_image

        for i, sample in enumerate(self.validation_samples[:5]):
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(
                sample["mel_spec"], aspect="auto", origin="lower", cmap="viridis"
            )
            plt.colorbar(im, ax=ax)
            ax.set_title(
                f"Pred: {sample['predicted_text']}\nTrue: {sample['ground_truth_text']}"
            )
            plt.tight_layout()

            image = _plt_image_to_pillow_image(fig)

            epoch = self.trainer.current_epoch

            self.logger.experiment.log_image(
                run_id=self.logger.run_id,
                image=image,
                artifact_file=f"val_samples/epoch_{epoch}/epoch_{epoch}_sample_{i}.png",
            )

        self.validation_samples = []

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
