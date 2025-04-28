import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.datasets import CustomNumbersASRDataset, collate_fn
from src.models import ASRLightningConformer, tokens_to_number_string
from src.my_beam_search import ctc_beam_search_fsa
from torchmetrics.functional import char_error_rate

ckpt_path = None  # "path/to/your/checkpoint.ckpt"  # если нужно продолжить обучение с чекпоинта

def main():
    # 1) Конфиг
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2) Оптимизатор
    opt = cfg["optimizer"]
    opt_type = opt.get("type", "AdamW")
    lr       = float(opt["lr"])
    wd       = float(opt["weight_decay"])

    # 3) Scheduler
    sch = cfg["scheduler"]
    scheduler_t_0    = int(sch["t_0"])
    scheduler_t_mult = int(sch["t_mult"])
    scheduler_min_lr = float(sch["eta_min"])

    # 4) Обучение
    tr = cfg["training"]
    batch_size  = int(tr["batch_size"])
    max_epochs  = int(tr["max_epochs"])
    es_patience = int(tr["es_patience"])
    es_monitor  = tr["es_monitor"]
    log_every_n = int(tr["log_every_n_steps"])

    # 5) Аугментация
    aug_cfg          = cfg["augmentation"]
    noise_std        = float(aug_cfg["noise_std"])
    augment_start_ep = int(aug_cfg.get("start_epoch", 0))

    # 6) Датасеты
    train_ds = CustomNumbersASRDataset(
        cfg["data"]["train_csv"],
        cfg["data"]["train_audio_dir"],
        target_sample_rate=16000,
        noise_std=noise_std,
        augment=True,
    )
    dev_ds = CustomNumbersASRDataset(
        cfg["data"]["dev_csv"],
        cfg["data"]["dev_audio_dir"],
        target_sample_rate=16000,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds, batch_size=1, shuffle=False,
                              num_workers=8, collate_fn=collate_fn)

    # 7) Логгер и колбэки
    logger = MLFlowLogger("ASR_Conformer_Experiment", "file:./mlruns")
    early_stop = EarlyStopping(monitor=es_monitor, patience=es_patience,
                               mode="min", verbose=True)
    checkpoint = ModelCheckpoint(monitor=es_monitor, mode="min",
                                 save_top_k=1, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 8) Модель
    mcfg = cfg["model"]
    model = ASRLightningConformer(
        input_dim=int(mcfg["input_dim"]),
        num_classes=train_ds.num_classes,
        num_heads=int(mcfg["num_heads"]),
        ffn_dim=int(mcfg["ffn_dim"]),
        num_layers=int(mcfg["num_layers"]),
        depthwise_conv_kernel_size=int(mcfg["kernel_size"]),
        dropout=float(mcfg["dropout"]),
        learning_rate=lr,
        weight_decay=wd,
        subsampling_factor=int(mcfg["subsampling_factor"]),
        scheduler_t_0=scheduler_t_0,
        scheduler_t_mult=scheduler_t_mult,
        scheduler_min_lr=scheduler_min_lr,
        cer_monitor=es_monitor,
        optimizer_type=opt_type,
        augmentation_start_epoch=augment_start_ep,
        idx_to_vocab=train_ds.idx_to_vocab,
    )

    # 9) Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=log_every_n,
        logger=logger,
    )

    # 10) Обучение
    trainer.fit(model, train_loader, dev_loader, ckpt_path=ckpt_path)

    # 11) Финальный greedy+beam evaluation на DEV set
    print("\n=== Final greedy+beam evaluation on DEV set ===")
    best_ckpt = checkpoint.best_model_path
    model = ASRLightningConformer.load_from_checkpoint(best_ckpt)
    model.eval().to(trainer.strategy.root_device)

    total_chars      = 0
    total_err_tok_g  = 0.0
    total_err_num_g  = 0.0
    total_err_tok_b  = 0.0
    total_err_num_b  = 0.0

    with torch.no_grad():
        for specs, lens, targets, t_lens in tqdm(dev_loader, desc="Final eval"):
            specs = specs.to(model.device)
            lens  = lens.to(model.device)

            logits, out_lens = model(specs, lens)
            # Для simplicity: batch_size=1 на валидации
            logp = torch.log_softmax(logits[:,0,:], dim=-1).cpu()

            # — greedy decode —
            greedy_ids = logp.argmax(dim=-1).tolist()
            toks_g, prev = [], 0
            for idx in greedy_ids:
                if idx != prev and idx != 0:
                    toks_g.append(model.idx_to_vocab[idx])
                prev = idx
            if toks_g and toks_g[0] == "|":
                toks_g = ["<1>","|"] + toks_g[1:]
            p_tok_g = "".join(toks_g)
            p_num_g = tokens_to_number_string(toks_g)

            # — beam-search decode —
            beams = ctc_beam_search_fsa(
                logp,
                beam_width=cfg["decoding"]["beam_width"],
                blank=0
            )
            seq_b, _ = beams[0]
            toks_b = [model.idx_to_vocab[i] for i in seq_b]
            if toks_b and toks_b[0] == "|":
                toks_b = ["<1>","|"] + toks_b[1:]
            p_tok_b = "".join(toks_b)
            p_num_b = tokens_to_number_string(toks_b)

            # reference
            L = t_lens.item()
            ref_idx = targets[:L].cpu().tolist()
            r_toks   = [model.idx_to_vocab[i] for i in ref_idx]
            r_tok    = "".join(r_toks)
            r_num    = tokens_to_number_string(r_toks)

            # CER greedy
            ce_t_g = char_error_rate([p_tok_g], [r_tok]).item()
            ce_n_g = char_error_rate([p_num_g], [r_num]).item()
            total_err_tok_g += ce_t_g * len(r_tok)
            total_err_num_g += ce_n_g * len(r_tok)

            # CER beam
            ce_t_b = char_error_rate([p_tok_b], [r_tok]).item()
            ce_n_b = char_error_rate([p_num_b], [r_num]).item()
            total_err_tok_b += ce_t_b * len(r_tok)
            total_err_num_b += ce_n_b * len(r_tok)

            total_chars += len(r_tok)

    # Печатаем два набора метрик
    print(f"FINAL Greedy CER tokens:  {total_err_tok_g/total_chars:.4f}")
    print(f"FINAL Greedy CER numeric: {total_err_num_g/total_chars:.4f}")
    print(f" FINAL Beam CER tokens:  {total_err_tok_b/total_chars:.4f}")
    print(f" FINAL Beam CER numeric: {total_err_num_b/total_chars:.4f}")

if __name__ == "__main__":
    main()