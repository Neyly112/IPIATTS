"""
Script training Matcha-TTS với Prosody Analysis (PhoBERT)
Dành cho dữ liệu tiếng Việt
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from matcha.models.matcha_tts import MatchaTTS
from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.text.symbols import symbols


# ============================================================================
# CẤU HÌNH - CHỈNH SỬA PHẦN NÀY CHO PHÙ HỢP VỚI DỮ LIỆU CỦA BẠN
# ============================================================================

CONFIG = {
    # Đường dẫn dữ liệu
    "train_filelist": "data/99-audio-text-file-list/audio_text_train.txt.cleaned",
    "val_filelist": "data/99-audio-text-file-list/audio_text_val.txt.cleaned",

    # Thư mục lưu checkpoint và logs
    "output_dir": "outputs/matcha_prosody",

    # Thiết lập model
    "n_spks": 1,     # 1 = single speaker, >1 = multi-speaker
    "spk_emb_dim": 64,
    "n_feats": 80,   # Mel-spectrogram features

    # Prosody settings
    "llm_model_name": "vinai/phobert-base",  # PhoBERT cho tiếng Việt
    "prosody_dim": 256,

    # Training hyperparameters
    "batch_size": 1,  # Giảm xuống 1 cho GPU 4GB
    "learning_rate": 1e-4,
    "max_epochs": 2,
    "num_workers": 0,  # Số worker cho DataLoader
    "accumulate_grad_batches": 4,  # Gradient accumulation = effective batch_size 4

    # GPU/CPU
    "accelerator": "gpu",  # "gpu" hoặc "cpu"
    "devices": 1,

    # Checkpoint để resume (nếu có)
    # Đường dẫn đến .ckpt file nếu muốn tiếp tục training
    "resume_from_checkpoint": None,
    "audio_root": "data/subs",
}
# Nếu add_blank=True trong TextMelDataModule
CONFIG["n_vocab"] = len(symbols) + 1
print(">>> n_vocab =", CONFIG["n_vocab"], "len(symbols) =", len(symbols))

CONFIG["optimizer"] = None
CONFIG["optimizer_kwargs"] = {"lr": CONFIG["learning_rate"]}
CONFIG["scheduler"] = None


# ============================================================================
# ENCODER CONFIG
# ============================================================================

ENCODER_CONFIG = {
    "encoder_type": "RoPE Encoder",
    "encoder_params": {
        "n_feats": CONFIG["n_feats"],
        "n_channels": 384,
        "filter_channels": 1024,
        "filter_channels_dp": 256,
        "n_heads": 8,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "spk_emb_dim": CONFIG["spk_emb_dim"],
        "n_spks": CONFIG["n_spks"],
        "prenet": True,
    },
    "duration_predictor_params": {
        "filter_channels_dp": 256,
        "kernel_size": 3,
        "p_dropout": 0.1,
    },
}


# ============================================================================
# DECODER CONFIG (CFM - Conditional Flow Matching)
# ============================================================================

DECODER_CONFIG = {
    "channels": [256, 256],
    "dropout": 0.2,
    "attention_head_dim": 64,
    "n_blocks": 4,
    "num_mid_blocks": 4,
    "num_heads": 8,
    "act_fn": "gelu",
}

CFM_CONFIG = {
    "sigma_min": 1e-4,
    "solver": "euler",
    "t_scheduler": "cosine",
}


# ============================================================================
# DATA STATISTICS (sẽ được cập nhật từ dữ liệu)
# ============================================================================

DATA_STATISTICS = {
    "mel_mean": 0.0,
    "mel_std": 1.0,
}


# ============================================================================
# HÀM TẠO MODEL
# ============================================================================

def create_model(config):
    """Tạo Matcha-TTS model với Prosody"""

    # Tạo namespace objects cho encoder, decoder, cfm
    from argparse import Namespace

    encoder = Namespace(
        encoder_type=ENCODER_CONFIG["encoder_type"],
        encoder_params=Namespace(**ENCODER_CONFIG["encoder_params"]),
        duration_predictor_params=Namespace(
            **ENCODER_CONFIG["duration_predictor_params"]),
    )

    decoder = DECODER_CONFIG.copy()
    cfm = Namespace(**CFM_CONFIG)
    data_statistics = DATA_STATISTICS

    # Khởi tạo model
    model = MatchaTTS(
        n_vocab=config["n_vocab"],
        n_spks=config["n_spks"],
        spk_emb_dim=config["spk_emb_dim"],
        n_feats=config["n_feats"],
        encoder=encoder,
        decoder=decoder,
        cfm=cfm,
        data_statistics=data_statistics,
        out_size=None,
        llm_model_name=config["llm_model_name"],
        prosody_dim=config["prosody_dim"],
        optimizer=config.get("optimizer"),
        scheduler=config.get("scheduler"),
    )

    return model


# ============================================================================
# HÀM TẠO DATAMODULE
# ============================================================================

def create_datamodule(config):
    """
    Tạo DataModule cho training

    LƯU Ý: Bạn cần implement TextMelDataModule phù hợp với format dữ liệu
    của bạn. File filelist cần có format:

    audio_path|text|phonemes

    Ví dụ:
    data/vad1/audio_001.wav|xin chào|s i n ch a o
    """

    datamodule = TextMelDataModule(
        name="matcha_prosody",
        train_filelist_path=config["train_filelist"],
        valid_filelist_path=config["val_filelist"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        cleaners=["basic_cleaners_phothong"],  # Text cleaner cho tiếng Việt
        add_blank=True,
        n_spks=config["n_spks"],
        n_fft=1024,
        n_feats=config["n_feats"],
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
        data_statistics=None,  # Sẽ được tính tự động
        seed=1234,
        load_durations=False,
        audio_root=config.get("audio_root"),
    )

    return datamodule


# ============================================================================
# HÀM TRAINING
# ============================================================================

def train(config):
    """Main training function"""

    print("=" * 80)
    print("🍵 MATCHA-TTS TRAINING VỚI PROSODY ANALYSIS")
    print("=" * 80)
    print(f"PhoBERT Model: {config['llm_model_name']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max epochs: {config['max_epochs']}")
    print(f"Output dir: {config['output_dir']}")
    print("=" * 80)

    # 1. Tạo DataModule
    print("\n[1/4] Đang load dữ liệu...")
    datamodule = create_datamodule(config)

    # 2. Tạo Model
    print("[2/4] Đang khởi tạo model...")
    if config["resume_from_checkpoint"]:
        print(
            f"      Resume từ checkpoint: {config['resume_from_checkpoint']}")
        model = MatchaTTS.load_from_checkpoint(
            config["resume_from_checkpoint"],
            llm_model_name=config["llm_model_name"],
            prosody_dim=config["prosody_dim"],
        )
    else:
        model = create_model(config)

    # 3. Setup Callbacks
    print("[3/4] Đang setup callbacks...")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config['output_dir']}/checkpoints",
        filename="matcha-prosody-{epoch:03d}-{loss/val_epoch:.3f}",
        monitor="loss/val_epoch",
        mode="min",
        save_top_k=3,
        save_last=True,
        save_weights_only=True,
    )

    early_stopping = EarlyStopping(
        monitor="loss/val_epoch",
        patience=50,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. Setup Logger
    logger = TensorBoardLogger(
        save_dir=config["output_dir"],
        name="logs",
    )

    # 5. Setup Trainer
    print("[4/4] Đang setup trainer...")
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=1.0,
        precision="16-mixed" if config["accelerator"] == "gpu" else "32",
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),  # Gradient accumulation
    )

    # 6. Start Training
    print("\n" + "=" * 80)
    print("BẮT ĐẦU TRAINING!")
    print("=" * 80)
    print(f"Mở TensorBoard để xem quá trình training:")
    print(f"    tensorboard --logdir {config['output_dir']}/logs")
    print("=" * 80 + "\n")

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=config["resume_from_checkpoint"],
    )

    print("\n" + "=" * 80)
    print("✅ TRAINING HOÀN TẤT!")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Kiểm tra CUDA
    if CONFIG["accelerator"] == "gpu":
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(
                f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA not available, switching to CPU")
            CONFIG["accelerator"] = "cpu"

    # Bắt đầu training
    train(CONFIG)
