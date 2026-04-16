import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
import math
import torch
import hydra
from pathlib import Path
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
# from lightning.pytorch.loggers import WandbLogger
from src.utils.expriment_utils import generate_experiment_name


seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")


def main() -> None:
    with hydra.initialize_config_module(version_base=None, config_module="src.config"):
        cfg = hydra.compose(config_name="face_model.yaml")
        face_model_config = instantiate(cfg.FaceModelConfig)

    face_data_module = face_model_config.face_data_module
    face_model = face_model_config.face_model

    experiment = face_model_config.experiment
    epochs = face_model_config.epochs
    batch_size = face_model_config.batch_size

    train_dataset_length = len(face_data_module.train_dataset)
    max_steps = train_dataset_length * epochs
    log_every_n_steps = math.floor(train_dataset_length // batch_size / 10)

    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    experiment_name = generate_experiment_name(experiment)

    checkpoint_callback = ModelCheckpoint(
        dirpath=(Path("runs") / experiment_name / "checkpoints"),
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
        verbose=False,
        auto_insert_metric_name=False,
        filename="epoch-{epoch:02d}-step-{step:06d}-train_loss-{train_loss:.4f}-val_loss-{val_loss:.4f}",
    )

    wandb_logger = WandbLogger(
        project="P2B-PCFVWA-Paper",
        name=experiment_name,
        id=experiment_name,
        log_model=True,
    )

    # tensorboard_logger = TensorBoardLogger(
    #     save_dir="runs",
    #     name=experiment_name,
    # )

    trainer = Trainer(
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
        ],
        logger=[wandb_logger],
    )

    trainer.fit(face_model, face_data_module)


if __name__ == "__main__":
    main()
