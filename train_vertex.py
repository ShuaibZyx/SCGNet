import torch
import math
import hydra
from pathlib import Path
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from src.utils.expriment_utils import generate_experiment_name

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")


def main() -> None:
    with hydra.initialize_config_module(version_base=None, config_module="src.config"):
        cfg = hydra.compose(config_name="vertex_model.yaml")
        vertex_model_config = instantiate(cfg.VertexModelConfig)

    vertex_data_module = vertex_model_config.vertex_data_module
    vertex_model = vertex_model_config.vertex_model

    experiment = vertex_model_config.experiment
    epochs = vertex_model_config.epochs
    batch_size = vertex_model_config.batch_size

    train_dataset_length = len(vertex_data_module.train_dataset)
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

    trainer = Trainer(
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
        ],
        logger=[wandb_logger],
    )
    trainer.fit(model=vertex_model, datamodule=vertex_data_module, ckpt_path="/root/autodl-fs/P2B-PCFVWA-TorchSparse/runs/02212222_Vertex-Full-TorchSparse-Tallinn_proper-cap/checkpoints/last.ckpt")


if __name__ == "__main__":
    main()
