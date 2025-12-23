import json
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger

from dataset import BirdDataModule
from model import BirdClassifier
from utils import download_data, export_to_onnx, get_git_commit_id


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Основная функция обучения модели классификации птиц."""
    # Устанавливаем seed для воспроизводимости
    pl.seed_everything(42)

    # Подготавливаем данные
    download_data(cfg.data.data_dir, cfg.data.split_dir)

    # Создаем логгеры
    loggers = create_loggers(cfg)

    # Создаем коллбэки
    callbacks = create_callbacks(cfg)

    # Создаем модель и датамодуль
    model = BirdClassifier(cfg)
    data_module = BirdDataModule(
        data_dir=cfg.data.split_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        transform_mean=cfg.data.mean,
        transform_std=cfg.data.std,
    )

    # Создаем тренера
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Обучаем модель
    print("Начинаем обучение модели...")
    trainer.fit(model, datamodule=data_module)

    # Тестируем лучшую модель
    print("Тестируем лучшую модель...")
    trainer.test(model, datamodule=data_module)

    # Сохраняем модель и метаданные
    save_model_and_metadata(trainer, model, data_module, cfg)


def create_loggers(cfg: DictConfig):
    """Создает логгеры для экспериментов."""
    loggers = []

    # MLflow логгер для отслеживания экспериментов
    mlflow_logger = MLFlowLogger(
        experiment_name="bird-classification",
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"git_commit": get_git_commit_id()},
    )

    # Логируем гиперпараметры
    mlflow_logger.log_hyperparams({
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "image_size": cfg.data.image_size,
        "model": cfg.model.model_name,
        "optimizer": cfg.optimizer._target_,
        "learning_rate": cfg.optimizer.lr,
        "max_epochs": cfg.trainer.max_epochs,
    })

    loggers.append(mlflow_logger)

    # TensorBoard логгер для визуализации
    tb_logger = TensorBoardLogger(save_dir="lightning_logs/", name="tensorboard")
    loggers.append(tb_logger)

    # CSV логгер для локального мониторинга
    csv_logger = CSVLogger(save_dir="lightning_logs/", name="csv")
    loggers.append(csv_logger)

    return loggers


def create_callbacks(cfg: DictConfig):
    """Создает коллбэки для обучения."""
    callbacks = []

    # Model Checkpoint - сохраняет лучшую модель
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename="best-{epoch:02d}-{val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Early Stopping для предотвращения переобучения
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Learning Rate Monitor для отслеживания изменения learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Rich Progress Bar для красивого отображения прогресса
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    return callbacks


def save_model_and_metadata(trainer, model, data_module, cfg):
    """Сохраняет модель и метаданные."""
    print("Сохраняем модель и метаданные...")

    # Создаем директорию для чекпоинтов
    os.makedirs(cfg.checkpoint.dirpath, exist_ok=True)

    # Экспортируем в ONNX для продакшена
    export_to_onnx(model=model, output_path=cfg.export.onnx_path)

    # Сохраняем веса модели в формате PyTorch
    weights_path = os.path.join(cfg.checkpoint.dirpath, "best_model_weights.pth")
    torch.save(model.state_dict(), weights_path)

    # Сохраняем маппинг классов для инференса
    class_mapping_path = os.path.join(cfg.checkpoint.dirpath, "class_to_name.json")
    with open(class_mapping_path, "w") as f:
        json.dump(data_module.get_class_to_name(), f, indent=2)

    print("Модель успешно сохранена!")
    print(f"ONNX модель: {cfg.export.onnx_path}")
    print(f"Маппинг классов: {class_mapping_path}")
    print(f"Лучший чекпоинт: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
