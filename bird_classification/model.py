import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

from bird_classification.models.efficient_net import EfficientNetModel
from bird_classification.models.resnet_baseline import ResNetBaseline


class BirdClassifier(pl.LightningModule):
    """PyTorch Lightning модуль для классификации птиц.

    Поддерживает модели: ResNet18/50, EfficientNet-B0.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Создаем модель на основе конфигурации
        self.model = self._create_model(cfg.model)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Метрики
        num_classes = cfg.model.num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Сохраняем конфиг оптимизатора
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.get('scheduler', None)

    def _create_model(self, model_cfg):
        """Создает модель на основе конфигурации."""
        model_name = model_cfg.model_name

        if model_name.startswith("resnet"):
            return ResNetBaseline(
                model_name=model_name,
                num_classes=model_cfg.num_classes,
                freeze_backbone=model_cfg.get('freeze_backbone', True),
                dropout=model_cfg.get('dropout', 0.0)
            )
        elif model_name.startswith("efficientnet"):
            return EfficientNetModel(
                model_name=model_name,
                num_classes=model_cfg.num_classes
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch, batch_idx):
        """Обучение на одном батче."""
        x, labels = batch
        logits = self(x)
        loss = self.loss_fn(logits, labels)

        # Вычисляем метрики
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)

        # Логируем метрики
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Валидация на одном батче."""
        x, labels = batch
        logits = self(x)
        loss = self.loss_fn(logits, labels)

        # Вычисляем метрики
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)

        # Логируем метрики
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Тестирование на одном батче."""
        x, labels = batch
        logits = self(x)
        loss = self.loss_fn(logits, labels)

        # Вычисляем метрики
        self.test_acc(logits, labels)
        self.test_f1(logits, labels)

        # Логируем метрики
        self.log("test/loss", loss, on_epoch=True, logger=True)
        self.log("test/acc", self.test_acc, on_epoch=True, logger=True)
        self.log("test/f1", self.test_f1, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        """Настраивает оптимизатор и scheduler."""
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())

        if self.scheduler_config is not None:
            scheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
