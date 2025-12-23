import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class BirdDataset(Dataset):
    """Датасет для изображений птиц.

    Args:
        root: Путь к директории с данными
        transform: Трансформации для изображений
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.images = []
        self.labels = []
        self.class_to_idx = {}

        # Получаем список всех видов птиц (папки)
        species_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])

        for class_idx, species_dir in enumerate(species_dirs):
            self.class_to_idx[species_dir.name] = class_idx

            # Получаем все изображения в папке вида
            image_files = list(species_dir.glob("*.jpg")) + \
                         list(species_dir.glob("*.jpeg")) + \
                         list(species_dir.glob("*.png"))

            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(class_idx)

        print(f"Загружено {len(self.images)} изображений из {len(species_dirs)} классов")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        label = self.labels[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            # Возвращаем случайное изображение вместо сломанного
            return self.__getitem__(np.random.randint(0, len(self.images)))

        if self.transform:
            image = self.transform(image)

        return image, label


class BirdDataModule(pl.LightningDataModule):
    """LightningDataModule для датасета птиц.

    Args:
        data_dir: Путь к директории с данными (train/val/test)
        batch_size: Размер батча
        num_workers: Количество worker'ов для DataLoader
        image_size: Размер изображений
        transform_mean: Среднее для нормализации
        transform_std: Стандартное отклонение для нормализации
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        transform_mean: list = None,
        transform_std: list = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Значения по умолчанию для нормализации (ImageNet)
        self.transform_mean = transform_mean or [0.485, 0.456, 0.406]
        self.transform_std = transform_std or [0.229, 0.224, 0.225]

        # Создаем трансформации
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()

    def _create_train_transform(self):
        """Создает трансформации для тренировочных данных."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
        ])

    def _create_val_transform(self):
        """Создает трансформации для валидационных и тестовых данных."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
        ])

    def get_class_to_name(self) -> dict:
        """Возвращает маппинг индекс класса -> название вида птицы."""
        class_to_name = {}

        train_dir = self.data_dir / "train"
        if train_dir.exists():
            species_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
            for idx, species_dir in enumerate(species_dirs):
                class_to_name[idx] = species_dir.name

        return class_to_name

    def prepare_data(self):
        """Подготавливает данные (вызывается один раз на всех GPU)."""
        # Здесь можно добавить скачивание данных, если нужно
        pass

    def setup(self, stage: str = None):
        """Настраивает датасеты для текущего stage."""
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / "val"

            if train_dir.exists():
                self.train_dataset = BirdDataset(train_dir, transform=self.train_transform)
            if val_dir.exists():
                self.val_dataset = BirdDataset(val_dir, transform=self.val_transform)

        if stage == "test" or stage is None:
            test_dir = self.data_dir / "test"
            if test_dir.exists():
                self.test_dataset = BirdDataset(test_dir, transform=self.val_transform)

    def train_dataloader(self) -> DataLoader:
        """DataLoader для тренировочных данных."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader для валидационных данных."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader для тестовых данных."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
