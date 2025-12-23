import json

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from model import BirdClassifier
from models.conv_net import ConvNet
from models.efficient_net import EfficientNetModel


class BirdClassifierInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(cfg)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
            ]
        )

        with open(cfg.class_to_name_path, "r") as f:
            self.class_to_name = json.load(f)
            self.class_to_name = {int(k): v for k, v in self.class_to_name.items()}

    def _load_model(self, cfg: DictConfig):
        if self.cfg.model.model_name == "conv_net":
            base_model = ConvNet(num_classes=self.cfg.model.num_classes)
        elif self.cfg.model.model_name.startswith("efficientnet"):
            base_model = EfficientNetModel(
                model_name=self.cfg.model.model_name,
                num_classes=self.cfg.model.num_classes,
            )
        else:
            raise ValueError(f"Unsupported model: {self.cfg.model.model_name}")

        model = BirdClassifier(self.cfg)
        model.model = base_model

        checkpoint = torch.load(cfg.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_path: str) -> dict:
        input_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)

        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

        return {
            "species": self.class_to_name[pred_idx],
            "confidence": confidence,
            "class_probabilities": probs.cpu().numpy()[0],
        }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def classify_bird(cfg: DictConfig):
    classifier = BirdClassifierInference(cfg)
    result = classifier.predict(cfg.image_path)

    print(f"Predicted Bird Species: {result['species']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All probabilities: {np.round(result['class_probabilities'], 4)}")


if __name__ == "__main__":
    classify_bird()
