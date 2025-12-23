import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBaseline(nn.Module):
    """
    ResNet baseline model for bird classification.

    Supports both frozen backbone (baseline) and full fine-tuning modes.
    """

    def __init__(self, model_name="resnet18", num_classes=400, freeze_backbone=True, dropout=0.0):
        super().__init__()

        # Load pretrained ResNet model
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = 512
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Freeze backbone if specified (for baseline)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Custom classifier head
        classifier_layers = []
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        # Classify
        return self.classifier(features)

    def get_trainable_params(self):
        """Get only trainable parameters for optimizer"""
        return filter(lambda p: p.requires_grad, self.parameters())
