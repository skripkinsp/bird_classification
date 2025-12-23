# Model Architectures

## Data Distribution

The Birds400 dataset is automatically redistributed during preparation:
- **15 images per species** → validation set
- **15 images per species** → test set
- **Remaining images** → training set

This ensures balanced and representative splits for all 400 bird species.

## Baseline Models

### ConvNet (Simple CNN)
- **Architecture**: 4 convolutional blocks with BatchNorm, MaxPool + 3 FC layers with Dropout
- **Parameters**: ~3.5M
- **Training**: Full fine-tuning
- **Use case**: Fast baseline, limited hardware

### ResNet18 Baseline
- **Architecture**: ResNet18 with frozen backbone + trainable classifier
- **Parameters**: ~11M (backbone frozen)
- **Training**: Only classifier trained
- **Use case**: Strong baseline with transfer learning

## Main Models

### ResNet50
- **Architecture**: Full ResNet50 + Dropout(0.5) classifier
- **Parameters**: ~25M
- **Training**: Full fine-tuning with CosineAnnealingLR
- **Augmentations**: RandomResizedCrop, HorizontalFlip, ColorJitter

### EfficientNet-B0
- **Architecture**: EfficientNet-B0 + Dropout(0.2) classifier
- **Parameters**: ~5M
- **Training**: Full fine-tuning with CosineAnnealingLR
- **Augmentations**: RandomResizedCrop, HorizontalFlip, ColorJitter

## Training Configurations

### Baseline Training
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 10
- **Batch size**: 64
- **Scheduler**: None

### Main Model Training
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.0001)
- **Epochs**: 50
- **Batch size**: 32
- **Scheduler**: CosineAnnealingLR (eta_min=1e-6)

## Usage Examples

```bash
# Train baseline
python bird_classification/train.py --config-name=baseline

# Train EfficientNet-B0
python bird_classification/train.py --config-name=main_model

# Train ResNet50
python bird_classification/train.py --config-name=resnet50

# Custom configuration
python bird_classification/train.py \
    model=resnet_baseline \
    model.model_name=resnet50 \
    model.freeze_backbone=false \
    trainer.max_epochs=30
```
