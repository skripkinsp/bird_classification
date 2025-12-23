# bird-classification

This repository contains the code for the Bird Species Classification project.

The goal of this project is to classify bird species based on their images.

Classification is on 400 bird species.

Metrics:

- Accuracy
- F1 score (macro)

Baseline model is simple convolutional neural network.

Dataset:

- [Birds400](https://www.kaggle.com/datasets/antoniozarauzmoreno/birds400) dataset with 400 bird species
- Data is automatically downloaded from Kaggle
- **Automatic redistribution**: 15 images per species go to validation, 15 to test, rest to training
- Ensures balanced and representative splits for each bird species

**Results**:
- 0.46 F1 Score 0.63 Accuracy on Resnet18
- 0.54 F1 Score 0.7 Accuracy on EfficientNet


## Setup

1. Install
   [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
2. Install the project dependencies:

```bash
poetry install
```

3. **Set up Kaggle API** for data downloading (see [KAGGLE_SETUP.md](KAGGLE_SETUP.md) for detailed instructions)

3. Set up DVC for data management:

```bash
# Initialize DVC (if not already done)
dvc init

# Set up local data storage
dvc remote add -d birds_data ./data/birds400

# Add your birds400 dataset to DVC
dvc add data/birds400
```

4. Optional: Set up MLflow for experiment tracking:

```bash
# Start MLflow server in background
mlflow server --host 127.0.0.1 --port 8080 &
```

## Train

### Baseline Model (ResNet18 with frozen backbone)
```bash
poetry run python bird_classification/train.py --config-name=baseline
```

### Main Model (EfficientNet-B0 with full fine-tuning)
```bash
poetry run python bird_classification/train.py --config-name=main_model
```

### ResNet50 Model
```bash
poetry run python bird_classification/train.py --config-name=resnet50
```

### Custom Configuration
```bash
poetry run python bird_classification/train.py \
    model=resnet_baseline \
    model.model_name=resnet18 \
    model.freeze_backbone=true
```

The data will be automatically downloaded from Kaggle on first run.

See [MODELS.md](MODELS.md) for detailed model descriptions and configurations.

## Pruduction preparation

1. You need to put .onnx model in the `triton/export/onnx/model.onnx` file (if
   you fully trained model then it's already made onnx).
2. To prepare the production tensorrt model for triton go to `triton` folder and
   use [commands](triton/export_to_trt.md).

## Infer

To run simple inference on a single bird image, use the following command:

```bash
poetry run python bird_classification/infer.py \
    image_path=path/to/bird/image.jpg \
    model_path=checkpoints/best_model_weights.pth \
    class_to_name_path=checkpoints/class_to_name.json
```

The model will output the predicted bird species and confidence score.

If you want to infer via triton see [example notebook](triton/test.ipynb).
