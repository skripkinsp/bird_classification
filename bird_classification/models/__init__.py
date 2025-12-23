# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "ConvNet":
        from .conv_net import ConvNet
        return ConvNet
    elif name == "EfficientNetModel":
        from .efficient_net import EfficientNetModel
        return EfficientNetModel
    elif name == "ResNetBaseline":
        from .resnet_baseline import ResNetBaseline
        return ResNetBaseline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
