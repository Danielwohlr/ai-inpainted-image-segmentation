import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_unet_model():
    """
    Returns a UNet with 7 input channels:
      - The first 3 channels come from a ResNet34 pretrained on ImageNet
      - The extra 4 channels are randomly initialized.
    """
    # 1) Create a temporary 3-channel model to pull pretrained weights
    temp_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # loads pretrained on ImageNet
        in_channels=3,
        classes=1,
        activation=None,
    )
    # Save the pretrained conv1 weights
    pretrained_conv1 = temp_model.encoder.conv1.weight.clone()  # shape: [64, 3, 7, 7]

    # 2) Create our actual 7-channel model with no pretrained weights
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # do not auto-load, we will do it ourselves
        in_channels=7,  # total of 7 channels
        classes=1,
        activation=None,
    )

    # 3) Manually copy the first 3 channels from the pretrained conv1,
    #    and randomly init the remaining channels.
    with torch.no_grad():
        # The new conv1 has shape [64, 6, 7, 7].
        model.encoder.conv1.weight[:, :3, :, :] = pretrained_conv1
        # Initialize the extra 3 channels with small random values
        nn.init.normal_(model.encoder.conv1.weight[:, 3:, :, :], mean=0.0, std=0.01)

    return model
