import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_transformer_model():
    """
    Returns a SegFormer with 7 input channels:
      - The first 3 channels come from MiT-B0 pretrained on ImageNet
      - The extra 4 channels are randomly initialized.

    If your features are (256,256,7), this model can be plugged into your existing
    training loop with minimal changes (just import and use this function).
    """

    # 1) Create a temporary 3-channel SegFormer to pull pretrained weights
    temp_model = smp.Segformer(
        encoder_name="mit_b0",  # transformer-based MiT-B0
        encoder_weights="imagenet",  # loads pretrained on ImageNet
        in_channels=3,
        classes=1,
        activation=None,
    )

    # The first patch embedding is patch_embed1 (for MiT-B0).
    # We'll copy out its weight for the first 3 channels.
    pretrained_weight = (
        temp_model.encoder.patch_embed1.proj.weight.clone()
    )  # shape [embed_dim, 3, kernel, kernel]

    # 2) Create our actual 7-channel SegFormer, no automatic pretrained weights
    model = smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights=None,  # we will manually load partial weights
        in_channels=7,  # total of 7 channels
        classes=1,
        activation=None,
    )

    # 3) Manually copy the first 3 channels from the pretrained patch_embed1,
    #    and randomly init the remaining channels.
    with torch.no_grad():
        # Suppose the shape is [embed_dim, 7, kernel, kernel].
        model.encoder.patch_embed1.proj.weight[:, :3, :, :] = pretrained_weight
        nn.init.normal_(
            model.encoder.patch_embed1.proj.weight[:, 3:, :, :], mean=0.0, std=0.01
        )

    return model


def get_transformer_inference_model():
    """
    Returns a SegFormer with 7 input channels, but does NOT download or load
    any ImageNet weights. We will load our local .pth file for inference.
    """
    model = smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights=None,  # <--- no auto-downloading
        in_channels=7,
        classes=1,
        activation=None,
    )
    return model
