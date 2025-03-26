import torch
import torch.nn as nn


# ====================
# Loss and Dice Metric
# ====================


class HybridLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)

        # Dice part
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum(dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (
            outputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        return bce_loss + dice_loss


def dice_coefficient(outputs, targets, threshold=0.5, smooth=1.0):
    outputs = (torch.sigmoid(outputs) > threshold).float()
    intersection = (outputs * targets).sum(dim=(1, 2, 3))
    dice_score = (2.0 * intersection + smooth) / (
        outputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
    )
    return dice_score.mean().item()
