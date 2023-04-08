import torch

from src.loss.DiceLoss import DiceLoss


class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.cross_entropy_loss(inputs, targets)
        combined_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        return combined_loss
