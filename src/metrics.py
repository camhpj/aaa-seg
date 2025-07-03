
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-7) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.view(-1)
        target = target.view(-1)
        intersection = (input * target).sum()
        dice = (2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)
        return 1 - dice
