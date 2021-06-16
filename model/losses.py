from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MixLoss', 'DiceLoss']


class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, logits, targets):
        loss, loss_weight = [], []
        for idx, v in enumerate(self.args):
            if idx % 2 == 0:
                loss.append(v)
            else:
                loss_weight.append(v)
        mix = sum([w * l(logits, targets) for l, w in zip(loss, loss_weight)])
        return mix



class softDiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, logits, targets):
        logits = logits.sigmoid()
        smooth = 1
        intersection, union = [tmp.flatten(1).sum(1) if self.image else tmp.sum() for tmp in [logits * targets, logits + targets]]

        score = (2 * intersection + smooth) / (union + smooth)
        score = 1 - score.mean()
        return score
