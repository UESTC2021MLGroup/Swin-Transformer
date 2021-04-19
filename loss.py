import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, x, target):
        preds = torch.sigmoid(x)
        loss = self.bceloss(preds, target)
        return loss
