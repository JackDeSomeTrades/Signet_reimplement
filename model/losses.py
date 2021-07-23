import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_dist = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean_dist, 2) + \
                           label * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))

        return contrastive_loss
