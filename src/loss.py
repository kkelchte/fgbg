import torch.nn.functional as F
import torch.nn as nn


# Define triplet loss
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples
    and a target label == 1 if samples are from the same class
    and label == 0 otherwise.
    Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()
