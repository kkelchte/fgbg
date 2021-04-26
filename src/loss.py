import torch.nn.functional as F
import torch.nn as nn


# Define triplet loss
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=10):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        # print(f'pos dis: {distance_positive}, neg dis: {distance_negative}')
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
