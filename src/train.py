import os

import torch
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np

from model import ResEncoder
from loss import TripletLoss
from utils import get_date_time_tag
from data import LineDataset


def train_encoder_with_triplet_loss(encoder, torch_dataloader):
    triplet_loss = TripletLoss()
    for p in encoder.parameters():
        p.requires_grad = True
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=0.0001)
    for epoch in range(10):
        losses = []
        for batch_idx, data in enumerate(torch_dataloader):
            optimizer.zero_grad()
            anchor = encoder(data["reference"])
            positive = encoder(data["positive"])
            negative = encoder(data["negative"])
            loss = triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().item())
        print(
            f"{get_date_time_tag()}: epoch {epoch} - "
            f"loss {np.mean(losses): 0.3f} [{np.std(losses): 0.2f}]"
        )


if __name__ == "__main__":
    output_directory = "/Users/kelchtermans/data/contrastive_learning/line_encoder"
    os.makedirs(output_directory, exist_ok=True)
    encoder = ResEncoder()
    dataset = LineDataset(
        line_data_hdf5_file="/Users/kelchtermans/data/vanilla_128x128x3_pruned.hdf5",
        background_images_directory="/Users/kelchtermans/data/textured_dataset",
    )
    torch_dataloader = TorchDataLoader(dataset=dataset, batch_size=100, shuffle=True)
    train_encoder_with_triplet_loss(encoder=encoder, torch_dataloader=torch_dataloader)
    checkpoint = {'encoder': encoder.state_dict}
    torch.save(checkpoint, os.path.join(output_directory, 'checkpoint.ckpt'))
    print("finished")
