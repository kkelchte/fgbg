import os

import torch
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from tqdm import tqdm

from .model import ResEncoder
from .utils import get_date_time_tag
from .data import LineDataset


def train_encoder_with_triplet_loss(encoder, train_dataloader, val_dataloader):
    triplet_loss = TripletMarginLoss(swap=True)
    for p in encoder.parameters():
        p.requires_grad = True
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=0.0001)
    for epoch in range(10):
        losses = {"train": [], "val": []}
        encoder.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            anchor = encoder(data["reference"])
            positive = encoder(data["positive"])
            negative = encoder(data["negative"])
            loss = triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
        encoder.eval()
        for batch_idx, data in enumerate(tqdm(val_dataloader)):
            anchor = encoder(data["reference"])
            positive = encoder(data["positive"])
            negative = encoder(data["negative"])
            loss = triplet_loss(anchor, positive, negative)
            losses["val"].append(loss.cpu().detach().item())
        print(
            f"{get_date_time_tag()}: epoch {epoch} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        if lowest_validation_loss > np.mean(losses["val"]):
            ckpt_file = "checkpoint_augmented_swap_smooth.ckpt"
            print(f'Saving model in {os.path.join(output_directory, ckpt_file)}')
            checkpoint = {"encoder": encoder.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(output_directory, ckpt_file),
            )
            lowest_validation_loss = np.mean(losses["val"])


if __name__ == "__main__":
    print(f"{get_date_time_tag()} - started")
    output_directory = "/Users/kelchtermans/data/contrastive_learning/line_encoder"
    os.makedirs(output_directory, exist_ok=True)

    encoder = ResEncoder()
    # if os.path.isfile(os.path.join(output_directory, "checkpoint.ckpt")):
    #     encoder.load_state_dict(
    #         torch.load(os.path.join(output_directory, "checkpoint.ckpt"))["encoder"]()
    #     )
    dataset = LineDataset(
        line_data_hdf5_file="/Users/kelchtermans/data/vanilla_128x128x3_pruned.hdf5",
        background_images_directory="/Users/kelchtermans/data/textured_dataset",
    )
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )

    train_dataloader = TorchDataLoader(dataset=train_set, batch_size=100, shuffle=True)
    val_dataloader = TorchDataLoader(dataset=val_set, batch_size=100, shuffle=True)

    train_encoder_with_triplet_loss(encoder, train_dataloader, val_dataloader)
    print(f"{get_date_time_tag()} - finished")
