import torch
from torch.nn import TripletMarginLoss, BCELoss
import numpy as np
from tqdm import tqdm

from .utils import get_date_time_tag
from .losses import WeightedBinaryCrossEntropyLoss

def train_encoder_with_triplet_loss(
    encoder, train_dataloader, val_dataloader, checkpoint_file
):
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
            f"{get_date_time_tag()}: encoder epoch {epoch} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        if lowest_validation_loss > np.mean(losses["val"]):
            print(f"Saving model in {checkpoint_file}")
            checkpoint = {"encoder": encoder.state_dict()}
            torch.save(
                checkpoint, checkpoint_file,
            )
            lowest_validation_loss = np.mean(losses["val"])


def train_decoder_with_frozen_encoder(
    encoder, decoder, train_dataloader, val_dataloader, checkpoint_file
):
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01, weight_decay=0.0001)
    for epoch in range(10):
        losses = {"train": [], "val": []}
        decoder.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            loss = bce_loss(decoder(encoder(data["reference"])), data["target"])
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
        decoder.eval()
        for batch_idx, data in enumerate(tqdm(val_dataloader)):
            loss = bce_loss(decoder(encoder(data["reference"])), data["target"])
            losses["val"].append(loss.cpu().detach().item())
        print(
            f"{get_date_time_tag()}: decoder epoch {epoch} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        if lowest_validation_loss > np.mean(losses["val"]):
            print(f"Saving model in {checkpoint_file}")
            checkpoint = {"decoder": decoder.state_dict()}
            torch.save(
                checkpoint, checkpoint_file,
            )
            lowest_validation_loss = np.mean(losses["val"])
