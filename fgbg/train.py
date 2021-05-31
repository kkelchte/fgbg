import torch
from torch.nn import TripletMarginLoss
import numpy as np
from tqdm import tqdm

from .utils import get_date_time_tag
from .losses import WeightedBinaryCrossEntropyLoss


def train_autoencoder(
    autoencoder,
    train_dataloader,
    val_dataloader,
    checkpoint_file,
    tb_writer,
    triplet_loss: bool = False,
    num_epochs: int = 40,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9).to(device)
    if triplet_loss:
        trplt_loss = TripletMarginLoss(swap=True).to(device)
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=0.001, weight_decay=0.0001
    )
    for epoch in range(num_epochs):
        losses = {"train": [], "val": []}
        autoencoder.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            loss = bce_loss(
                autoencoder(data["reference"]).to(device), data["mask"].to(device)
            )
            if triplet_loss:
                anchor = autoencoder.encoder(data["reference"].to(device))
                positive = autoencoder.encoder(data["positive"].to(device))
                negative = autoencoder.encoder(data["negative"].to(device))
                loss += 0.1 * trplt_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
        autoencoder.eval()
        for batch_idx, data in enumerate(tqdm(val_dataloader)):
            loss = bce_loss(
                autoencoder(data["reference"]).to(device), data["mask"].to(device)
            )
            losses["val"].append(loss.cpu().detach().item())
        print(
            f"{get_date_time_tag()}: autoencder epoch {epoch} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        tb_writer.add_scalar(
            "train/bce_loss/autoencoder" + ("" if not triplet_loss else "_trplt"),
            np.mean(losses["train"]),
            global_step=epoch,
        )
        tb_writer.add_scalar(
            "val/bce_loss/autoencoder" + ("" if not triplet_loss else "_trplt"),
            np.mean(losses["val"]),
            global_step=epoch,
        )
        if lowest_validation_loss > np.mean(losses["val"]):
            print(f"Saving model in {checkpoint_file}")
            torch.save(
                autoencoder.state_dict(), checkpoint_file,
            )
            lowest_validation_loss = np.mean(losses["val"])
    autoencoder.to(torch.device("cpu"))


def train_encoder_with_triplet_loss(
    encoder, train_dataloader, val_dataloader, checkpoint_file, tb_writer
):
    triplet_loss = TripletMarginLoss(swap=True)
    for p in encoder.parameters():
        p.requires_grad = True
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=0.0001)
    for epoch in range(20):
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
        tb_writer.add_scalar(
            "train/triplet_loss/encoder", np.mean(losses["train"]), global_step=epoch
        )
        tb_writer.add_scalar(
            "val/triplet_loss/encoder", np.mean(losses["val"]), global_step=epoch
        )

        if lowest_validation_loss > np.mean(losses["val"]):
            print(f"Saving model in {checkpoint_file}")
            torch.save(
                encoder.state_dict(), checkpoint_file,
            )
            lowest_validation_loss = np.mean(losses["val"])


def train_decoder_with_frozen_encoder(
    encoder, decoder, train_dataloader, val_dataloader, checkpoint_file, tb_writer
):
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.0001)
    for epoch in range(20):
        losses = {"train": [], "val": []}
        decoder.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            projection = encoder(data["reference"])
            loss = bce_loss(decoder(projection), data["target"])
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
        decoder.eval()
        for batch_idx, data in enumerate(tqdm(val_dataloader)):
            projection = encoder(data["reference"])
            loss = bce_loss(decoder(projection), data["target"])
            losses["val"].append(loss.cpu().detach().item())
        print(
            f"{get_date_time_tag()}: decoder epoch {epoch} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        tb_writer.add_scalar(
            "train/bce_loss/decoder", np.mean(losses["train"]), global_step=epoch
        )
        tb_writer.add_scalar(
            "val/bce_loss/decoder", np.mean(losses["val"]), global_step=epoch
        )
        if lowest_validation_loss > np.mean(losses["val"]):
            print(f"Saving model in {checkpoint_file}")
            torch.save(
                decoder.state_dict(), checkpoint_file,
            )
            lowest_validation_loss = np.mean(losses["val"])
