import os

import torch
from torch.nn import TripletMarginLoss, MSELoss
import numpy as np
from tqdm import tqdm

from .utils import get_date_time_tag, get_IoU
from .losses import WeightedBinaryCrossEntropyLoss


def train_downstream_task(
    model,
    train_dataloader,
    val_dataloader,
    checkpoint_file,
    tb_writer,
    task: str = "velocities",
    num_epochs: int = 40,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mse_loss = MSELoss()
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    if os.path.isfile(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.global_step = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lowest_validation_loss = ckpt["lowest_val_loss"]
        print(
            f"loaded checkpoint {checkpoint_file}. "
            f"Starting from {model.global_step}"
        )

    while model.global_step < num_epochs:
        losses = {"train": [], "val": []}
        model.train()
        for _, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            predictions = model(data["reference"].to(device))
            loss = mse_loss(predictions, data[task].to(device))
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
        model.eval()
        for _, data in enumerate(val_dataloader):
            predictions = model(data["reference"].to(device))
            loss = mse_loss(predictions, data[task].to(device))
            losses["val"].append(loss.cpu().detach().item())

        print(
            f"{get_date_time_tag()}: epoch {model.global_step} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        tb_writer.add_scalar(
            "train/mse_loss", np.mean(losses["train"]), global_step=model.global_step,
        )
        tb_writer.add_scalar(
            "val/mse_loss", np.mean(losses["val"]), global_step=model.global_step,
        )
        model.global_step += 1
        if lowest_validation_loss > np.mean(losses["val"]):
            lowest_validation_loss = np.mean(losses["val"])
            ckpt = {
                "state_dict": model.state_dict(),
                "global_step": model.global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "lowest_val_loss": lowest_validation_loss,
            }
            torch.save(
                ckpt, checkpoint_file,
            )
            with open(os.path.join(tb_writer.get_logdir(), "results.txt"), "w") as f:
                f.write(
                    f"validation_mse_loss_avg: " f"{np.mean(losses['val']):10.3e}\n",
                )
                f.write(
                    f"validation_mse_loss_std: " f"{np.std(losses['val']):10.2e}\n",
                )
            print(f"Saved model in {checkpoint_file}.")
    model.to(torch.device("cpu"))


def train_autoencoder(
    autoencoder,
    train_dataloader,
    val_dataloader,
    checkpoint_file,
    tb_writer,
    triplet_loss_weight: float = 0.0,
    num_epochs: int = 40,
    deep_supervision: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9).to(device)
    if triplet_loss_weight != 0.0:
        trplt_loss = TripletMarginLoss(swap=True).to(device)
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=0.001, weight_decay=0.0001
    )
    if os.path.isfile(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device)
        autoencoder.load_state_dict(ckpt["state_dict"])
        autoencoder.global_step = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lowest_validation_loss = ckpt["lowest_val_loss"]
        print(
            f"loaded checkpoint {checkpoint_file}. "
            f"Starting from {autoencoder.global_step}"
        )

    while autoencoder.global_step < num_epochs:
        losses = {"train": [], "val": []}
        ious = {"train": [], "val": []}
        autoencoder.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            predictions = autoencoder(
                data["reference"].to(device), intermediate_outputs=deep_supervision
            )
            if not deep_supervision:
                loss = bce_loss(predictions, data["mask"].to(device))
            else:
                loss = 0
                for output in predictions:
                    loss += (
                        1 / len(predictions) * bce_loss(output, data["mask"].to(device))
                    )
            if triplet_loss_weight != 0:
                anchor = autoencoder.project(data["reference"].to(device))
                positive = autoencoder.project(data["positive"].to(device))
                negative = autoencoder.project(data["negative"].to(device))
                triplet = triplet_loss_weight * trplt_loss(anchor, positive, negative)
                # print(
                # f"loss: {loss} vs trplt: {triplet} (weight: {triplet_loss_weight})"
                # )
                loss += triplet
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
            ious["train"].append(
                get_IoU(
                    autoencoder(
                        data["reference"].to(device), intermediate_outputs=False
                    ),
                    data["mask"].to(device),
                )
                .detach()
                .cpu()
                .item()
            )
        autoencoder.eval()
        for batch_idx, data in enumerate(val_dataloader):
            predictions = autoencoder(
                data["reference"].to(device), intermediate_outputs=False
            )
            loss = bce_loss(predictions, data["mask"].to(device))
            losses["val"].append(loss.cpu().detach().item())
            ious["val"].append(
                get_IoU(predictions, data["mask"].to(device))
                .detach()
                .cpu()
                .item()
            )

        print(
            f"{get_date_time_tag()}: epoch {autoencoder.global_step} - "
            f"train {np.mean(losses['train']): 0.3f} [{np.std(losses['train']): 0.2f}]"
            f" - val {np.mean(losses['val']): 0.3f} [{np.std(losses['val']): 0.2f}]"
        )
        tb_writer.add_scalar(
            "train/bce_loss/autoencoder"
            + ("" if not triplet_loss_weight else "_trplt"),
            np.mean(losses["train"]),
            global_step=autoencoder.global_step,
        )
        tb_writer.add_scalar(
            "val/bce_loss/autoencoder" + ("" if not triplet_loss_weight else "_trplt"),
            np.mean(losses["val"]),
            global_step=autoencoder.global_step,
        )
        tb_writer.add_scalar(
            "train/iou/autoencoder" + ("" if not triplet_loss_weight else "_trplt"),
            np.mean(ious["train"]),
            global_step=autoencoder.global_step,
        )
        tb_writer.add_scalar(
            "val/iou/autoencoder" + ("" if not triplet_loss_weight else "_trplt"),
            np.mean(ious["val"]),
            global_step=autoencoder.global_step,
        )
        autoencoder.global_step += 1
        if lowest_validation_loss > np.mean(losses["val"]):
            lowest_validation_loss = np.mean(losses["val"])
            ckpt = {
                "state_dict": autoencoder.state_dict(),
                "global_step": autoencoder.global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "lowest_val_loss": lowest_validation_loss,
            }
            torch.save(
                ckpt, checkpoint_file,
            )
            with open(os.path.join(tb_writer.get_logdir(), "results.txt"), "w") as f:
                f.write(
                    f"validation_bce_loss_avg: " f"{np.mean(losses['val']):10.3e}\n",
                )
                f.write(
                    f"validation_bce_loss_std: " f"{np.std(losses['val']):10.2e}\n",
                )
                f.write(f"validation_iou_avg: " f"{np.mean(ious['val']):10.3e}\n",)
                f.write(f"validation_iou_std: " f"{np.std(ious['val']):10.2e}\n",)
            print(f"Saved model in {checkpoint_file}.")
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
