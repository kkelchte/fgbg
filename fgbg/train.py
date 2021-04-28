import json
import torch
from torch.nn import TripletMarginLoss
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import get_date_time_tag, normalize
from .losses import WeightedBinaryCrossEntropyLoss


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


def train_autoencoder(
    autoencoder,
    train_dataloader,
    val_dataloader,
    checkpoint_file,
    tb_writer,
    triplet_loss: bool = False,
):
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9)
    if triplet_loss:
        trplt_loss = TripletMarginLoss(swap=True)
    lowest_validation_loss = 100
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=0.001, weight_decay=0.0001
    )
    for epoch in range(40):
        losses = {"train": [], "val": []}
        autoencoder.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            loss = bce_loss(autoencoder(data["reference"]), data["target"])
            if triplet_loss:
                anchor = autoencoder.encoder(data["reference"])
                positive = autoencoder.encoder(data["positive"])
                negative = autoencoder.encoder(data["negative"])
                loss += 0.01 * trplt_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            losses["train"].append(loss.cpu().detach().item())
        autoencoder.eval()
        for batch_idx, data in enumerate(tqdm(val_dataloader)):
            loss = bce_loss(autoencoder(data["reference"]), data["target"])
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


def evaluate_models(ood_dataset, autoencoder_trplt, autoencoder, output_file):
    autoencoder_trplt.eval()
    autoencoder.eval()
    projection_results = {"trplt": [], "ae": []}
    reconstruction_results = {"trplt": [], "ae": []}
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            data_item = ood_dataset[i + 3 * j]

            # get encodings and compare distance
            stacked_input = torch.stack(
                [data_item["reference"], data_item["positive"], data_item["negative"]]
            )
            trplt_projection = autoencoder_trplt.encoder(stacked_input)
            ae_projection = autoencoder.encoder(stacked_input)

            def avg_neg_vs_pos_distance(projection):
                ref_x = projection[0]
                pos_x = projection[1]
                neg_x = projection[2]
                pos_dis = torch.norm((ref_x - pos_x))
                neg_dis = torch.norm((ref_x - neg_x))
                return (neg_dis - pos_dis).mean().detach().cpu().numpy().item()

            projection_results["trplt"].append(
                avg_neg_vs_pos_distance(trplt_projection)
            )
            projection_results["ae"].append(avg_neg_vs_pos_distance(ae_projection))

            # get reconstructions and add quantitative results
            reconstruction_triplet = autoencoder_trplt(
                data_item["reference"].unsqueeze(0)
            )
            reconstruction_ae = autoencoder(data_item["reference"].unsqueeze(0))
            reconstruction_loss = torch.nn.L1Loss()
            reconstruction_results["trplt"].append(
                reconstruction_loss(
                    reconstruction_triplet.squeeze(), data_item["target"].squeeze()
                ).item()
            )
            reconstruction_results["ae"].append(
                reconstruction_loss(
                    reconstruction_ae.squeeze(), data_item["target"].squeeze()
                ).item()
            )

            # add reconstruction to overview plots
            stacked = np.stack(
                [
                    data_item["reference"].permute(1, 2, 0).numpy().squeeze(),
                    normalize(reconstruction_ae.cpu().detach().squeeze().numpy()),
                    normalize(reconstruction_triplet.cpu().detach().squeeze().numpy()),
                ],
                axis=-1,
            )
            axes[i, j].imshow(stacked)
            axes[i, j].text(0, 5, "Image", color=(1, 0, 0))
            axes[i, j].text(0, 12, "AE", color=(0, 1, 0))
            axes[i, j].text(0, 19, "Triplet", color=(0, 0, 1))
            axes[i, j].axis("off")

    fig.suptitle("Red: image, Green: AE, Blue: Triplet")
    plt.tight_layout()
    plt.savefig(output_file + ".jpg")
    results = {
        "reconstruction L1": {
            "ae": np.mean(reconstruction_results["ae"]),
            "triplet": np.mean(reconstruction_results["trplt"]),
        },
        "projection distance": {
            "ae": np.mean(projection_results["ae"]),
            "triplet": np.mean(projection_results["trplt"]),
        },
    }
    with open(output_file + ".json", "w") as f:
        json.dump(
            results, f,
        )
