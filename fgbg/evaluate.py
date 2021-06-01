import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset as TorchDataset

from .utils import normalize, get_IoU
from .losses import WeightedBinaryCrossEntropyLoss


def evaluate_on_dataset(
    dataset: TorchDataset,
    model: Module,
    tb_writer: SummaryWriter,
    save_outputs: bool = False,
):
    losses = []
    ious = []
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9)
    if save_outputs:
        save_dir = os.path.join(tb_writer.get_logdir(), "imgs")
        os.makedirs(save_dir, exist_ok=True)
    for _ in range(min(len(dataset), 100)):
        data = dataset[_]
        prediction = model(data["observation"].unsqueeze(0))
        if "mask" in data.keys():
            loss = bce_loss(prediction, data["mask"])
            losses.append(loss.detach().cpu())
            ious.append(get_IoU(prediction.squeeze(0), data["mask"].unsqueeze(0)).detach().cpu())
        if save_outputs and _ == 0:  # store first image
            mask = prediction.detach().cpu().squeeze().numpy()
            obs = data["observation"].detach().cpu().squeeze().permute(1, 2, 0).numpy()
            combined = obs * np.stack([mask + 0.3] * 3, axis=-1)
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].imshow(obs)
            ax[0].axis("off")
            ax[1].imshow(mask)
            ax[1].axis("off")
            ax[2].imshow(combined)
            ax[2].axis("off")
            fig.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f'{dataset.name.replace("/", "_")}_{_}.jpg')
            )
            tb_writer.add_image(dataset.name.replace("/", "_"), combined, dataformats="HWC")
    if len(losses) != 0:
        tb_writer.add_scalar(
            dataset.name.replace("/", "_") + "_bce_loss_avg",
            torch.as_tensor(losses).mean(),
            global_step=model.global_step,
        )
        tb_writer.add_scalar(
            dataset.name.replace("/", "_") + "_bce_loss_std",
            torch.as_tensor(losses).std(),
            global_step=model.global_step,
        )
        tb_writer.add_scalar(
            dataset.name.replace("/", "_") + "_iou_avg",
            torch.as_tensor(ious).mean(),
            global_step=model.global_step,
        )
        tb_writer.add_scalar(
            dataset.name.replace("/", "_") + "_iou_std",
            torch.as_tensor(ious).std(),
            global_step=model.global_step,
        )
        with open(os.path.join(tb_writer.get_logdir(), "results.txt"), "w") as f:
            f.write(
                f"{dataset.name.replace('/', '_')}_bce_loss_avg: "
                f"{torch.as_tensor(losses).mean()}\n",
            )
            f.write(
                f"{dataset.name.replace('/', '_')}_bce_loss_std: "
                f"{torch.as_tensor(losses).std()}\n",
            )
            f.write(
                f"{dataset.name.replace('/', '_')}_ious_avg: "
                f"{torch.as_tensor(ious).mean()}\n",
            )
            f.write(
                f"{dataset.name.replace('/', '_')}_ious_std: "
                f"{torch.as_tensor(ious).std()}\n",
            )


def compare_models(ood_dataset, autoencoder_trplt, autoencoder, output_file):
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
