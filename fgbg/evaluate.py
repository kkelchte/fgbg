import torch
import matplotlib.pyplot as plt
import numpy as np
import json

from .utils import normalize


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
