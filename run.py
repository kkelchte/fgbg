import os
from argparse import ArgumentParser

import json
from pprint import pprint
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter

import fgbg

parser = ArgumentParser()
parser.add_argument("--config_file")
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--output_dir", type=str)
config = vars(parser.parse_args())
if config["config_file"] is not None:
    with open(config["config_file"], "r") as f:
        json_config = json.load(f)
    for k, v in config.items():
        if v is not None:
            json_config[k] = v
    config = json_config  # update config to json's config
pprint(config)

if __name__ == "__main__":
    output_directory = (
        f'data/{config["config_file"][:-5]}'
        if "output_dir" not in config.keys()
        else config["output_dir"]
    )
    os.makedirs(output_directory, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=output_directory)

    print(f"{fgbg.get_date_time_tag()} - Generate dataset")
    if not bool(config["augment"]):
        dataset = fgbg.CleanDataset(
            hdf5_file=config["training_directory"] + "/data.hdf5",
            json_file=config["training_directory"] + "/data.json",
        )
    else:
        dataset = fgbg.AugmentedTripletDataset(
            hdf5_file=config["training_directory"] + "/data.hdf5",
            json_file=config["training_directory"] + "/data.json",
            background_images_directory=config["texture_directory"],
            blur=bool(config["blur"]),
        )
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )
    train_dataloader = TorchDataLoader(dataset=train_set, batch_size=100, shuffle=True)
    val_dataloader = TorchDataLoader(dataset=val_set, batch_size=100, shuffle=True)

    print(f"{fgbg.get_date_time_tag()} - Train autoencoder")
    model = fgbg.AutoEncoder(
        feature_size=512,
        projected_size=512,
        input_channels=3,
        decode_from_projection=True,
    )
    checkpoint_file = os.path.join(output_directory, "checkpoint_model.ckpt")
    if os.path.isfile(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file))
    else:
        fgbg.train_autoencoder(
            model,
            train_dataloader,
            val_dataloader,
            checkpoint_file,
            tb_writer,
            triplet_loss=bool(config["triplet"]),
        )
    print(f"{fgbg.get_date_time_tag()} - Evaluate Out-of-distribution")
    fgbg.evaluate_models(
        ood_dataset,
        
    )

    print(f"{fgbg.get_date_time_tag()} - Finished")
