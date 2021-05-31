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
parser.add_argument("--target", type=str)
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
    target = config["target"]
    output_directory = (
        f'data/{os.path.basename(config["config_file"][:-5])}/{target}'
        if "output_dir" not in config.keys()
        else config["output_dir"]
    )
    os.makedirs(output_directory, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=output_directory)

    print(f"{fgbg.get_date_time_tag()} - Generate dataset")
    if not bool(config["augment"]):
        dataset = fgbg.CleanDataset(
            hdf5_file=os.path.join(config["training_directory"], target, "data.hdf5"),
            json_file=os.path.join(config["training_directory"], target, "data.json"),
        )
    else:
        dataset = fgbg.AugmentedTripletDataset(
            hdf5_file=os.path.join(config["training_directory"], target, "data.hdf5"),
            json_file=os.path.join(config["training_directory"], target, "data.json"),
            background_images_directory=config["texture_directory"],
            blur=bool(config["blur"]),
        )
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )
    train_dataloader = TorchDataLoader(dataset=train_set, batch_size=100, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    val_dataloader = TorchDataLoader(dataset=val_set, batch_size=100, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)

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
    
    fgbg.train_autoencoder(
        model,
        train_dataloader,
        val_dataloader,
        checkpoint_file,
        tb_writer,
        triplet_loss=bool(config["triplet"]),
        num_epochs=config["number_of_epochs"],
    )
    # set weights to best validation checkpoint
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device("cpu"))
    )
    model.eval()

    print(f"{fgbg.get_date_time_tag()} - Evaluate Out-of-distribution")
    ood_dataset = fgbg.CleanDataset(
        hdf5_file=os.path.join(config["ood_directory"], target, "data.hdf5"),
        json_file=os.path.join(config["ood_directory"], target, "data.json"),
    )
    fgbg.evaluate_on_dataset(ood_dataset, model, tb_writer, save_outputs=True)

    print(f"{fgbg.get_date_time_tag()} - Evaluate qualitatively on real images")
    real_dataset = fgbg.ImagesDataset(target=target, dir_name=config["real_directory"])
    fgbg.evaluate_on_dataset(real_dataset, model, tb_writer, save_outputs=True)

    print(f"{fgbg.get_date_time_tag()} - Finished")
