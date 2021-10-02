import os
from argparse import ArgumentParser
import shutil

import json
from pprint import pprint
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter

import fgbg

parser = ArgumentParser()
parser.add_argument("--config_file")
parser.add_argument("--task", type=str)
parser.add_argument("--texture_directory", type=str)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--encoder_ckpt_dir", type=str)
parser.add_argument("--evaluate", type=bool, default=False)
parser.add_argument("--rm", type=bool, default=False)
parser.add_argument("--batch_normalisation", type=bool)
parser.add_argument("--end_to_end", type=bool)
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
    if config["rm"] and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "config"), "w") as f:
        json.dump(config, f)

    tb_writer = SummaryWriter(log_dir=output_directory)
    checkpoint_file = os.path.join(output_directory, "checkpoint_model.ckpt")
    if config["task"] == "pretrain":
        model = fgbg.DeepSupervisionNet(batch_norm=config["batch_normalisation"])
    elif config["task"] == "velocities":
        model = fgbg.DownstreamNet(
            output_size=(4,),
            encoder_ckpt_dir=config["encoder_ckpt_dir"],
            end_to_end=bool(config["end_to_end"]),
            batch_norm=bool(config["batch_normalisation"]),
        )
    elif config["task"] == "waypoints":
        model = fgbg.DownstreamNet(
            output_size=(3,),
            encoder_ckpt_dir=config["encoder_ckpt_dir"],
            end_to_end=bool(config["end_to_end"]),
            batch_norm=bool(config["batch_normalisation"]),
        )

    print(f"{fgbg.get_date_time_tag()} - Generate dataset")
    if not bool(config["augment"]):
        dataset = fgbg.CleanDataset(
            hdf5_file=os.path.join(config["training_directory"], target, "data.hdf5"),
            json_file=os.path.join(config["training_directory"], target, "data.json"),
            fg_augmentation=bool(config["fg_augmentation"])
        )
    else:
        dataset = fgbg.AugmentedTripletDataset(
            hdf5_file=os.path.join(config["training_directory"], target, "data.hdf5"),
            json_file=os.path.join(config["training_directory"], target, "data.json"),
            background_images_directory=config["texture_directory"],
            blur=bool(config["blur"]),
            fg_augmentation=bool(config["fg_augmentation"])
        )
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )
    if not config["evaluate"]:
        train_dataloader = TorchDataLoader(
            dataset=train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
        )
        val_dataloader = TorchDataLoader(
            dataset=val_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
        )

        print(f"{fgbg.get_date_time_tag()} - Train autoencoder")
        if config["task"] == "pretrain":
            fgbg.train_autoencoder(
                model,
                train_dataloader,
                val_dataloader,
                checkpoint_file,
                tb_writer,
                triplet_loss_weight=config["triplet"],
                num_epochs=config["number_of_epochs"],
                learning_rate=config["learning_rate"],
            )
        else:
            fgbg.train_downstream_task(
                model,
                train_dataloader,
                val_dataloader,
                checkpoint_file,
                tb_writer,
                task=config["task"],
                num_epochs=config["number_of_epochs"],
                learning_rate=config["learning_rate"],
            )
    # set weights to best validation checkpoint
    ckpt = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    model.global_step = ckpt["global_step"]
    model.eval()

    if config["task"] == "pretrain":
        print(f"{fgbg.get_date_time_tag()} - Evaluate on training images")
        fgbg.evaluate_qualitatively_on_dataset("training", train_set, model, tb_writer)

        print(f"{fgbg.get_date_time_tag()} - Evaluate on validation images")
        fgbg.evaluate_qualitatively_on_dataset("validation", val_set, model, tb_writer)

        print(f"{fgbg.get_date_time_tag()} - Evaluate on real image")
        real_dataset = fgbg.ImagesDataset(
            target=target, dir_name=config["real_directory"]
        )
        fgbg.evaluate_qualitatively_on_dataset("real", real_dataset, model, tb_writer)
        print(f"{fgbg.get_date_time_tag()} - Evaluate on real image sequence")
        real_dataset = fgbg.ImageSequenceDataset(
            hdf5_file=os.path.join(
                config["real_sequence_directory"], target, "pruned_data.hdf5"
            )
        )
        fgbg.evaluate_qualitatively_on_sequences(
            "eval_real_sequence", real_dataset, model, output_directory
        )

    print(f"{fgbg.get_date_time_tag()} - Evaluate on out-of-distribution images")
    ood_dataset = fgbg.CleanDataset(
        hdf5_file=os.path.join(config["ood_directory"], target, "data.hdf5"),
        json_file=os.path.join(config["ood_directory"], target, "data.json"),
        fg_augmentation=False
    )
    if config["task"] == "pretrain":
        fgbg.evaluate_qualitatively_on_dataset(
            "out-of-distribution", ood_dataset, model, tb_writer
        )
    fgbg.evaluate_quantitatively_on_dataset(
        "out-of-distribution", ood_dataset, model, tb_writer, config["task"]
    )

    print(f"{fgbg.get_date_time_tag()} - Finished")
    os.system(f'touch {output_directory}/FINISHED')
