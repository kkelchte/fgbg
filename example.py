import os

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from fgbg import ResEncoder, get_date_time_tag, LineDataset, Decoder

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

#    train_encoder_with_triplet_loss(encoder, train_dataloader, val_dataloader)

#    train_decoder(encoder, train_dataloader, val_dataloader)
    print(f"{get_date_time_tag()} - finished")
