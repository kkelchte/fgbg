import os

import torch
import torchvision
from torch.utils.data import DataLoader as TorchDataLoader
import matplotlib.pyplot as plt

import fgbg


def test_model_architecture():
    model = fgbg.AutoEncoder(
        feature_size=512,
        projected_size=512,
        input_channels=3,
        decode_from_projection=True,
    )

    feature = model.encoder(torch.rand(1, 3, 128, 128))
    print(feature.shape)
    print(model.decoder(feature).shape)


def test_data_loading_clean():
    dataset = fgbg.CleanDataset(
        hdf5_file="data/debug_data/cone/data.hdf5",
        json_file="data/debug_data/cone/data.json",
    )
    dataloader = TorchDataLoader(dataset, 9, shuffle=True)
    for batch in dataloader:
        grid = torchvision.utils.make_grid(batch["observation"], nrow=3)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.show()
        break


def test_data_loading_augment():
    dataset = fgbg.AugmentedTripletDataset(
        hdf5_file="data/debug_data/cone/data.hdf5",
        json_file="data/debug_data/cone/data.json",
        target="cone",
        background_images_directory="data/datasets/dtd",
    )
    dataloader = TorchDataLoader(dataset, 9, shuffle=True)
    for batch in dataloader:
        grid_observation = torchvision.utils.make_grid(batch['observation'], nrow=3)
        plt.imshow(grid_observation.permute(1, 2, 0).numpy())
        plt.title('observation')
        plt.show()
        grid_positive = torchvision.utils.make_grid(batch['positive'], nrow=3)
        plt.imshow(grid_positive.permute(1, 2, 0).numpy())
        plt.title('positive')
        plt.show()
        grid_negative = torchvision.utils.make_grid(batch['negative'], nrow=3)
        plt.imshow(grid_negative.permute(1, 2, 0).numpy())
        plt.title('negative')
        plt.show()
        break


def test_foreground_map():
    image = fgbg.create_random_gradient_image(size=(200, 200, 3))
    plt.imshow(image)
    plt.show()


def test_data_image_sequence():
    output_dir = 'data/test'
    model = fgbg.DeepSupervisionNet(batch_norm=False)
    checkpoint_file = os.path.join('data/best_encoders/cone', "checkpoint_model.ckpt")
    ckpt = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])

    real_dataset = fgbg.ImageSequenceDataset(
        hdf5_file="data/datasets/bebop_real_movies/cone/pruned_data.hdf5"
    )
    fgbg.evaluate_qualitatively_on_sequences(
        "eval_real_sequence", real_dataset, model, output_dir
    )
    print(f'stored in {output_dir}')
    print('done')


if __name__ == "__main__":
    test_data_image_sequence()
