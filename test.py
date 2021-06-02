import torch
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
        hdf5_file="data/gate_cone_line/cone/data.hdf5",
        json_file="data/gate_cone_line/cone/data.json",
    )
    data_item = dataset[0]
    print(data_item["mask"].min(), data_item["mask"].max())


def test_data_loading_augment():
    dataset = fgbg.AugmentedTripletDataset(
        hdf5_file="data/gate_cone_line/cone/data.hdf5",
        json_file="data/gate_cone_line/cone/data.json",
        background_images_directory="data/textured_dataset",
    )
    data_item = dataset[0]
    # for k in data_item.keys():
    #    print(k, data_item[k].shape)

    plt.imshow(data_item["observation"].permute(1, 2, 0).numpy())
    plt.show()


if __name__ == "__main__":
    test_data_loading_augment()
