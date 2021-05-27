import torch

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


def test_data_loading():
    dataset = fgbg.CleanDataset(
            hdf5_file="data/gate_cone_line/cone/data.hdf5",
            json_file="data/gate_cone_line/cone/data.json",
        )
    data_item = dataset[0]
    print(data_item['mask'].min(), data_item['mask'].max())


if __name__ == "__main__":
    test_data_loading()