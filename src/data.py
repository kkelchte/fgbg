import os
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from utils import get_binary_mask, load_img, combine


class LineDataset(TorchDataset):
    def __init__(
        self, line_data_hdf5_file: str, background_images_directory: str,
    ):
        self.hdf5_file = h5py.File(line_data_hdf5_file, "r", libver="latest", swmr=True)
        self.observations = self.hdf5_file["dataset"]["observations"]
        self.background_images = [
            os.path.join(background_images_directory, sub_directory, image)
            for sub_directory in os.listdir(background_images_directory)
            if os.path.isdir(os.path.join(background_images_directory, sub_directory))
            for image in os.listdir(
                os.path.join(background_images_directory, sub_directory)
            )
            if image.endswith(".jpg")
        ]
        self._size = len(self.observations)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # 128x128x3
        image = self.observations[index]

        # create binary mask as target: 128x128
        target = get_binary_mask(image)

        # select foreground color and background map
        background_img = load_img(
            np.random.choice(self.background_images), size=image.shape
        )
        foreground_img = np.zeros(image.shape)
        foreground_img[:, :, 0] = np.random.uniform(0, 1)
        foreground_img[:, :, 1] = np.random.uniform(0, 1)
        foreground_img[:, :, 2] = np.random.uniform(0, 1)

        # combine both as reference image
        reference = combine(target, foreground_img, background_img)

        # add different background for positive sample
        new_background_img = load_img(
            np.random.choice(self.background_images), size=image.shape
        )
        positive = combine(
            target, foreground_img, new_background_img
        )

        # get different line with different background for negative sample
        random_other_index = index
        # make sure new index is at least 5 frames away
        while abs(random_other_index - index) < 5:
            random_other_index = np.random.randint(0, self._size)

        new_image = self.observations[random_other_index]
        negative = combine(
            get_binary_mask(new_image), foreground_img, background_img
        )
        return {
            'target': torch.from_numpy(target).permute(2, 0, 1).float(),
            'reference': torch.from_numpy(reference).permute(2, 0, 1).float(),
            'positive': torch.from_numpy(positive).permute(2, 0, 1).float(),
            'negative': torch.from_numpy(negative).permute(2, 0, 1).float()
        }
