import os
from typing import Dict
import copy

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .utils import get_binary_mask, load_img, combine, generate_random_square


class SquareCircleDataset(TorchDataset):
    def __len__(self) -> int:
        return 1000

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target_img = generate_random_square()
        # copy target image with square to create input image
        reference_img = copy.deepcopy(target_img)
        # Add random circle
        height, width = 128, 128
        color = (1, 1, 1)
        circle_radius = 5 + np.random.randint(int(width / 8))
        circle_location = (
            circle_radius + np.random.randint(width - 2 * circle_radius),
            circle_radius + np.random.randint(height - 2 * circle_radius),
        )
        cv2.circle(reference_img, circle_location, circle_radius, color, -1)
        # negative image with same circle
        negative_img = generate_random_square()
        cv2.circle(negative_img, circle_location, circle_radius, color, -1)
        # positive image with new circle
        positive_img = copy.deepcopy(target_img)
        circle_radius = 5 + np.random.randint(int(width / 8))
        circle_location = (
            circle_radius + np.random.randint(width - 2 * circle_radius),
            circle_radius + np.random.randint(height - 2 * circle_radius),
        )
        cv2.circle(positive_img, circle_location, circle_radius, color, -1)
        result = {
            "reference": torch.from_numpy(reference_img).permute(2, 0, 1).float(),
            "positive": torch.from_numpy(positive_img).permute(2, 0, 1).float(),
            "negative": torch.from_numpy(negative_img).permute(2, 0, 1).float(),
            "target": torch.from_numpy(target_img).permute(2, 0, 1).float(),
        }
        return result


class SquareDoubleCircleDataset(TorchDataset):
    def __len__(self) -> int:
        return 1000

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target_img = generate_random_square()
        # copy target image with square to create input image
        reference_img = copy.deepcopy(target_img)
        # Add two random circles
        height, width = 128, 128
        color = (1, 1, 1)
        circle_radius = 5 + np.random.randint(int(width / 8))
        circle_location = (
            circle_radius + np.random.randint(width - 2 * circle_radius),
            circle_radius + np.random.randint(height - 2 * circle_radius),
        )
        cv2.circle(reference_img, circle_location, circle_radius, color, -1)
        circle_radius_two = 5 + np.random.randint(int(width / 8))
        circle_location_two = (
            circle_radius_two + np.random.randint(width - 2 * circle_radius_two),
            circle_radius_two + np.random.randint(height - 2 * circle_radius_two),
        )
        cv2.circle(reference_img, circle_location_two, circle_radius_two, color, -1)
        # negative image with same circle
        negative_img = generate_random_square()
        cv2.circle(negative_img, circle_location, circle_radius, color, -1)
        cv2.circle(negative_img, circle_location_two, circle_radius_two, color, -1)
        # positive image with new circle
        positive_img = copy.deepcopy(target_img)
        circle_radius = 5 + np.random.randint(int(width / 8))
        circle_location = (
            circle_radius + np.random.randint(width - 2 * circle_radius),
            circle_radius + np.random.randint(height - 2 * circle_radius),
        )
        cv2.circle(positive_img, circle_location, circle_radius, color, -1)
        circle_radius_two = 5 + np.random.randint(int(width / 8))
        circle_location_two = (
            circle_radius_two + np.random.randint(width - 2 * circle_radius_two),
            circle_radius_two + np.random.randint(height - 2 * circle_radius_two),
        )
        cv2.circle(positive_img, circle_location_two, circle_radius_two, color, -1)
        result = {
            "reference": torch.from_numpy(reference_img).permute(2, 0, 1).float(),
            "positive": torch.from_numpy(positive_img).permute(2, 0, 1).float(),
            "negative": torch.from_numpy(negative_img).permute(2, 0, 1).float(),
            "target": torch.from_numpy(target_img).permute(2, 0, 1).float(),
        }
        return result


class SquareTriangleDataset(TorchDataset):
    def __len__(self) -> int:
        return 1000

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        width, height = 128, 128
        target_img = generate_random_square()
        # copy target image with square to create input image
        reference_img = copy.deepcopy(target_img)

        # add triangle
        square_location = (
            np.random.randint(width - 50),
            np.random.randint(height - 50),
        )
        contour = [
            (
                square_location[0] + np.random.randint(-10, 10),
                square_location[0] + np.random.randint(-10, 10),
            )
            for _ in range(3)
        ]
        cv2.drawContours(reference_img, [np.asarray(contour)], 0, (1, 1, 1), -1)

        # negative image with same traingle
        negative_img = generate_random_square()
        cv2.drawContours(negative_img, [np.asarray(contour)], 0, (1, 1, 1), -1)

        # positive image with new triangle
        positive_img = copy.deepcopy(target_img)
        square_location = (
            np.random.randint(width - 50),
            np.random.randint(height - 50),
        )
        contour = [
            (
                square_location[0] + np.random.randint(-10, 10),
                square_location[0] + np.random.randint(-10, 10),
            )
            for _ in range(3)
        ]
        cv2.drawContours(positive_img, [np.asarray(contour)], 0, (1, 1, 1), -1)

        result = {
            "reference": torch.from_numpy(reference_img).permute(2, 0, 1).float(),
            "positive": torch.from_numpy(positive_img).permute(2, 0, 1).float(),
            "negative": torch.from_numpy(negative_img).permute(2, 0, 1).float(),
            "target": torch.from_numpy(target_img).permute(2, 0, 1).float(),
        }
        return result


class LineDataset(TorchDataset):
    def __init__(
        self, line_data_hdf5_file: str, background_images_directory: str = None,
    ):
        self.hdf5_file = h5py.File(line_data_hdf5_file, "r", libver="latest", swmr=True)
        self.observations = self.hdf5_file["dataset"]["observations"]
        self.background_images = (
            [
                os.path.join(background_images_directory, sub_directory, image)
                for sub_directory in os.listdir(background_images_directory)
                if os.path.isdir(
                    os.path.join(background_images_directory, sub_directory)
                )
                for image in os.listdir(
                    os.path.join(background_images_directory, sub_directory)
                )
                if image.endswith(".jpg")
            ]
            if background_images_directory is not None
            else []
        )
        self._size = len(self.observations)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # 128x128x3
        image = self.observations[index]

        # create binary mask as target: 128x128
        target = get_binary_mask(image, gaussian_blur=True)

        # select foreground color and background map
        background_img = (
            load_img(np.random.choice(self.background_images), size=image.shape)
            if len(self.background_images) != 0
            else np.zeros(image.shape) + np.random.uniform(0, 1)
        )
        foreground_img = np.zeros(image.shape)
        foreground_img[:, :, 0] = np.random.uniform(0, 1)
        foreground_img[:, :, 1] = np.random.uniform(0, 1)
        foreground_img[:, :, 2] = np.random.uniform(0, 1)

        # combine both as reference image
        reference = combine(target, foreground_img, background_img)

        # add different background for positive sample
        new_background_img = (
            load_img(np.random.choice(self.background_images), size=image.shape)
            if len(self.background_images) != 0
            else np.zeros(image.shape) + np.random.uniform(0, 1)
        )
        # new_background_img = np.zeros(image.shape) + np.random.uniform(0, 1)
        positive = combine(target, foreground_img, new_background_img)

        # get different line with different background for negative sample
        random_other_index = index
        # make sure new index is at least 5 frames away
        while abs(random_other_index - index) < 5:
            random_other_index = np.random.randint(0, self._size)

        new_image = self.observations[random_other_index]
        negative = combine(
            get_binary_mask(new_image, gaussian_blur=True),
            foreground_img,
            background_img,
        )
        return {
            "target": torch.from_numpy(target).permute(2, 0, 1).float(),
            "reference": torch.from_numpy(reference).permute(2, 0, 1).float(),
            "positive": torch.from_numpy(positive).permute(2, 0, 1).float(),
            "negative": torch.from_numpy(negative).permute(2, 0, 1).float(),
        }
