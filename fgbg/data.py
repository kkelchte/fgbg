import os
from typing import Dict
import copy

from PIL import Image
import json
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T

from .utils import (
    load_img,
    combine,
    generate_random_square,
)

IMAGE_SIZE = (200, 200)


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


class CleanDataset(TorchDataset):
    def __init__(self, hdf5_file: str, json_file: str, fg_augmentation: bool = False):
        self.name = (
            os.path.basename(os.path.dirname(os.path.dirname(hdf5_file)))
            + "/"
            + os.path.basename(os.path.dirname(hdf5_file))
        )
        self.hdf5_file = h5py.File(hdf5_file, "r", libver="latest", swmr=True)
        with open(json_file, "r") as f:
            self.json_data = json.load(f)

        self.hash_index_tuples = [
            (h, index)
            for h in list(self.json_data.keys())
            for index in range(len(self.json_data[h]["velocities"]))
        ]
        self.transforms = [T.Resize(IMAGE_SIZE)]
        if fg_augmentation:
            self.transforms.extend([
                T.ColorJitter(brightness=0.1, hue=0.1, saturation=0.1, contrast=0.1),
                T.GaussianBlur(kernel_size=(1, 9), sigma=(0.1, 2)),
            ])
        self.transforms = torch.nn.Sequential(*self.transforms)

    def __len__(self) -> int:
        return len(self.hash_index_tuples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hsh, sample_index = self.hash_index_tuples[index]
        observation = torch.as_tensor(
            np.asarray(self.hdf5_file[hsh]["observation"][sample_index])
        ).permute(2, 0, 1)
        observation = self.transforms(observation)

        mask = np.asarray(self.hdf5_file[hsh]["mask"][sample_index])
        mask = cv2.resize(
            np.asarray(mask), dsize=IMAGE_SIZE, interpolation=cv2.INTER_NEAREST
        )
        mask = torch.from_numpy(mask).float()

        relative_target_location = self.json_data[hsh]["relative_target_location"][
            sample_index
        ]
        relative_target_location = torch.as_tensor(relative_target_location).float()

        velocities = self.json_data[hsh]["velocities"][sample_index]
        velocities = torch.as_tensor(velocities).float()

        return {
            "observation": observation,
            "reference": observation,
            "mask": mask,
            "velocities": velocities,
            "waypoints": relative_target_location,
        }


class AugmentedTripletDataset(CleanDataset):
    def __init__(
        self,
        hdf5_file: str,
        json_file: str,
        background_images_directory: str,
        blur: bool = False,
        fg_augmentation: bool = False,
    ):
        super().__init__(hdf5_file, json_file, fg_augmentation)
        self._background_images = (
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
        self._blur = blur

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hsh, sample_index = self.hash_index_tuples[index]
        result = super().__getitem__(index)

        foreground = np.asarray(self.hdf5_file[hsh]["observation"][sample_index])
        foreground = cv2.resize(
            np.asarray(foreground), dsize=IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4
        )

        # select background map
        background_img = load_img(
            np.random.choice(self._background_images), size=foreground.shape
        )

        # combine both as reference image
        result["reference"] = combine(
            result["mask"].numpy(), foreground, background_img, blur=self._blur
        )
        result["reference"] = self.transforms(result["reference"])
        result["observation"] = result["reference"]

        # add different background for positive sample
        new_background_img = load_img(
            np.random.choice(self._background_images), size=foreground.shape
        )
        # new_background_img = np.zeros(image.shape) + np.random.uniform(0, 1)
        result["positive"] = combine(
            result["mask"].numpy(), foreground, new_background_img, blur=self._blur
        )
        result["positive"] = self.transforms(result["positive"])

        # get different line with different background for negative sample
        random_other_index = index
        # make sure new index is at least 5 frames away
        while abs(random_other_index - index) < 5:
            random_other_index = np.random.randint(0, len(self))

        second_hsh, second_sample_index = self.hash_index_tuples[random_other_index]
        second_foreground = np.asarray(
            self.hdf5_file[second_hsh]["observation"][second_sample_index]
        )
        second_foreground = cv2.resize(
            np.asarray(second_foreground),
            dsize=IMAGE_SIZE,
            interpolation=cv2.INTER_LANCZOS4,
        )
        second_mask = np.asarray(
            self.hdf5_file[second_hsh]["mask"][second_sample_index]
        )
        second_mask = cv2.resize(
            np.asarray(second_mask), dsize=IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4
        )
        result["negative"] = combine(
            second_mask, second_foreground, background_img, blur=self._blur,
        )
        result["negative"] = self.transforms(result["negative"])
        return result


class ImagesDataset(TorchDataset):
    def __init__(self, dir_name: str, target: str) -> None:
        super().__init__()
        self.name = os.path.basename(dir_name)
        self.images = [
            os.path.join(dir_name, f)
            for f in os.listdir(dir_name)
            if f.endswith(".png") and target in f
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = self.images[index]
        image = Image.open(img_file)
        image = np.array(image.resize(IMAGE_SIZE), dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"observation": image}


class ImageSequenceDataset(TorchDataset):
    def __init__(self, hdf5_file: str) -> None:
        super().__init__()
        self.hdf5_file = h5py.File(hdf5_file, "r", libver="latest", swmr=True)
        self.image_sequences = list(self.hdf5_file.keys())

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, index):
        seq_key = self.image_sequences[index]
        images = np.asarray(self.hdf5_file[seq_key]["observation"])
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return {"observations": images}
