import os
from typing import Dict

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
    combine_fg_bg,
)


class CleanDataset(TorchDataset):
    def __init__(
        self,
        hdf5_file: str,
        json_file: str,
        fg_augmentation: bool = False,
        input_size: tuple = (3, 200, 200),
        output_size: tuple = (200, 200),
    ):
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
        self.input_size = input_size
        self.output_size = output_size
        self.transforms = [T.Resize(self.input_size[1:])]
        if fg_augmentation:
            self.transforms.extend(
                [
                    T.ColorJitter(
                        brightness=0.1, hue=0.1, saturation=0.1, contrast=0.1
                    ),
                    T.GaussianBlur(kernel_size=(1, 9), sigma=(0.1, 2)),
                ]
            )
        self.transforms = torch.nn.Sequential(*self.transforms)

    def __len__(self) -> int:
        return len(self.hash_index_tuples)

    def load_from_hdf5(self, image) -> torch.Tensor:
        image = torch.as_tensor(np.asarray(image))
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)
        else:
            image.unsqueeze_(0)
        image = self.resize(image)
        return (
            self.augment(image)
            if self.fg_augmentation and image.shape[0] == 3
            else image
        )

    def load_from_file(self, img_path: str) -> torch.Tensor:
        image = np.asarray(Image.open(img_path))
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = np.stack([image.squeeze()] * 3, axis=-1)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.resize(image)
        return self.augment(image) if self.fg_augmentation else image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hsh, sample_index = self.hash_index_tuples[index]
        observation = torch.as_tensor(
            np.asarray(self.hdf5_file[hsh]["observation"][sample_index])
        ).permute(2, 0, 1)
        observation = self.transforms(observation)

        mask = np.asarray(self.hdf5_file[hsh]["mask"][sample_index])
        mask = cv2.resize(
            np.asarray(mask), dsize=self.output_size, interpolation=cv2.INTER_NEAREST
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
        input_size: tuple = (3, 200, 200),
        output_size: tuple = (200, 200),
    ):
        super().__init__(hdf5_file, json_file, fg_augmentation, input_size, output_size)
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

    def combine_fg_bg(
        self, mask: torch.Tensor, foreground: torch.Tensor, background: torch.Tensor
    ) -> torch.Tensor:
        mask = self.resize(mask)
        mask = torch.stack([mask.squeeze()] * 3, axis=0)
        combination = mask * foreground + (1 - mask) * background
        return combination

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hsh, sample_index = self.hash_index_tuples[index]
        result = super().__getitem__(index)

        foreground = np.asarray(self.hdf5_file[hsh]["observation"][sample_index])
        foreground = cv2.resize(
            np.asarray(foreground),
            dsize=self.input_size[1:],
            interpolation=cv2.INTER_LANCZOS4,
        )

        # select background map
        background_img = load_img(
            np.random.choice(self._background_images), size=foreground.shape
        )

        # combine both as reference image
        result["reference"] = combine_fg_bg(
            result["mask"].numpy(), foreground, background_img, blur=self._blur
        )
        result["reference"] = self.transforms(result["reference"])
        result["observation"] = result["reference"]

        # add different background for positive sample
        new_background_img = load_img(
            np.random.choice(self._background_images), size=foreground.shape
        )
        result["positive"] = combine_fg_bg(
            result["mask"].numpy(), foreground, new_background_img, blur=self._blur
        )
        result["positive"] = self.transforms(result["positive"])

        # get different line with different background for negative sample
        random_other_index = index
        # make sure new index is at least 10 frames away
        while abs(random_other_index - index) < 10:
            random_other_index = np.random.randint(0, len(self))

        second_hsh, second_sample_index = self.hash_index_tuples[random_other_index]
        second_foreground = np.asarray(
            self.hdf5_file[second_hsh]["observation"][second_sample_index]
        )
        second_foreground = cv2.resize(
            np.asarray(second_foreground),
            dsize=self.input_size[1:],
            interpolation=cv2.INTER_LANCZOS4,
        )
        second_mask = np.asarray(
            self.hdf5_file[second_hsh]["mask"][second_sample_index]
        )
        second_mask = cv2.resize(
            np.asarray(second_mask),
            dsize=self.output_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        result["negative"] = combine_fg_bg(
            second_mask, second_foreground, background_img, blur=self._blur,
        )
        result["negative"] = self.transforms(result["negative"])
        return result


class ImagesDataset(TorchDataset):
    def __init__(
        self,
        dir_name: str,
        target: str,
        input_size: tuple = (3, 200, 200),
        output_size: tuple = (100, 100),
    ) -> None:
        super().__init__()
        self.name = os.path.basename(dir_name)
        self.images = [
            os.path.join(dir_name, f)
            for f in os.listdir(dir_name)
            if f.endswith(".png") and target in f
        ]
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = self.images[index]
        image = Image.open(img_file)
        image = np.array(image.resize(self.input_size[1:]), dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"observation": image}


class ImageSequenceDataset(TorchDataset):
    def __init__(
        self,
        hdf5_file: str,
        input_size: tuple = (3, 200, 200),
        output_size: tuple = (100, 100),
    ) -> None:
        super().__init__()
        self.hdf5_file = h5py.File(hdf5_file, "r", libver="latest", swmr=True)
        self.image_sequences = list(self.hdf5_file.keys())
        self.input_size = input_size
        self.output_size = output_size
        self.transforms = torch.nn.Sequential(T.Resize(self.input_size[1:]))

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, index):
        seq_key = self.image_sequences[index]
        images = np.asarray(self.hdf5_file[seq_key]["observation"])
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        images = self.transforms(images)
        return {"observations": images}
