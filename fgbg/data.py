import os
from typing import Dict

from PIL import Image
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T


class CleanDataset(TorchDataset):
    def __init__(
        self,
        hdf5_file: str,
        json_file: str,
        fg_augmentation: dict = None,
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
        self.resize = torch.nn.Sequential(T.Resize(self.input_size[1:]))
        if fg_augmentation is not None:
            augmentation_transforms = []
            if fg_augmentation["fg_color"] != {}:
                augmentation_transforms.append(
                    T.ColorJitter(
                        brightness=fg_augmentation["fg_color"]["brightness"],
                        hue=fg_augmentation["fg_color"]["hue"],
                        saturation=fg_augmentation["fg_color"]["saturation"],
                        contrast=fg_augmentation["fg_color"]["hue"],
                    )
                )
            if fg_augmentation["fg_blur"] != {}:
                augmentation_transforms.append(
                    T.GaussianBlur(
                        kernel_size=fg_augmentation["fg_blur"]["kernel"],
                        sigma=(
                            fg_augmentation["fg_blur"]["min_sigma"],
                            fg_augmentation["fg_blur"]["max_sigma"],
                        ),
                    ),
                )
            self.augment = torch.nn.Sequential(*augmentation_transforms)
        else:
            self.augment = None

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
            if self.augment is not None and image.shape[0] == 3
            else image
        )

    def load_from_file(self, img_path: str) -> torch.Tensor:
        image = np.asarray(Image.open(img_path))
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = np.stack([image.squeeze()] * 3, axis=-1)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.resize(image)
        return self.augment(image) if self.augment is not None else image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hsh, sample_index = self.hash_index_tuples[index]
        observation = self.load_from_hdf5(
            self.hdf5_file[hsh]["observation"][sample_index]
        )
        mask = self.load_from_hdf5(self.hdf5_file[hsh]["mask"][sample_index]).squeeze()

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
        combined_blur: dict = {},
        fg_augmentation: dict = {},
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
        self._blur = (
            torch.nn.Sequential(
                T.GaussianBlur(
                    kernel_size=combined_blur["kernel"],
                    sigma=(combined_blur["min_sigma"], combined_blur["max_sigma"]),
                )
            )
            if combined_blur != {}
            else None
        )

    def combine_fg_bg(
        self, mask: torch.Tensor, foreground: torch.Tensor, background: torch.Tensor
    ) -> torch.Tensor:
        mask = self.resize(mask.unsqueeze(0))
        mask = torch.stack([mask.squeeze()] * 3, axis=0)
        combination = mask * foreground + (1 - mask) * background
        return self._blur(combination) if self._blur is not None else combination

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(index)

        foreground = result["observation"]
        background_img = self.load_from_file(np.random.choice(self._background_images))
        result["reference"] = self.combine_fg_bg(
            result["mask"], foreground, background_img
        )
        result["observation"] = result["reference"]

        # add different background for positive sample
        new_background_img = self.load_from_file(
            np.random.choice(self._background_images)
        )
        result["positive"] = self.combine_fg_bg(
            result["mask"], foreground, new_background_img
        )

        # get different line with different background for negative sample
        random_other_index = index
        # make sure new index is at least 10 frames away
        while abs(random_other_index - index) < 10:
            random_other_index = np.random.randint(0, len(self))

        second_hsh, second_sample_index = self.hash_index_tuples[random_other_index]
        second_foreground = self.load_from_hdf5(
            self.hdf5_file[second_hsh]["observation"][second_sample_index]
        )
        second_mask = self.load_from_hdf5(
            self.hdf5_file[second_hsh]["mask"][second_sample_index]
        )
        result["negative"] = self.combine_fg_bg(
            second_mask, second_foreground, background_img
        )
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
            if (f.endswith(".png") or f.endswith(".jpg")) and target in f
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


class LabelledImagesDataset(ImagesDataset):
    def __init__(
        self,
        img_dir_name: str,
        target: str,
        mask_dir_name: str,
        input_size: tuple = (3, 200, 200),
        output_size: tuple = (200, 200),
    ) -> None:
        super().__init__(
            dir_name=img_dir_name,
            target=target,
            input_size=input_size,
            output_size=output_size,
        )
        self.mask_dir_name = mask_dir_name

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # get mask
        mask_file = os.path.join(
            self.mask_dir_name,
            os.path.basename(self.images[index]).replace("jpg", "npy"),
        )
        mask = np.load(mask_file)
        mask = torch.from_numpy(mask // 255).float()
        data["mask"] = mask
        return data


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
