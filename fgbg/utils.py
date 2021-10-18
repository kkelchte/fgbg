from datetime import datetime
import os

import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio


def create_trajectory_gif(filename: str, data: list):
    imageio.mimsave(filename, data)


def get_IoU(predictions, labels):
    """Code inspired by
    https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    """
    eps = 1e-6
    # assert N x H x W
    if len(predictions.shape) != 3:
        predictions.unsqueeze_(0)
    if len(labels.shape) != 3:
        labels.unsqueeze_(0)
    outputs = predictions.round().int()
    labels = labels.int()
    # Will be zero if Truth=0 or Prediction=0
    intersection = (outputs & labels).float().sum((1, 2))
    # Will be zero if both are
    union = (outputs | labels).float().sum((1, 2))
    # We smooth our devision to avoid 0/0
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def normalize(d: np.ndarray):
    d_min = np.amin(d)
    d_max = np.amax(d)
    return (d - d_min) / (d_max)


def combine_mask_observation(mask: np.array, observation: np.array) -> np.array:
    mask = np.stack([mask + 0.3] * 3, axis=-1)
    mask = np.clip(mask, 0, 1)
    if mask.shape != observation.shape:
        if len(mask.shape) == 4:
            mask = np.stack(
                [cv2.resize(m, observation.shape[1:-1]) for m in mask], axis=0
            )
        else:
            mask = cv2.resize(mask, observation.shape[:-1])
    combination = observation * mask
    return combination


def plot(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def plot_data_item(data_item):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(data_item["reference"].permute(1, 2, 0))
    axes[0].set_title("reference")
    axes[0].axis("off")
    axes[1].imshow(data_item["positive"].permute(1, 2, 0))
    axes[1].set_title("positive")
    axes[1].axis("off")
    axes[2].imshow(data_item["negative"].permute(1, 2, 0))
    axes[2].set_title("negative")
    axes[2].axis("off")
    plt.show()


def get_binary_mask(image: np.ndarray, gaussian_blur: bool = False) -> np.ndarray:
    mask = cv2.threshold(image.mean(axis=-1), 0.5, 1, cv2.THRESH_BINARY_INV)[1]
    if gaussian_blur:
        # select random odd kernel size
        kernel_size = int(np.random.choice([1, 3, 5, 7, 9]))
        sigma = np.random.uniform(0.1, 3)  # select deviation
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return np.expand_dims(mask, axis=-1)


def create_random_gradient_image(
    size: tuple, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    num_channels = size[2]
    # select start and end color gradient location
    locations = (np.random.randint(size[0]), np.random.randint(size[0]))
    while abs(locations[0] - locations[1]) < 10:
        locations = (np.random.randint(size[0]), np.random.randint(size[0]))
    x1, x2 = min(locations), max(locations)
    # for each channel, select an intensity
    channels = []
    for _ in range(num_channels):
        color_a = np.random.uniform(low, high)
        color_b = np.random.uniform(low, high)
        gradient = np.arange(color_a, color_b, ((color_b - color_a) / (x2 - x1)))
        gradient = gradient[: x2 - x1]
        assert len(gradient) == x2 - x1, f"failed: {len(gradient)} vs {x2 - x1}"
        vertical_gradient = np.concatenate(
            [np.ones(x1) * color_a, gradient, np.ones(size[0] - x2) * color_b]
        )
        channels.append(np.stack([vertical_gradient] * size[1], axis=1))
    image = np.stack(channels, axis=-1)
    return image


def get_date_time_tag() -> str:
    return str(datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S"))


def draw_trajectory(filename, goal: list, trajectory: list) -> None:
    """
    filename: path to jpg file
    goal: list of 3 coordinates of goal
    trajectory: list of lists with drone coordinates
    """
    three_d = False
    fig = plt.figure(figsize=(10, 10))
    if three_d:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            trajectory[0][0],
            trajectory[0][1],
            trajectory[0][2],
            s=200,
            color="green",
            marker="v",
            label="Start",
        )
        ax.scatter(
            goal[0], goal[1], goal[2], s=100, color="red", marker="X", label="Goal"
        )
        ax.scatter(
            [_[0] for _ in trajectory[1:]],
            [_[1] for _ in trajectory[1:]],
            [_[2] for _ in trajectory[1:]],
            s=20,
            color="blue",
            marker="o",
            label="Path",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
    else:
        plt.scatter(
            trajectory[0][0],
            trajectory[0][1],
            s=200,
            color="green",
            marker="v",
            label="Start",
        )
        plt.scatter(goal[0], goal[1], s=100, color="red", marker="X", label="Goal")
        plt.scatter(
            [_[0] for _ in trajectory[1:]],
            [_[1] for _ in trajectory[1:]],
            s=20,
            color="blue",
            marker="o",
            label="Path",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    plt.savefig(os.path.join(filename))
    plt.clf()
