from datetime import datetime

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_img(img_path: str, size: tuple = (128, 128, 3)) -> np.ndarray:
    data = Image.open(img_path, mode="r")
    data = cv2.resize(
        np.asarray(data), dsize=(size[0], size[1]), interpolation=cv2.INTER_LANCZOS4
    )
    if size[-1] == 1:
        data = data.mean(axis=-1, keepdims=True)
    data = np.array(data).astype(np.float32) / 255.0  # uint8 -> float32
    return data


def combine(
    mask: np.ndarray, foreground: np.ndarray, background: np.ndarray
) -> np.ndarray:
    combination = mask * foreground + (1 - mask) * background
    return combination


def plot(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def get_binary_mask(image: np.ndarray) -> np.ndarray:
    return np.expand_dims(
        cv2.threshold(image.mean(axis=-1), 0.5, 1, cv2.THRESH_BINARY_INV)[1], axis=-1
    )


def create_random_gradient_image(
    size: tuple, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    color_a = np.random.uniform(low, high)
    color_b = np.random.uniform(low, high)
    locations = (np.random.randint(size[0]), np.random.randint(size[0]))
    while locations[0] == locations[1]:
        locations = (np.random.randint(size[0]), np.random.randint(size[0]))
    x1, x2 = min(locations), max(locations)
    gradient = np.arange(color_a, color_b, ((color_b - color_a) / (x2 - x1)))
    gradient = gradient[: x2 - x1]
    assert len(gradient) == x2 - x1, f"failed: {len(gradient)} vs {x2 - x1}"
    vertical_gradient = np.concatenate(
        [np.ones(x1) * color_a, gradient, np.ones(size[0] - x2) * color_b]
    )
    image = np.stack([vertical_gradient] * size[1], axis=1).reshape(size)
    return image


def get_date_time_tag() -> str:
    return str(datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S"))
