import os

import imageio
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Module, MSELoss
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset as TorchDataset

from .utils import get_IoU, combine_mask_observation
from .losses import WeightedBinaryCrossEntropyLoss


def evaluate_qualitatively_on_sequences(
    tag: str,
    dataset: TorchDataset,
    model: Module,
    output_directory: str,
    device: str = "cpu",
):
    save_dir = os.path.join(output_directory, "imgs")
    os.makedirs(save_dir, exist_ok=True)
    for _ in range(len(dataset)):
        data = dataset[_]
        prediction = model(data["observations"].to(device), intermediate_outputs=False)
        masks = prediction.detach().cpu().squeeze().numpy()
        obs = data["observations"].detach().cpu().squeeze().permute(0, 2, 3, 1).numpy()
        combined = combine_mask_observation(masks, obs)
        images = list((combined * 255.0).astype(np.uint8))
        imageio.mimsave(os.path.join(save_dir, f"{tag}_{_}.gif"), images)


def evaluate_qualitatively_on_dataset(
    tag: str,
    dataset: TorchDataset,
    model: Module,
    tb_writer: SummaryWriter,
    max_number_of_images: int = 15,
):
    save_dir = os.path.join(tb_writer.get_logdir(), "imgs")
    os.makedirs(save_dir, exist_ok=True)
    images = []
    for _ in range(min(len(dataset), max_number_of_images)):
        data = dataset[_]
        prediction = model(data["observation"].unsqueeze(0), intermediate_outputs=False)
        mask = prediction.detach().cpu().squeeze().numpy()
        obs = data["observation"].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        combined = combine_mask_observation(mask, obs)
        images.append(torch.from_numpy(combined).permute(2, 0, 1))
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(obs)
        ax[0].axis("off")
        ax[1].imshow(mask)
        ax[1].axis("off")
        ax[2].imshow(combined)
        ax[2].axis("off")
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{tag}_{_}.jpg"))
        plt.close(fig=fig)
    grid = torchvision.utils.make_grid(torch.stack(images), nrow=5)
    tb_writer.add_image(tag, grid, dataformats="CHW")


def evaluate_quantitatively_on_dataset(
    tag: str,
    dataset: TorchDataset,
    model: Module,
    tb_writer: SummaryWriter,
    task: str = "pretrain",
):
    if task == "pretrain":
        return evaluate_mask_prediction(tag, dataset, model, tb_writer)
    else:
        return evaluate_downstream_task(tag, dataset, model, tb_writer, task)


def evaluate_downstream_task(
    tag: str,
    dataset: TorchDataset,
    model: Module,
    tb_writer: SummaryWriter,
    task: str = "velocities",
):
    losses = []
    mse_loss = MSELoss()
    for _ in range(min(len(dataset), 100)):
        data = dataset[_]
        prediction = model(data["observation"].unsqueeze(0))
        loss = mse_loss(prediction, data[task].unsqueeze(0))
        losses.append(loss.detach().cpu())
    tb_writer.add_scalar(
        tag + "_mse_loss_avg",
        torch.as_tensor(losses).mean(),
        global_step=model.global_step,
    )
    tb_writer.add_scalar(
        tag + "_mse_loss_std",
        torch.as_tensor(losses).std(),
        global_step=model.global_step,
    )
    with open(os.path.join(tb_writer.get_logdir(), "results.txt"), "a") as f:
        f.write(f"{tag}_mse_loss_avg: " f"{torch.as_tensor(losses).mean():10.3e}\n",)
        f.write(f"{tag}_mse_loss_std: " f"{torch.as_tensor(losses).std():10.2e}\n",)


def evaluate_mask_prediction(
    tag: str, dataset: TorchDataset, model: Module, tb_writer: SummaryWriter,
):
    losses = []
    ious = []
    bce_loss = WeightedBinaryCrossEntropyLoss(beta=0.9)
    for _ in range(min(len(dataset), 100)):
        data = dataset[_]
        prediction = model(data["observation"].unsqueeze(0), intermediate_outputs=False)
        loss = bce_loss(prediction, data["mask"])
        losses.append(loss.detach().cpu())
        ious.append(get_IoU(prediction, data["mask"].unsqueeze(0)).detach().cpu())
    tb_writer.add_scalar(
        tag + "_bce_loss_avg",
        torch.as_tensor(losses).mean(),
        global_step=model.global_step,
    )
    tb_writer.add_scalar(
        tag + "_bce_loss_std",
        torch.as_tensor(losses).std(),
        global_step=model.global_step,
    )
    tb_writer.add_scalar(
        tag + "_iou_avg", torch.as_tensor(ious).mean(), global_step=model.global_step,
    )
    tb_writer.add_scalar(
        tag + "_iou_std", torch.as_tensor(ious).std(), global_step=model.global_step,
    )
    with open(os.path.join(tb_writer.get_logdir(), "results.txt"), "a") as f:
        f.write(f"{tag}_bce_loss_avg: " f"{torch.as_tensor(losses).mean():10.3e}\n",)
        f.write(f"{tag}_bce_loss_std: " f"{torch.as_tensor(losses).std():10.2e}\n",)
        f.write(f"{tag}_ious_avg: " f"{torch.as_tensor(ious).mean():10.3e}\n",)
        f.write(f"{tag}_ious_std: " f"{torch.as_tensor(ious).std():10.2e}\n",)

