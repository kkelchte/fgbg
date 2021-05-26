from typing import Tuple, OrderedDict

import torch
from torch import nn


class ResEncoder(nn.Module):
    def __init__(
        self, feature_size: int = 256, projected_size: int = 5, input_channels: int = 1
    ):
        super().__init__()
        self.encoding_layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(input_channels, 8, 3, stride=2)),
                    ("bn_1", torch.nn.BatchNorm2d(8)),
                    ("relu1", nn.ReLU()),
                    ("res2", ResidualBlock(8, 16, padding=(1, 1), strides=(2, 1))),
                    ("res3", ResidualBlock(16, 32, padding=(1, 1), strides=(2, 1))),
                    ("res4", ResidualBlock(32, 64, padding=(1, 1), strides=(2, 1))),
                    ("res5", ResidualBlock(64, 128, padding=(1, 1), strides=(2, 1))),
                    ("res6", ResidualBlock(128, 256, padding=(1, 1), strides=(2, 1)),),
                    (
                        "res7",
                        ResidualBlock(
                            256, feature_size, padding=(1, 1), strides=(2, 1)
                        ),
                    ),
                    ("pool", nn.MaxPool2d(2)),
                ]
            )
        )
        self.projection = nn.Sequential(
            OrderedDict(
                [
                    # ("fc1", nn.Linear(feature_size, feature_size)),
                    # ("relu2", nn.ReLU()),
                    ("fc", nn.Linear(feature_size, projected_size)),
                ]
            )
        )

    def forward(self, inputs: torch.Tensor):
        high_dim_feature = self.encoding_layers(inputs).squeeze(-1).squeeze(-1)
        return self.projection(high_dim_feature)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        batch_norm: bool = True,
        activation: torch.nn.ReLU = torch.nn.ReLU(),
        strides: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        kernel_sizes: Tuple[int, int] = (3, 3),
    ):
        """
        Create a residual block
        :param input_channels: number of input channels at input
        :param output_channels: number of input channels at input
        :param batch_norm: bool specifying to use batch norm 2d (True)
        :param activation: specify torch nn module activation (ReLU)
        :param pool: specify pooling layer applied as first layer
        :param strides: tuple specifying the stride and so the down sampling
        """
        super().__init__()
        self._down_sample = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=strides[0],
            bias=False,
        )
        self._final_activation = activation
        elements = []
        elements.append(
            (
                "conv_0",
                torch.nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_sizes[0],
                    padding=padding[0],
                    stride=strides[0],
                    bias=False,
                ),
            )
        )
        if batch_norm:
            elements.append(("bn_0", torch.nn.BatchNorm2d(output_channels)))
        elements.append(("act_0", activation))
        elements.append(
            (
                "conv_1",
                torch.nn.Conv2d(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_sizes[1],
                    padding=padding[1],
                    stride=strides[1],
                    bias=False,
                ),
            )
        )
        if batch_norm:
            elements.append(("bn_1", torch.nn.BatchNorm2d(output_channels)))
        elements.append(("act_1", activation))
        self.residual_net = torch.nn.Sequential(OrderedDict(elements))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.residual_net(inputs)
        x += self._down_sample(inputs)
        return self._final_activation(x)


class Decoder(nn.Module):
    def __init__(self, input_size: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("deconv0", nn.ConvTranspose2d(input_size, 256, 3, stride=2)),
                    ("relu0", nn.ReLU()),
                    ("deconv1", nn.ConvTranspose2d(256, 128, 3, stride=2)),
                    ("relu1", nn.ReLU()),
                    ("deconv2", nn.ConvTranspose2d(128, 64, 3, stride=2)),
                    ("relu2", nn.ReLU()),
                    ("deconv3", nn.ConvTranspose2d(64, 32, 3, stride=2)),
                    ("relu3", nn.ReLU()),
                    ("deconv4", nn.ConvTranspose2d(32, 16, 3, stride=2)),
                    ("relu4", nn.ReLU()),
                    ("deconv5", nn.ConvTranspose2d(16, 8, 3, stride=2)),
                    ("relu5", nn.ReLU()),
                    ("deconv6", nn.ConvTranspose2d(8, 1, 3, stride=1)),
                    ("relu6", nn.ReLU()),
                ]
            )
        )

    def forward(self, inpt: torch.Tensor):
        logits = self.layers(inpt.unsqueeze(-1).unsqueeze(-1))
        return torch.sigmoid(logits)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int = 256,
        projected_size: int = 5,
        input_channels: int = 1,
        decode_from_projection: bool = True,
    ):
        super().__init__()
        self.encoder = ResEncoder(feature_size, projected_size, input_channels)
        self.decoder = Decoder(
            projected_size if decode_from_projection else feature_size
        )
        self.decode_from_projection = decode_from_projection
        if not self.decode_from_projection:
            raise NotImplementedError

    def forward(self, input):
        projection = self.encoder(input)
        return self.decoder(projection)
