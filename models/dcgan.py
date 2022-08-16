from typing import Dict, List

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_size: int = 128,
        hidden_dims_gen: List[Dict] = None,
        latent_dim: int = 100,
        **kwargs,
    ):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.hidden_dims_gen = hidden_dims_gen
        self.latent_dim = latent_dim

        # Build Generator
        module_list = nn.ModuleList()
        for h_dim_dict in self.hidden_dims_gen[-1]:
            module_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=latent_dim,
                        out_channels=h_dim_dict["num_channels"],
                        kernel_size=h_dim_dict["kernel_size"],
                        stride=h_dim_dict["stride"],
                        padding=h_dim_dict["padding"],
                        bias=False,
                    ),
                    nn.BatchNorm2d(h_dim_dict["num_channels"]),
                    nn.ReLU(True),
                )
            )
            current_size = current_size * 2
            latent_dim = h_dim_dict["num_channels"]

        # Adding output block
        module_list.append(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=self.in_channels,
                kernel_size=self.hidden_dims_gen[-1]["kernel_size"],
                stride=self.hidden_dims_gen[-1]["stride"],
                padding=self.hidden_dims_gen[-1]["padding"],
                bias=False,
            ),
            nn.Tanh(),
        )

        self.generator = nn.Sequential(*module_list)

    def forward(self, input):
        return self.generator(input)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_size: int = 128,
        hidden_dims_disc: List[Dict] = None,
        **kwargs,
    ):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.hidden_dims_disc = hidden_dims_disc

        # Build Discriminator
        module_list = nn.ModuleList()
        for h_dim_dict in self.hidden_dims_disc[:-1]:
            module_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=h_dim_dict["num_channels"],
                        kernel_size=h_dim_dict["kernel_size"],
                        stride=h_dim_dict["stride"],
                        padding=h_dim_dict["padding"],
                        bias=False,
                    ),
                    nn.BatchNorm2d(h_dim_dict["num_channels"]),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            in_channels = h_dim_dict["num_channels"]

        # Adding output block
        module_list.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=self.hidden_dims_disc[-1]["kernel_size"],
                stride=self.hidden_dims_disc[-1]["stride"],
                padding=self.hidden_dims_disc[-1]["padding"],
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.discriminator = nn.Sequential(*module_list)

    def forward(self, input: torch.Tensor):
        return self.discriminator(input)
