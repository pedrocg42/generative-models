from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_size: int = 128,
        hidden_dims: List[int] = [32, 32, 64, 128],
        latent_dim: int = 512,
        **kwargs,
    ) -> None:
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        current_size = self.input_size
        # Build Encoder
        module_list = nn.ModuleList()
        for h_dim in hidden_dims:
            module_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            current_size = current_size // 2
            in_channels = h_dim
        self.encoder = nn.Sequential(*module_list)

        self.intermediate_size = current_size

        self.fc_mu = nn.Linear((current_size**2) * self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear((current_size**2) * self.hidden_dims[-1], self.latent_dim)

        hidden_dims.reverse()

        # Build Decoder
        self.decoder_input = nn.Linear(self.latent_dim, (current_size**2) * self.hidden_dims[-1])
        module_list = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            module_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        module_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding="same"),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*module_list)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Encodes the input by passing through the encoder network
        and returns the latent codes.

        :param input: _description_
        :type input: torch.Tensor
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: List[torch.Tensor]
        """

        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """_summary_

        :param input: _description_
        :type input: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """
        x = self.decoder_input(input)

        # Reshaping back to 2D with the channels and the intermediate size it had after encoder
        x = x.view(-1, self.hidden_dims[-1], self.intermediate_size, self.intermediate_size)

        # Reconstructing image
        x = self.decoder(x)

        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        :param mu: _description_
        :type mu: torch.Tensor
        :param logvar: _description_
        :type logvar: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """_summary_

        :param input: _description_
        :type input: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """

        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)

        return [self.decode(z), mu, logvar]

    def loss_function(
        self,
        input: torch.Tensor,
        outputs: List[torch.Tensor],
        kld_weight: float,
        **kwargs,
    ) -> dict:

        recons = outputs[0]
        mu = outputs[1]
        logvar = outputs[2]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return loss, {"loss": loss.detach(), "Reconstruction_Loss": recons_loss.detach(), "KLD": -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int) -> torch.Tensor:

        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        return self.decode(z)
