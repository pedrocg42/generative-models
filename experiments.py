# Networks
from models.vae import VAE

# Criterions
from torch.nn import BCELoss

# Optimizers
from torch.optim import Adam


vae = {
    "experiment": "vae",
    "network": VAE,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "patience": 10,
    "kld_weight": 0.00025,
}

dcgan = {
    "experiment": "dcgan",
    "network": VAE,
    "criterion": BCELoss,
    "optimizer_disc": Adam,
    "adam_beta1": 0.5,
    "learning_rate": 3e-4,
    "num_epochs": 500,
    "patience": 10,
    "hidden_dims_gen": [
        {
            "num_channels": 128,
            "kernel_size": 4,
            "stride": 1,
            "padding": 0,
        },
        {
            "num_channels": 64,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 64,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 32,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 32,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
    ],
    "hidden_dims_disc": [
        {
            "num_channels": 32,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 32,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 64,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 64,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "num_channels": 128,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        {
            "kernel_size": 4,
            "stride": 1,
            "padding": 0,
        },
    ],
}
