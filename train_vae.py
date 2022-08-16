import os

import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchsummary as summary

import config
from data import FFHQDataset
from utils.parse_experiment import parse_experiment


@parse_experiment
def train(
    network: nn.Module,
    optimizer: nn.Module,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    **experiment,
):

    # Create Model
    model = network()
    model.cuda()
    print(model)

    # Create Datasets and DataLoaders
    train_dataset = FFHQDataset(split="train", transform=True)
    val_dataset = FFHQDataset(split="val")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, num_workers=8, pin_memory=True)

    # Configure Training
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Training starts
    best_val_loss = float(1e12)
    best_model_epoch = None
    for epoch in range(num_epochs):

        # Turning on gradient tracking
        model.train(True)

        # Training one batch
        running_loss = 0.0
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):

            # Zero gradient before every batch
            optimizer.zero_grad()

            # Inference
            batch = batch.cuda()
            output = model(batch)

            # Compute loss and gradients
            loss, extra_losses = model.loss_function(batch, output, **experiment)
            loss.backward()

            # Adjust weights
            optimizer.step()

            # Gather data
            running_loss += loss.item()

            # if (i + 1) % 25 == 0:
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix({"train_loss": avg_loss})

        avg_loss = running_loss / (i + 1)

        # Evaluate on validation set
        # We are not training so we turn off the gradient tracking
        model.train(False)

        running_val_loss = 0.0
        pbar = tqdm(val_dataloader)
        for i, batch in enumerate(pbar):

            batch = batch.cuda()
            output = model(batch)

            loss, extra_losses = model.loss_function(batch, output, **experiment)

            running_val_loss += loss.item()

            avg_val_loss = running_val_loss / (i + 1)
            pbar.set_postfix({"val_loss": avg_val_loss})

        avg_val_loss = running_val_loss / (i + 1)

        print(f" > Best validation loss: {best_val_loss}")
        print(f" > Epoch {epoch} validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_epoch = epoch
            patience_iterations = 0

            model_file_path = os.path.join(config.model_path, experiment["experiment"])
            torch.save(model.state_dict(), model_file_path)

            print(f" > New best model found with best validation loss: {avg_val_loss}")
            print(f" > New best model saved in {model_file_path}")

        else:
            patience_iterations += 1
            if patience_iterations >= patience:
                break

    print("Training Finished!")
    print(f" > The Best Model was saved from epoch {best_model_epoch}")
    print(f" > The validation loss from the best model was {best_val_loss}")


if __name__ == "__main__":
    fire.Fire(train)
