import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import cerebras.pytorch as cstorch
from cerebras.modelzoo import Trainer
from cerebras.modelzoo.trainer.callbacks import TrainingLoop
from cerebras.modelzoo.trainer.loggers import (
    ProgressLogger,
    TensorBoardLogger,
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch):
        x, y = batch
        prediction = self.network(x)
        return F.mse_loss(prediction, y)


def create_dataloader(batch_size: int):
    generator = torch.Generator().manual_seed(42)

    x = torch.randn(1024, 2, generator=generator)
    y = 2 * x[:, :1] - x[:, 1:2]

    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def main(device: str):
    train_loader = cstorch.utils.data.DataLoader(
        create_dataloader,
        batch_size=64,
    )

    trainer = Trainer(
        device=device.upper(),
        model_dir=f"./runs/{device}",
        model=MLP,
        optimizer=lambda model: cstorch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
        loop=TrainingLoop(num_steps=32),
        loggers=[
            ProgressLogger(),
            TensorBoardLogger(),
        ],
        seed=42,
    )

    trainer.fit(train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "csx"],
        default="cpu",
    )

    args = parser.parse_args()
    main(args.device)
