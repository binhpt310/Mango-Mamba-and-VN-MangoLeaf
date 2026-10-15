"""Smoke test for trainer gradient accumulation."""

from contextlib import nullcontext
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from train_mango_mamba import train_one_epoch


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 10, bias=False)

    def forward(self, x):
        return {"logits": self.linear(x)}


class Args:
    grad_accum_steps = 4
    clip_grad = None
    log_interval = 99


class Scheduler:
    def __init__(self):
        self.updates = []

    def step_update(self, update):
        self.updates.append(update)


def loss_fn(output, target):
    loss = output["logits"][:, 0].mean()
    return loss, {"loss": loss.item()}


def main():
    model = Model().cuda()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    scheduler = Scheduler()
    loader = [(torch.ones(1, 1).cuda(), torch.zeros(1, dtype=torch.long).cuda()) for _ in range(5)]
    train_one_epoch(0, model, loader, optimizer, loss_fn, Args(), scheduler, lambda *_: nullcontext())
    assert optimizer.steps == 2, optimizer.steps
    assert scheduler.updates == [0, 1], scheduler.updates


if __name__ == "__main__":
    main()
