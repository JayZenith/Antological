#!/usr/bin/env python3
"""Locked infrastructure only; Task-B methods are created during research."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# CUDA/cuBLAS option for deterministic GPU behavior.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED, BATCH_SIZE, LR, EPOCHS = 20260919, 128, 1e-3, 5


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RotatedDataset(Dataset):
    def __init__(self, source: Dataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    # rotate by 90
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.source[index]
        return torch.rot90(image, 1, dims=(-2, -1)), label

# force random seed
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # prefer deterministic ops
    torch.use_deterministic_algorithms(True)

# --device auto uses CUDA else CPU
def device_from_arg(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)

# create training and test sets
def datasets_fixed() -> tuple[Dataset, Dataset, Dataset, Dataset]:
    transform = transforms.ToTensor()
    train_a = datasets.MNIST(DATA, train=True, download=True, transform=transform)
    test_a = datasets.MNIST(DATA, train=False, download=True, transform=transform)
    return train_a, test_a, RotatedDataset(train_a), RotatedDataset(test_a)

# turn dataset into batches
# shuffle=True shuffles using fixed seed, so training order reproduced
def loader(dataset: Dataset, *, shuffle: bool, seed: int = SEED) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=generator, num_workers=0)

# compute classification accuracy
@torch.no_grad() # dont track gradients during eval
def accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        correct += int((model(images).argmax(1) == labels).sum())
        total += labels.numel()
    return correct / total

# compute hash of file for byte identicality
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# hash saved checkpoint
def checkpoint_sha256() -> str:
    return file_sha256(ARTIFACTS / "task_a_checkpoint.pt")


def train_task_a(device: torch.device) -> None:
    set_seed()
    train_a, test_a, _, _ = datasets_fixed()
    train_loader, test_loader = loader(train_a, shuffle=True), loader(test_a, shuffle=False)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
        score = accuracy(model, test_loader, device)
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(train_a), "accuracy": score})
        print(f"Task A epoch {epoch + 1}/{EPOCHS}: loss={history[-1]['loss']:.6f} acc={score:.4f}", flush=True)
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "task_a_optimizer": optimizer.state_dict(),
                "seed": SEED, "history": history}, ARTIFACTS / "task_a_checkpoint.pt")
    (ARTIFACTS / "task_a_metrics.json").write_text(
        json.dumps({"A_before_B": history[-1]["accuracy"], "history": history}, indent=2) + "\n")

# everythign Task B needs
@dataclass
class AttemptContext:
    model: MLP
    optimizer: torch.optim.Adam
    task_a_train: Dataset
    task_a_test_loader: DataLoader
    task_b_train_loader: DataLoader
    task_b_test_loader: DataLoader
    device: torch.device
    a_before_b: float


def new_attempt(device: torch.device) -> AttemptContext:
    """Create a clean attempt from the exact checkpoint and fixed components."""
    checkpoint_path = ARTIFACTS / "task_a_checkpoint.pt"
    metrics_path = ARTIFACTS / "task_a_metrics.json"
    if not checkpoint_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run `python3 experiment.py train-a` first")
    set_seed()
    train_a, test_a, train_b, test_b = datasets_fixed()
    # creates blank MLP with fixed arch
    model = MLP().to(device)
    # laod saved TaskA weights into it
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # eval loaded task a to compare to original, if differ, something wrong
    test_a_loader = loader(test_a, shuffle=False)
    a_before = accuracy(model, test_a_loader, device)
    recorded = json.loads(metrics_path.read_text())["A_before_B"]
    if a_before != recorded:
        raise RuntimeError(f"Task-A control changed: saved={recorded}, recomputed={a_before}")
    return AttemptContext(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        task_a_train=train_a,
        task_a_test_loader=test_a_loader,
        task_b_train_loader=loader(train_b, shuffle=True),
        task_b_test_loader=loader(test_b, shuffle=False),
        device=device,
        a_before_b=a_before,
    )

# measure resulting model on both
def evaluate_attempt(context: AttemptContext) -> dict[str, float | str]:
    """Run the single locked evaluation used by every attempt."""
    a_after = accuracy(context.model, context.task_a_test_loader, context.device)
    b_after = accuracy(context.model, context.task_b_test_loader, context.device)
    return {
        "A_before_B": context.a_before_b,
        "A_after_B": a_after,
        "B_after_B": b_after,
        "forgetting": context.a_before_b - a_after,
        "checkpoint_sha256": checkpoint_sha256(),
        "infrastructure_sha256": file_sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train-a"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    train_task_a(device_from_arg(args.device))


if __name__ == "__main__":
    main()
