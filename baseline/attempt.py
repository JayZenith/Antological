#!/usr/bin/env python3
"""Fixed baseline: ordinary Task-B cross-entropy fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import EPOCHS, device_from_arg, evaluate_attempt, new_attempt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = device_from_arg(args.device)
    ctx = new_attempt(device)
    criterion = nn.CrossEntropyLoss()
    log_lines = []

    for epoch in range(EPOCHS):
        ctx.model.train()
        loss_sum = 0.0
        example_count = 0
        for images, labels in ctx.task_b_train_loader:
            images, labels = images.to(device), labels.to(device)
            ctx.optimizer.zero_grad(set_to_none=True)
            loss = criterion(ctx.model(images), labels)
            loss.backward()
            ctx.optimizer.step()
            loss_sum += loss.item() * labels.numel()
            example_count += labels.numel()
        line = f"Task B epoch {epoch + 1}/{EPOCHS}: loss={loss_sum / example_count:.9f}"
        log_lines.append(line)
        print(line, flush=True)

    metrics = evaluate_attempt(ctx)
    metrics.update({
        "attempt_id": "baseline",
        "condition": "baseline",
        "mechanism": "ordinary Task-B cross-entropy training",
    })
    output_dir = Path(__file__).resolve().parent
    torch.save(ctx.model.state_dict(), output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
