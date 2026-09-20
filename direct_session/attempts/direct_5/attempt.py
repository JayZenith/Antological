import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_ROOT))

from experiment import EPOCHS, device_from_arg, evaluate_attempt, new_attempt

import copy
import json
import torch
import torch.nn.functional as F

ATTEMPT_ID = "direct_5"
MECHANISM = "Original-view hidden-feature anchoring"
OUT = Path(__file__).resolve().parent


def hidden_features(model, images):
    flat = model.net[0](images)
    first = model.net[2](model.net[1](flat))
    second = model.net[4](model.net[3](first))
    return first, second


def main():
    ctx = new_attempt(device_from_arg("cuda"))
    teacher = copy.deepcopy(ctx.model).eval()
    teacher.requires_grad_(False)
    print(f"{ATTEMPT_ID}: {MECHANISM}", flush=True)
    for epoch in range(EPOCHS):
        ctx.model.train()
        steps = 0
        examples = 0
        for images, labels in ctx.task_b_train_loader:
            images, labels = images.to(ctx.device), labels.to(ctx.device)
            ctx.optimizer.zero_grad(set_to_none=True)
            loss_b = F.cross_entropy(ctx.model(images), labels)
            original = torch.rot90(images, -1, dims=(-2, -1))
            with torch.no_grad():
                target_first, target_second = hidden_features(teacher, original)
            first, second = hidden_features(ctx.model, original)
            loss = loss_b + F.mse_loss(first, target_first) + F.mse_loss(second, target_second)
            loss.backward()
            ctx.optimizer.step()
            steps += 1
            examples += labels.numel()
        print(f"Epoch {epoch + 1}/{EPOCHS} complete: {steps} updates, {examples} Task-B examples", flush=True)
    torch.save(ctx.model.state_dict(), OUT / "model.pt")
    metrics = evaluate_attempt(ctx)
    metrics.update(attempt_id=ATTEMPT_ID, condition="direct", mechanism=MECHANISM)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print("Final model saved and locked evaluation complete; metrics saved.", flush=True)


if __name__ == "__main__":
    main()
