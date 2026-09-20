import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_ROOT))

from experiment import EPOCHS, device_from_arg, evaluate_attempt, new_attempt

import json
import torch
import torch.nn.functional as F

ATTEMPT_ID = "direct_4"
MECHANISM = "Checkpoint-centered quadratic penalty"
OUT = Path(__file__).resolve().parent
PENALTY_COEFFICIENT = 0.01


def main():
    ctx = new_attempt(device_from_arg("cuda"))
    parameters = tuple(ctx.model.parameters())
    anchors = tuple(parameter.detach().clone() for parameter in parameters)
    print(f"{ATTEMPT_ID}: {MECHANISM}", flush=True)
    for epoch in range(EPOCHS):
        ctx.model.train()
        steps = 0
        examples = 0
        for images, labels in ctx.task_b_train_loader:
            images, labels = images.to(ctx.device), labels.to(ctx.device)
            ctx.optimizer.zero_grad(set_to_none=True)
            loss_b = F.cross_entropy(ctx.model(images), labels)
            penalty = torch.stack([(parameter - anchor).square().sum() for parameter, anchor in zip(parameters, anchors)]).sum()
            loss = loss_b + 0.5 * PENALTY_COEFFICIENT * penalty
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
