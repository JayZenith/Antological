import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_ROOT))

from experiment import EPOCHS, device_from_arg, evaluate_attempt, new_attempt

import json
import torch
import torch.nn.functional as F

ATTEMPT_ID = "direct_3"
MECHANISM = "Task-A constrained Adam displacement"
OUT = Path(__file__).resolve().parent


def main():
    ctx = new_attempt(device_from_arg("cuda"))
    parameters = tuple(ctx.model.parameters())
    print(f"{ATTEMPT_ID}: {MECHANISM}", flush=True)
    for epoch in range(EPOCHS):
        ctx.model.train()
        steps = 0
        examples = 0
        for images, labels in ctx.task_b_train_loader:
            images, labels = images.to(ctx.device), labels.to(ctx.device)
            ctx.optimizer.zero_grad(set_to_none=True)
            original = torch.rot90(images, -1, dims=(-2, -1))
            loss_a = F.cross_entropy(ctx.model(original), labels)
            gradients_a = torch.autograd.grad(loss_a, parameters)
            before = tuple(parameter.detach().clone() for parameter in parameters)
            loss_b = F.cross_entropy(ctx.model(images), labels)
            loss_b.backward()
            ctx.optimizer.step()
            with torch.no_grad():
                dot = torch.stack([(gradient * (parameter - old)).sum() for gradient, parameter, old in zip(gradients_a, parameters, before)]).sum()
                norm_sq = torch.stack([gradient.square().sum() for gradient in gradients_a]).sum()
                if dot.item() > 0.0 and norm_sq.item() > 0.0:
                    coefficient = dot / norm_sq
                    for parameter, gradient in zip(parameters, gradients_a):
                        parameter.sub_(coefficient * gradient)
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
