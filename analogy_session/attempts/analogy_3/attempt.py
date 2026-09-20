import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_ROOT))

from experiment import EPOCHS, device_from_arg, evaluate_attempt, new_attempt

import copy
import hashlib
import json
import torch
import torch.nn.functional as F

OUTPUT = Path(__file__).resolve().parent

METADATA = {
  "attempt_id": "analogy_3",
  "condition": "analogy",
  "mechanism_name": "Canal-wall deflection",
  "source_analogy": "Canal navigation: a rigid bank removes the bank-directed component of a vessel's attempted movement while permitting tangential movement.",
  "translation": "At each Task-B batch, compute the original-view labeled loss gradient, obtain the supplied Adam displacement from Task-B cross-entropy, and remove its component along that gradient only when the component would increase original-view loss to first order."
}

if not torch.cuda.is_available():
    raise RuntimeError("The fixed environment requires CUDA.")
ctx = new_attempt(device_from_arg("cuda"))
history = []

parameters = list(ctx.model.parameters())
for epoch in range(EPOCHS):
    ctx.model.train()
    task_sum = original_sum = 0.0
    seen = steps = corrections = 0
    for images, labels in ctx.task_b_train_loader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)
        ctx.optimizer.zero_grad(set_to_none=True)
        original_loss = F.cross_entropy(
            ctx.model(torch.rot90(images, -1, dims=(-2, -1))), labels)
        guard_gradients = [g.detach() for g in torch.autograd.grad(original_loss, parameters)]
        initial = [p.detach().clone() for p in parameters]
        loss = F.cross_entropy(ctx.model(images), labels)
        loss.backward()
        ctx.optimizer.step()
        with torch.no_grad():
            displacements = [p - old for p, old in zip(parameters, initial)]
            component = sum((g * d).sum() for g, d in zip(guard_gradients, displacements))
            norm_squared = sum((g * g).sum() for g in guard_gradients)
            if component.item() > 0.0 and norm_squared.item() > 0.0:
                scale = component / norm_squared
                for p, old, d, g in zip(parameters, initial, displacements, guard_gradients):
                    p.copy_(old + d - scale * g)
                corrections += 1
        count = labels.numel()
        seen += count
        steps += 1
        task_sum += loss.item() * count
        original_sum += original_loss.item() * count
    record = {"epoch": epoch + 1, "examples": seen, "steps": steps,
              "loss": task_sum / seen, "original_view_loss": original_sum / seen,
              "projected_steps": corrections}
    history.append(record)
    print(json.dumps(record), flush=True)

torch.save(ctx.model.state_dict(), OUTPUT / "model.pt")
metrics = evaluate_attempt(ctx)
metrics.update(METADATA)
metrics["epochs"] = EPOCHS
metrics["history"] = history
metrics["attempt_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
metrics["model_sha256"] = hashlib.sha256((OUTPUT / "model.pt").read_bytes()).hexdigest()
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
