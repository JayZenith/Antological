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
  "attempt_id": "analogy_2",
  "condition": "analogy",
  "mechanism_name": "Protected structural connections",
  "source_analogy": "Building renovation: preserve designated load-bearing connections while adapting the remaining structure for a new use.",
  "translation": "Within each parameter tensor, protect the half of entries with largest initial absolute magnitude, breaking ties by flattened index; train Task B using cross-entropy while masking protected gradients and restoring protected entries after every supplied Adam step."
}

if not torch.cuda.is_available():
    raise RuntimeError("The fixed environment requires CUDA.")
ctx = new_attempt(device_from_arg("cuda"))
history = []

protected = []
for name, parameter in ctx.model.named_parameters():
    initial = parameter.detach().clone()
    flat = initial.abs().flatten()
    count = (flat.numel() + 1) // 2
    indices = torch.argsort(flat, descending=True, stable=True)[:count]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    protected.append((name, parameter, initial, mask.reshape_as(parameter)))
for epoch in range(EPOCHS):
    ctx.model.train()
    loss_sum = 0.0
    seen = steps = 0
    for images, labels in ctx.task_b_train_loader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ctx.model(images), labels)
        loss.backward()
        for _, parameter, _, mask in protected:
            parameter.grad.masked_fill_(mask, 0.0)
        ctx.optimizer.step()
        with torch.no_grad():
            for _, parameter, initial, mask in protected:
                parameter.copy_(torch.where(mask, initial, parameter))
        count = labels.numel()
        seen += count
        steps += 1
        loss_sum += loss.item() * count
    for _, parameter, initial, mask in protected:
        assert torch.equal(parameter.detach()[mask], initial[mask])
    record = {"epoch": epoch + 1, "examples": seen, "steps": steps,
              "loss": loss_sum / seen,
              "protected_counts": {name: int(mask.sum()) for name, _, _, mask in protected},
              "protected_invariant": True}
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
