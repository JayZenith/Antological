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
  "attempt_id": "analogy_5",
  "condition": "analogy",
  "mechanism_name": "Mooring-radius constraint",
  "source_analogy": "Marine mooring: a vessel can respond to changing demands within the reach of a fixed-length tether anchored at its home position.",
  "translation": "After every supplied Adam update on Task-B cross-entropy, project each parameter tensor onto a Euclidean ball centered at its initial Task-A value with radius equal to 10 percent of that tensor's initial Euclidean norm."
}

if not torch.cuda.is_available():
    raise RuntimeError("The fixed environment requires CUDA.")
ctx = new_attempt(device_from_arg("cuda"))
history = []

moorings = []
for name, parameter in ctx.model.named_parameters():
    anchor = parameter.detach().clone()
    initial_norm = torch.linalg.vector_norm(anchor)
    radius = 0.10 * initial_norm
    moorings.append((name, parameter, anchor, initial_norm, radius))
for epoch in range(EPOCHS):
    ctx.model.train()
    loss_sum = 0.0
    seen = steps = projections = 0
    for images, labels in ctx.task_b_train_loader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ctx.model(images), labels)
        loss.backward()
        ctx.optimizer.step()
        with torch.no_grad():
            for _, parameter, anchor, _, radius in moorings:
                if radius.item() == 0.0:
                    parameter.copy_(anchor)
                    projections += 1
                else:
                    displacement = parameter - anchor
                    distance = torch.linalg.vector_norm(displacement)
                    if distance.item() > radius.item():
                        parameter.copy_(anchor + (radius / distance) * displacement)
                        projections += 1
        count = labels.numel()
        seen += count
        steps += 1
        loss_sum += loss.item() * count
    with torch.no_grad():
        fractions = {
            name: (float(torch.linalg.vector_norm(parameter - anchor) / initial_norm)
                   if initial_norm.item() > 0.0 else 0.0)
            for name, parameter, anchor, initial_norm, _ in moorings
        }
    record = {"epoch": epoch + 1, "examples": seen, "steps": steps,
              "loss": loss_sum / seen, "tensor_projections": projections,
              "displacement_fractions": fractions}
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
