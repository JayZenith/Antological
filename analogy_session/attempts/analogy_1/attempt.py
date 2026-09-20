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
  "attempt_id": "analogy_1",
  "condition": "analogy",
  "mechanism_name": "Reference-panel conservation",
  "source_analogy": "Art conservation: retain an untouched reference panel while restoring a working panel.",
  "translation": "For every Task-B batch, train on its labels and preserve the frozen Task-A model's probability distribution on the same images rotated back to their original orientation, using temperature 2 and a temperature-squared KL penalty with coefficient 1."
}

if not torch.cuda.is_available():
    raise RuntimeError("The fixed environment requires CUDA.")
ctx = new_attempt(device_from_arg("cuda"))
history = []

teacher = copy.deepcopy(ctx.model).eval()
teacher.requires_grad_(False)
for epoch in range(EPOCHS):
    ctx.model.train()
    loss_sum = task_sum = reference_sum = 0.0
    seen = steps = 0
    for images, labels in ctx.task_b_train_loader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)
        original_images = torch.rot90(images, -1, dims=(-2, -1))
        ctx.optimizer.zero_grad(set_to_none=True)
        task_loss = F.cross_entropy(ctx.model(images), labels)
        with torch.no_grad():
            reference = F.softmax(teacher(original_images) / 2.0, dim=1)
        preservation = 4.0 * F.kl_div(
            F.log_softmax(ctx.model(original_images) / 2.0, dim=1),
            reference, reduction="batchmean")
        loss = task_loss + preservation
        loss.backward()
        ctx.optimizer.step()
        count = labels.numel()
        seen += count
        steps += 1
        loss_sum += loss.item() * count
        task_sum += task_loss.item() * count
        reference_sum += preservation.item() * count
    record = {"epoch": epoch + 1, "examples": seen, "steps": steps,
              "loss": loss_sum / seen, "task_loss": task_sum / seen,
              "reference_loss": reference_sum / seen}
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
