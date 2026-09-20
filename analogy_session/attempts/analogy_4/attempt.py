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
  "attempt_id": "analogy_4",
  "condition": "analogy",
  "mechanism_name": "Scheduled dormant-branch flushing",
  "source_analogy": "Water-network maintenance: periodically flush low-use branches while the main network serves current demand.",
  "translation": "Keep the first 1280 Task-A training examples as ten fixed batches of 128; on every fourth Task-B update, starting with update zero, add one-half of the cross-entropy on the next archive batch and cycle through the ten batches without shuffling."
}

if not torch.cuda.is_available():
    raise RuntimeError("The fixed environment requires CUDA.")
ctx = new_attempt(device_from_arg("cuda"))
history = []

archive = [ctx.task_a_train[i] for i in range(1280)]
archive_images = torch.stack([x for x, _ in archive]).to(ctx.device)
archive_labels = torch.tensor([y for _, y in archive], dtype=torch.long, device=ctx.device)
global_update = archive_cursor = 0
for epoch in range(EPOCHS):
    ctx.model.train()
    loss_sum = task_sum = archive_sum = 0.0
    seen = steps = archive_updates = 0
    for images, labels in ctx.task_b_train_loader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)
        ctx.optimizer.zero_grad(set_to_none=True)
        task_loss = F.cross_entropy(ctx.model(images), labels)
        loss = task_loss
        if global_update % 4 == 0:
            first = 128 * archive_cursor
            archive_loss = F.cross_entropy(
                ctx.model(archive_images[first:first + 128]),
                archive_labels[first:first + 128])
            loss = loss + 0.5 * archive_loss
            archive_sum += archive_loss.item()
            archive_updates += 1
            archive_cursor = (archive_cursor + 1) % 10
        loss.backward()
        ctx.optimizer.step()
        global_update += 1
        count = labels.numel()
        seen += count
        steps += 1
        loss_sum += loss.item() * count
        task_sum += task_loss.item() * count
    record = {"epoch": epoch + 1, "examples": seen, "steps": steps,
              "loss": loss_sum / seen, "task_loss": task_sum / seen,
              "archive_loss": archive_sum / archive_updates,
              "archive_updates": archive_updates, "archive_cursor": archive_cursor,
              "global_update": global_update}
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
