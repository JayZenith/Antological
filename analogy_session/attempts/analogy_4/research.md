# analogy_4

## 1. Abstract problem
A shared service system is redirected toward new demand. Its formerly used pathways can lose readiness when they receive no attention. Total system size and the new-demand schedule are fixed. Preserve readiness by reserving a predetermined fraction of maintenance opportunities for a stable collection of old-use cases.

## 2. Distant analogy
Water-network maintenance: periodically flush low-use branches while the main network serves current demand. A routine maintenance schedule deliberately sends activity through branches that current demand might otherwise leave idle.

## 3. Concrete mechanism
Scheduled dormant-branch flushing: preserve a fixed registry of dormant branches and visit them in a deterministic round-robin schedule. The mechanism is periodic exercise, not a redesign of the main network or a diagnosis-driven change in maintenance frequency.

## 4. Translation and precommitted settings
Keep the first 1280 Task-A training examples as ten fixed batches of 128; on every fourth Task-B update, starting with update zero, add one-half of the cross-entropy on the next archive batch and cycle through the ten batches without shuffling.

Load archive indices 0 through 1279 from ctx.task_a_train in ascending order, without any class balancing, replacement, or selection by performance. Store the resulting image and label tensors on the supplied device. Archive batch j uses indices 128*j through 128*j+127. Set global_update=0 and archive_cursor=0 before epoch 1. On every supplied Task-B batch, compute Task-B cross-entropy. When global_update modulo 4 equals zero, additionally compute old-archive cross-entropy and add 0.5 times that value to the Task-B loss. Advance archive_cursor by one modulo 10 only on those scheduled updates. Backpropagate the combined loss once, perform one step with the supplied Adam, and increment global_update. Keep both counters across epoch boundaries.

No other data selection, sampling, teacher, parameter constraints, loss coefficients, or extra optimizer steps are used. Task-B batch order and batch sizes are untouched, and every archive forward also uses exactly 128 examples. Train for exactly EPOCHS=5 with the fixed architecture, checkpoint, seeds and optimization settings. This small archive can omit important old variations and can be overfit; maintenance may also reduce capacity available for new demand. These are limitations of the hypothesis, not grounds to change the fixed settings. Log scheduled archive updates and descriptive losses, evaluate once through evaluate_attempt(ctx), and save all required artifacts and evaluator hashes.

## Metadata (verbatim values used in metrics.json)
```json
{
  "attempt_id": "analogy_4",
  "condition": "analogy",
  "mechanism_name": "Scheduled dormant-branch flushing",
  "source_analogy": "Water-network maintenance: periodically flush low-use branches while the main network serves current demand.",
  "translation": "Keep the first 1280 Task-A training examples as ten fixed batches of 128; on every fourth Task-B update, starting with update zero, add one-half of the cross-entropy on the next archive batch and cycle through the ten batches without shuffling."
}
```
