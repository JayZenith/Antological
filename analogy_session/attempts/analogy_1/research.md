# analogy_1

## 1. Abstract problem
A shared device already performs one set of services. It must acquire a second service using the same internal parts. Adjustments that help the new service can damage the old service. The old device is available as a reference, and the new cases have a known reversible change of orientation. Preserve old responses while improving new responses, without changing the device size, schedule, or adjustment machinery.

## 2. Distant analogy
Art conservation: retain an untouched reference panel while restoring a working panel. A conservator must improve a damaged working surface while retaining its established appearance. An untouched panel provides a stable reference that does not drift during restoration.

## 3. Concrete mechanism
Reference-panel conservation: compare the working surface against the protected reference on corresponding patches and penalize departures while performing the restoration. This is a structural analogy, not a claim about a universal restoration protocol.

## 4. Translation and precommitted settings
For every Task-B batch, train on its labels and preserve the frozen Task-A model's probability distribution on the same images rotated back to their original orientation, using temperature 2 and a temperature-squared KL penalty with coefficient 1.

Create a frozen deep copy of the initial model, used only during training. Each fixed Task-B batch (x,y) contributes cross_entropy(model(x),y). Construct original-view images as rot90(x,-1). Add 4 * KL(teacher probabilities || student probabilities) on those images with logits divided by 2, reduction batchmean. Use all examples in each supplied batch, including the final partial batch. Sum both terms, backpropagate once, and take one step with the supplied Adam. Teacher parameters never receive gradients. No additional data loader, stochastic sampling, replay buffer, or parameter selection is used. Train exactly EPOCHS=5, keeping the supplied order, seed, architecture, optimizer, learning rate, and batch size. The old-view inputs are obtained from training inputs only; no test examples or test labels inform the procedure.

The reference constrains responses rather than requiring all internal parts to remain unchanged. It may help both services coexist but could constrain acquisition of the new service. Coefficients are fixed before execution and will not be tuned. Evaluate once through evaluate_attempt(ctx) after the fifth epoch, save the final student state dictionary and all required provenance. Training loss summaries are descriptive only and never control the method.

## Metadata (verbatim values used in metrics.json)
```json
{
  "attempt_id": "analogy_1",
  "condition": "analogy",
  "mechanism_name": "Reference-panel conservation",
  "source_analogy": "Art conservation: retain an untouched reference panel while restoring a working panel.",
  "translation": "For every Task-B batch, train on its labels and preserve the frozen Task-A model's probability distribution on the same images rotated back to their original orientation, using temperature 2 and a temperature-squared KL penalty with coefficient 1."
}
```
