# analogy_2

## 1. Abstract problem
One structure supports an established service. A new service requires changing some of its connections, but the original service must remain usable. The size of the structure and the available adjustment process are fixed. Partition its connections into a protected core and an adaptable remainder, then make the new service work through the adaptable portion.

## 2. Distant analogy
Building renovation: preserve designated load-bearing connections while adapting the remaining structure for a new use. Renovating a building does not require every connection to be editable: selected structural elements are retained while other components are adapted.

## 3. Concrete mechanism
Protected structural connections: designate protected connections before construction begins and prevent construction updates from moving them. Translate the idea of substantial structural connections into a deliberately simple magnitude-based proxy. Magnitude does not establish actual functional importance; that limitation is part of this fixed hypothesis.

## 4. Translation and precommitted settings
Within each parameter tensor, protect the half of entries with largest initial absolute magnitude, breaking ties by flattened index; train Task B using cross-entropy while masking protected gradients and restoring protected entries after every supplied Adam step.

After new_attempt(device), clone initial parameters without modifying them. For every parameter tensor, including all weight and bias tensors, protect ceil(numel/2) entries with largest absolute value. Obtain deterministic ties using a stable descending sort of flattened magnitudes. Unprotected entries remain freely trainable. For every supplied Task-B batch, compute only ordinary cross-entropy on its labels, backpropagate, set protected gradient entries to zero, and take exactly one supplied Adam step. Restore protected values from the saved initial tensor after that step as an exact invariant. Adam hyperparameters and state handling otherwise remain unchanged; protected entries have zero gradients from the first step. Do not rescale the remaining gradients. Do not access additional training examples, construct a reference model, or change data order.

Train exactly EPOCHS=5 with the provided seed, architecture, optimizer, learning rate, batch size and checkpoint. No calibration, hyperparameter search, mask revisions, early stopping, or metric-driven decisions are allowed. Freezing half the entries may restrict new-task acquisition and may fail to retain old behavior because editable entries still influence every response. Evaluate once through evaluate_attempt(ctx) after training. Save the final state dictionary and all required artifacts; record protected counts and verify protected entries stayed equal to their initial values.

## Metadata (verbatim values used in metrics.json)
```json
{
  "attempt_id": "analogy_2",
  "condition": "analogy",
  "mechanism_name": "Protected structural connections",
  "source_analogy": "Building renovation: preserve designated load-bearing connections while adapting the remaining structure for a new use.",
  "translation": "Within each parameter tensor, protect the half of entries with largest initial absolute magnitude, breaking ties by flattened index; train Task B using cross-entropy while masking protected gradients and restoring protected entries after every supplied Adam step."
}
```
