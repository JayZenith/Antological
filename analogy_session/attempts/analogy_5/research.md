# analogy_5

## 1. Abstract problem
An existing configuration delivers a useful service and must adjust to new demands. Unrestricted drift can take it far from the starting configuration. Permit limited movement around its initial position while keeping a fixed bound on total displacement. The adjustment mechanism, operating schedule, and system size stay fixed.

## 2. Distant analogy
Marine mooring: a vessel can respond to changing demands within the reach of a fixed-length tether anchored at its home position. An idealized tether permits motion within its reach and exerts a restoring constraint at the boundary. The analogy uses this geometric constraint, not a detailed hydrodynamic simulation.

## 3. Concrete mechanism
Mooring-radius constraint: retain a home anchor and an immutable movement radius. Accept motion within the radius; if a proposed position exceeds it, shorten its displacement along the same direction until it reaches the boundary.

## 4. Translation and precommitted settings
After every supplied Adam update on Task-B cross-entropy, project each parameter tensor onto a Euclidean ball centered at its initial Task-A value with radius equal to 10 percent of that tensor's initial Euclidean norm.

Immediately after new_attempt(device), clone each initial parameter tensor, including all weights and biases. For tensor i, fix anchor a_i and radius r_i=0.10*norm(a_i), where norm is the flattened Euclidean norm. For each supplied Task-B batch, compute standard cross-entropy, backpropagate once, and take exactly one step using the supplied Adam. Under no_grad, let d_i be the updated parameter tensor minus a_i. If norm(d_i)>r_i, replace the parameter with a_i + (r_i/norm(d_i))*d_i. If r_i=0, restore a_i exactly. Otherwise accept the update unchanged. Apply the independent fixed-radius projection to every tensor, with native float32 arithmetic. The anchors and radii never change, and Adam's moments, step count, learning rate, and defaults remain untouched.

Use no additional data, frozen prediction model, reference labels, or change to supplied batch order. Train exactly EPOCHS=5 with all fixed components. The 10 percent radius is precommitted without a calibration run and is never enlarged or reduced based on results. Distance from the old parameters is only a proxy for preserved service: small movements can still change outputs substantially, while the tether can prevent movement necessary for learning the new service. Log training loss, numbers of tensor projections, and end-of-epoch displacement fractions for description only. Evaluate once through evaluate_attempt(ctx), save the final model, exact source, full unedited log, unrounded metrics, metadata, and evaluator hashes.

## Metadata (verbatim values used in metrics.json)
```json
{
  "attempt_id": "analogy_5",
  "condition": "analogy",
  "mechanism_name": "Mooring-radius constraint",
  "source_analogy": "Marine mooring: a vessel can respond to changing demands within the reach of a fixed-length tether anchored at its home position.",
  "translation": "After every supplied Adam update on Task-B cross-entropy, project each parameter tensor onto a Euclidean ball centered at its initial Task-A value with radius equal to 10 percent of that tensor's initial Euclidean norm."
}
```
