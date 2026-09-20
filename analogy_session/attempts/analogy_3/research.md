# analogy_3

## 1. Abstract problem
A shared configuration must move toward a new objective while protecting an established service. A proposed movement can contain both useful and harmful components. A local measure of harm supplies a direction to avoid. Retain the parts of movement parallel to the local boundary and remove the component that crosses into harm, while leaving the movement generator intact.

## 2. Distant analogy
Canal navigation: a rigid bank removes the bank-directed component of a vessel's attempted movement while permitting tangential movement. This is an idealized frictionless contact model: the constraint reacts to attempted penetration, while motion along the bank remains possible.

## 3. Concrete mechanism
Canal-wall deflection: project an unsafe proposed displacement onto the tangent plane of a local safety boundary. Safe displacements are accepted unchanged. This mechanism concerns direction, not trial-and-error search for a shorter step.

## 4. Translation and precommitted settings
At each Task-B batch, compute the original-view labeled loss gradient, obtain the supplied Adam displacement from Task-B cross-entropy, and remove its component along that gradient only when the component would increase original-view loss to first order.

For each fixed Task-B batch (x,y), recover its original orientation with rot90(x,-1). Before updating, compute g, the gradient across all model parameters of cross_entropy(model(original_x), y). The same training labels are valid because the supplied rotation preserves labels. Use autograd.grad, detach the resulting gradient tensors, and retain the starting parameter tensors. Independently compute cross_entropy(model(x),y), backpropagate, and call the supplied Adam step once. Let d be the actual full concatenated parameter displacement after that step, including Adam's moment-based rescaling. Compute s=sum_i dot(g_i,d_i) and q=sum_i dot(g_i,g_i). If s>0 and q>0, replace the resulting parameters by starting_parameters + d - (s/q)*g; otherwise keep the Adam result. Apply this single projection jointly across all tensors, not separate layerwise projections. Do not alter Adam's moments, learning rate, or step counter. Use native float32 calculations. There is no tolerance, backtracking, coefficient tuning, extra optimizer step, or search; if q is zero, skip correction.

Train exactly EPOCHS=5, keeping all supplied fixed components and Task-B order. Each batch provides one new-task update and one local old-view constraint. No additional data loader or test data are used. The safeguard is first-order and local to the current training batch; curvature and other examples can still permit forgetting. It can restrict acquisition of the new service. Record training losses and projection counts for description only, then evaluate once with evaluate_attempt(ctx), save the final model and the required provenance.

## Metadata (verbatim values used in metrics.json)
```json
{
  "attempt_id": "analogy_3",
  "condition": "analogy",
  "mechanism_name": "Canal-wall deflection",
  "source_analogy": "Canal navigation: a rigid bank removes the bank-directed component of a vessel's attempted movement while permitting tangential movement.",
  "translation": "At each Task-B batch, compute the original-view labeled loss gradient, obtain the supplied Adam displacement from Task-B cross-entropy, and remove its component along that gradient only when the component would increase original-view loss to first order."
}
```
