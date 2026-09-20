# Research response: direct_4

Mechanism: Checkpoint-centered quadratic penalty

Preserve Task-A knowledge by making movement away from the supplied checkpoint costly during Task-B training. Copy every initial model parameter into a fixed detached anchor immediately after new_attempt(device). Train on the supplied Task-B batches with mean cross-entropy plus a quadratic penalty on all weight and bias displacements from those anchors.

The exact objective is loss = CE(model(x_B), y) + (lambda / 2) * sum_over_all_parameter_elements((theta - theta_A)^2), with lambda = 0.01. The squared displacement is a global sum, not a parameter-wise or layer-wise mean. Include every weight and bias. The anchors never change and have no gradients. Use one backward pass and one step of the supplied Adam optimizer per Task-B batch; do not alter Adam's weight decay or other options.

Precommitted settings: lambda 0.01 for every update; the final model after exactly EPOCHS = 5 complete epochs; no schedule, parameter importance weighting, replay, teacher, layer freezing, checkpoint interpolation, or intermediate selection. Preserve the fixed model architecture, Task-A checkpoint, Adam optimizer, learning rate 1e-3, batch size 128, final short batches, seeds, and Task-B loader order. Only Task-B training examples enter the objective.

The intended benefit is limiting destructive parameter drift while allowing all layers to adapt to rotated inputs. The penalty is a coarse approximation to preserving function: some important directions may remain insufficiently protected, while harmless directions may be overconstrained. The fixed coefficient is selected before any evaluation and will not be tuned.

After training, save the final model state and evaluate once with evaluate_attempt(ctx). Save its unrounded metrics and returned hashes without modification, augmented with attempt_id, condition, and this exact mechanism name. Preserve the executed source and complete unedited run log. Do not use metrics to revise the attempt or develop subsequent methods.
