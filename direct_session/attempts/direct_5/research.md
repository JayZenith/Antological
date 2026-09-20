# Research response: direct_5

Mechanism: Original-view hidden-feature anchoring

Preserve the internal representations learned for Task A while fitting the Task-B labels. Immediately after new_attempt(device), make a frozen teacher copy of the initial model. For each unchanged Task-B batch, compute the student's ordinary Task-B cross-entropy. In addition, invert the known 90-degree rotation on the batch and compare student and teacher activations after each of the two hidden ReLU layers on these original-orientation training images.

Let h1 and h2 denote the 256-dimensional activations after the first and second hidden ReLUs. The exact objective is loss = CE(student(x_B), y) + mean((h1_student(x_A) - h1_teacher(x_A))^2) + mean((h2_student(x_A) - h2_teacher(x_A))^2). Each mean is over all examples and all 256 hidden coordinates of its layer. The coefficient of each feature penalty is 1.0. Use raw activations without normalization, centering, or temperature. The teacher is frozen in evaluation mode and never updated. The student's full architecture remains unchanged, and all student parameters, including the readout, are trainable.

Precommitted settings: inverse rotation exactly -90 degrees, two post-ReLU feature targets, mean-squared-error coefficient 1.0 per layer, and constant coefficients for all five epochs. Do not add Task-A label loss or output distillation. Perform exactly EPOCHS = 5 complete passes through the existing Task-B loader, with one provided Adam step per batch. Keep the model architecture, checkpoint, optimizer, learning rate 1e-3, batch size 128, final short batches, seeds, and example order fixed. Do not tune settings, select checkpoints, or stop early.

The intended benefit is retaining reusable original-view representations while allowing adaptation on rotated inputs. Unlike an output constraint, the auxiliary loss acts directly on both hidden layers. It can overconstrain representations that could safely change, and it does not directly prevent readout drift. Costs include one frozen model and original-view feature computations for both models.

Save the final student state after training and evaluate only once through evaluate_attempt(ctx). Preserve all unrounded evaluator fields and hashes and add attempt_id, condition, and this exact mechanism name. Save the exact executed implementation and complete unedited log. No evaluation result will be used to revise the mechanism or settings.
