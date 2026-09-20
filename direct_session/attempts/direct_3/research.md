# Research response: direct_3

Mechanism: Task-A constrained Adam displacement

Use the original-orientation training loss to constrain the actual parameter displacement proposed by Adam on Task B. For each unchanged Task-B batch, reconstruct its Task-A view by rotating the images by -90 degrees and retain the labels. Compute the gradient g_A of cross-entropy on this original view at the current parameters. Separately compute Task-B cross-entropy and perform one step using the supplied Adam optimizer, retaining the parameters from before that step.

Let d be the concatenated proposed parameter displacement, theta_after_Adam - theta_before. If the global inner product g_A dot d is positive and the global squared norm of g_A is nonzero, replace the displacement with d_corrected = d - ((g_A dot d) / (g_A dot g_A)) * g_A. Otherwise retain the proposed displacement. The resulting displacement has nonpositive first-order change in the original-view loss. Apply the correction directly to the parameters after Adam's step. Leave Adam's moment estimates and step counter exactly as produced by its Task-B gradient update.

Precommitted settings: zero allowed first-order increase; a single global projection across every trainable parameter; no epsilon, clipping, extra loss coefficient, margin, or line search. If the original-view gradient norm is exactly zero, skip projection. Use -90-degree inverse rotation and cross-entropy averaged over each provided batch. Train exactly EPOCHS = 5 full Task-B epochs, with one Adam step per batch and no additional optimizer steps. Preserve the supplied architecture, Task-A checkpoint, Adam learning rate 1e-3, batch size 128, final short batches, seeds, and Task-B data order.

This targets interference in the actual adaptive optimizer update, whose direction can differ from its raw gradient. The protection is local and first order: curvature and batch sampling can still cause forgetting, and the constraint may slow acquisition of Task B. It requires an auxiliary backward computation and storage of one gradient and one parameter snapshot per batch.

Initialize through new_attempt(device). After five epochs, save the final model and evaluate once through evaluate_attempt(ctx). Preserve every evaluator field and hash without rounding and add attempt_id, condition, and this exact mechanism name. Save the executed source and unedited log. Do not tune, select intermediate models, or revise this mechanism from any observed metrics.
