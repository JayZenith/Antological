# Research response: direct_2

Mechanism: Original-view teacher distillation

Keep a frozen copy of the supplied Task-A model as a teacher. During Task-B learning, constrain the trainable model to retain that teacher's class probabilities on the original orientation of the current training inputs. Construct those original views with the known inverse of Task B's fixed 90-degree rotation. Only training inputs participate in this constraint; no Task-A hard labels or test examples are used in the auxiliary objective.

For every batch from the unchanged Task-B loader, use ordinary cross-entropy on the Task-B images. On the inverse-rotated images, compute teacher probabilities and student log-probabilities at temperature T = 2. The auxiliary objective is T squared times KL(teacher || student), summed over classes and averaged over examples. The total loss is Task-B cross-entropy plus this auxiliary objective with coefficient 1.0. Backpropagate only through the student and perform one provided Adam step per Task-B batch.

Precommitted settings: temperature 2.0, distillation coefficient 1.0, inverse rotation exactly -90 degrees, and a frozen teacher copied once immediately after new_attempt(device). The teacher remains in evaluation mode. Train the student for exactly EPOCHS = 5 complete passes of the existing Task-B loader, with the existing final short batches, Adam optimizer, learning rate 1e-3, batch size 128, architecture, checkpoint, seeds, and example order. Do not interpolate checkpoints, schedule the loss weights, tune settings, or stop early.

The intended benefit is retaining the starting model's decision function on its original input distribution while fitting rotated inputs. Distillation also preserves relative probabilities between classes. Costs include an additional frozen model and two auxiliary forward passes; the teacher can constrain corrections to its own original errors.

Evaluate the final student exactly once using evaluate_attempt(ctx). Save the final student state, verbatim evaluator output augmented with attempt_id, condition, and this exact mechanism name, the exact executed source, and the unedited run log. Evaluation values will not be used to change the method or develop later attempts.
