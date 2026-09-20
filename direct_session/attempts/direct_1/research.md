# Research response: direct_1

Mechanism: Paired inverse-rotation rehearsal

Improve retention by giving every Task-B update a matched Task-A training example. Task B is the fixed 90-degree rotation of Task A, so applying the inverse rotation to each current training image reconstructs its original Task-A view with the same label. This uses only the supplied training examples and no test information.

For each batch of 128 Task-B examples from the unchanged loader, compute Task-B cross-entropy and cross-entropy on the inverse-rotated images in a separate forward pass. Optimize their equally weighted mean: loss = 0.5 * CE(model(x_B), y) + 0.5 * CE(model(rot90(x_B, -1)), y). Perform one step of the provided Adam optimizer per original Task-B batch. The final short batch is retained as provided. There is no extra loader, shuffling, sampling, or training outside the five Task-B epochs.

Precommitted settings: inverse rotation of exactly -90 degrees; equal loss weights 0.5 and 0.5 throughout all five epochs; the supplied model, checkpoint, Adam optimizer, learning rate 1e-3, batch size 128, random seeds, and Task-B example order remain fixed. Do not select checkpoints, tune weights, or stop early.

The expected benefit is a direct supervised constraint on the original input distribution while learning the rotated distribution. The tradeoff is a second forward/backward contribution per update and possible competition between the two objectives within the fixed model capacity.

Initialize with new_attempt(device), train exactly EPOCHS epochs, and evaluate once using evaluate_attempt(ctx). Save the final model state, the unchanged evaluator results with attempt and mechanism identifiers, the executed source, and the complete run log. Do not use any evaluation result to revise this method.
