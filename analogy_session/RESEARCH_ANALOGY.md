# Distant-analogy-search condition

Use this file in a new model session separate from the direct-search session.
That session must not read baseline or direct metrics, research artifacts, or
mechanisms. Within this one session, complete exactly five attempts in order:
`analogy_1` through `analogy_5`.

## Instruction-level access boundary

Read only this `RESEARCH_ANALOGY.md`, the shared root `experiment.py` exposed
in this directory as `experiment.py`, and files you create inside this
`analogy_session/` directory. Do not list, open, search, or inspect any other
repository path. In particular, do not read the baseline implementation, the
direct-session prompt or artifacts, root comparison/report files, or Git
history. This is an instruction-level isolation boundary, not a filesystem
sandbox.

Do not generate all five methods in advance. When an attempt is reached,
develop exactly one analogy and one resulting mechanism, save it, implement
it, run it once, and finish all artifacts before beginning the next attempt.
Do not tune or replace a method after seeing its metrics, and do not inspect
previous attempt metrics when developing a later attempt.

## Fixed setup

Do not modify `experiment.py`. MNIST splits, rotation, architecture, Task-A
checkpoint, Adam optimizer, learning rate `1e-3`, batch size `128`, five
Task-B epochs, seeds, data order, and evaluation are fixed. Only the Task-B
learning procedure may change.

## Prompt for each attempt

> Improve retention of Task A while learning Task B.
>
> Before proposing an ML solution:
>
> 1. Abstract the problem into its essential entities, relationships,
>    constraints, and objective without using ML-specific terminology.
> 2. Find a structurally analogous problem in a distant non-ML domain.
> 3. Identify a concrete mechanism used in that domain to solve the analogous
>    problem.
> 4. Translate that mechanism back into a Task B training procedure.
> 5. Implement and evaluate the resulting method.
>
> Do not begin by searching for known continual-learning algorithms.
>
> You have one attempt. Choose one distant analogy and one mechanism derived
> from it.
>
> Train for exactly 5 Task B epochs using the fixed optimizer, learning rate,
> batch size, model architecture, Task A checkpoint, datasets, and evaluation
> procedure.
>
> Do not modify those fixed components.

## Required artifacts

For attempt `n`, first save the complete research response verbatim at
`attempts/analogy_n/research.md`. Then implement it in
`attempts/analogy_n/attempt.py` using `new_attempt(device)` and exactly
`EPOCHS` Task-B epochs. Evaluate only through `evaluate_attempt(ctx)`.

Save the exact executed source, final `model.pt`, unedited `run.log`, and
unrounded `metrics.json`. Add `attempt_id`, `condition: "analogy"`, the
mechanism name, source analogy, and translation exactly as written in the
research response. Preserve the checkpoint and infrastructure hashes returned
by the evaluator.

Correcting a software defect is allowed, but changing the proposed mechanism
or its precommitted settings is not.
