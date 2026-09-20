# Direct-search condition

Use this file in one new model session dedicated only to the direct-search
condition. That session must not read baseline metrics, analogy artifacts, or
any prior proposed mechanisms. Within this one session, complete exactly five
attempts in order: `direct_1` through `direct_5`.

## Instruction-level access boundary

Read only this `RESEARCH_DIRECT.md`, the shared root `experiment.py` exposed in
this directory as `experiment.py`, and files you create inside this
`direct_session/` directory. Do not list, open, search, or inspect any other
repository path. In particular, do not read the baseline implementation, the
analogy-session prompt or artifacts, root comparison/report files, or Git
history. This is an instruction-level isolation boundary, not a filesystem
sandbox.

Do not generate all five methods in advance. When an attempt is reached,
develop exactly one approach, save it, implement it, run it once, and finish
all artifacts before beginning the next attempt. Do not tune or replace a
method after seeing its metrics, and do not inspect previous attempt metrics
when developing a later attempt.

## Fixed setup

Do not modify `experiment.py`. MNIST splits, rotation, architecture, Task-A
checkpoint, Adam optimizer, learning rate `1e-3`, batch size `128`, five
Task-B epochs, seeds, data order, and evaluation are fixed. Only the Task-B
learning procedure may change.

## Prompt for each attempt

> Improve retention of Task A while learning Task B.
> You may modify the Task B training procedure.
> Develop one approach, implement it, and evaluate it.
>
> You have one attempt. Train for exactly 5 Task B epochs using the fixed
> optimizer, learning rate, batch size, model architecture, Task A checkpoint,
> datasets, and evaluation procedure.

Do not provide the research context with examples of continual-learning
methods.

## Required artifacts

For attempt `n`, first save the complete research response verbatim at
`attempts/direct_n/research.md`. Then implement it in
`attempts/direct_n/attempt.py` using `new_attempt(device)` and exactly `EPOCHS`
Task-B epochs. Evaluate only through `evaluate_attempt(ctx)`.

Because `attempt.py` is two directories below the session root, it must use
this exact import bootstrap before importing the experiment API:

```python
import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_ROOT))

from experiment import EPOCHS, device_from_arg, evaluate_attempt, new_attempt
```

This imports the session-level `experiment.py` symlink, which resolves to the
single shared root infrastructure and root Task-A checkpoint.

Save the exact executed source, final `model.pt`, unedited `run.log`, and
unrounded `metrics.json`. Add `attempt_id`, `condition: "direct"`, and the
mechanism name exactly as written in the research response. Preserve the
checkpoint and infrastructure hashes returned by the evaluator.

Correcting a software defect is allowed, but changing the proposed mechanism
or its precommitted settings is not.
