# FrameShift

FrameShift is a controlled MNIST continual-learning experiment comparing one
fixed fine-tuning baseline with five direct-search attempts and five
distant-analogy-search attempts.

The model, Task-A checkpoint procedure, datasets, optimizer, learning rate,
batch size, epoch budget, and evaluation path are fixed in `experiment.py`.
The direct and analogy conditions are prepared as isolated research sessions
with unchanged prompts and no prewritten attempts.

## Layout

- `experiment.py` — locked training and evaluation infrastructure
- `baseline/attempt.py` — fixed ordinary Task-B fine-tuning baseline
- `direct_session/` — direct-search prompt, empty attempt workspace, and a
  symlink to the shared root infrastructure
- `analogy_session/` — analogy-search prompt, empty attempt workspace, and a
  symlink to the shared root infrastructure
- `compare.py` — raw and pairwise comparison for all 11 completed runs
- `validate.py` — integrity checks for the baseline and empty research sessions

## Setup

```bash
python3 -m pip install -r requirements.txt
mkdir -p direct_session/attempts analogy_session/attempts
python3 validate.py
```

Train Task A once and run the fixed baseline:

```bash
python3 experiment.py train-a --device cuda
python3 baseline/attempt.py --device cuda
```

Both research sessions use the same root `experiment.py` through a symlink, so
`new_attempt` reads the one Task-A checkpoint under root `artifacts/`. There are
no copied infrastructure files or per-session Task-A checkpoints.

Run the direct and analogy research conditions in their respective isolated
directories using their local research instructions. After all ten research
attempts finish, generate the comparison from the repository root:

```bash
python3 compare.py
```

This is an exploratory experiment; five search attempts per condition do not
support claims of statistical significance.
