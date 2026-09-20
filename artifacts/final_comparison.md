# Final comparison

> Exploratory comparison only; no statistical significance is claimed.

## Raw attempts

| Attempt | Condition | Mechanism (verbatim) | A before B | A after B | B after B | Forgetting |
|---|---|---|---:|---:|---:|---:|
| baseline | baseline | ordinary Task-B cross-entropy training | 0.978200 | 0.268400 | 0.977400 | 0.709800 |
| direct_1 | direct | Paired inverse-rotation rehearsal | 0.978200 | 0.977600 | 0.971000 | 0.000600 |
| direct_2 | direct | Original-view teacher distillation | 0.978200 | 0.976800 | 0.969500 | 0.001400 |
| direct_3 | direct | Task-A constrained Adam displacement | 0.978200 | 0.453700 | 0.979000 | 0.524500 |
| direct_4 | direct | Checkpoint-centered quadratic penalty | 0.978200 | 0.551500 | 0.945000 | 0.426700 |
| direct_5 | direct | Original-view hidden-feature anchoring | 0.978200 | 0.807400 | 0.974200 | 0.170800 |
| analogy_1 | analogy | Reference-panel conservation | 0.978200 | 0.976800 | 0.969500 | 0.001400 |
| analogy_2 | analogy | Protected structural connections | 0.978200 | 0.294600 | 0.976300 | 0.683600 |
| analogy_3 | analogy | Canal-wall deflection | 0.978200 | 0.422000 | 0.977800 | 0.556200 |
| analogy_4 | analogy | Scheduled dormant-branch flushing | 0.978200 | 0.881000 | 0.976200 | 0.097200 |
| analogy_5 | analogy | Mooring-radius constraint | 0.978200 | 0.923300 | 0.653200 | 0.054900 |

## Condition summaries

| Condition | Mean A after B | Mean forgetting | Mean B after B |
|---|---:|---:|---:|
| baseline | 0.268400 | 0.709800 | 0.977400 |
| direct | 0.753400 | 0.224800 | 0.967740 |
| analogy | 0.699540 | 0.278660 | 0.910600 |

## Pairwise comparisons

Deltas are `second condition − first condition`. Higher A-after-B and B-after-B are better; lower forgetting is better.

| Comparison | Metric | First mean | Second mean | Delta |
|---|---|---:|---:|---:|
| baseline vs direct | A_after_B | 0.268400 | 0.753400 | +0.485000 |
| baseline vs direct | forgetting | 0.709800 | 0.224800 | -0.485000 |
| baseline vs direct | B_after_B | 0.977400 | 0.967740 | -0.009660 |
| baseline vs analogy | A_after_B | 0.268400 | 0.699540 | +0.431140 |
| baseline vs analogy | forgetting | 0.709800 | 0.278660 | -0.431140 |
| baseline vs analogy | B_after_B | 0.977400 | 0.910600 | -0.066800 |
| direct vs analogy | A_after_B | 0.753400 | 0.699540 | -0.053860 |
| direct vs analogy | forgetting | 0.224800 | 0.278660 | +0.053860 |
| direct vs analogy | B_after_B | 0.967740 | 0.910600 | -0.057140 |

## Qualitative analysis

After all runs, compare the verbatim direct and analogy research artifacts here, beginning with the raw mechanisms before any retrospective grouping.
