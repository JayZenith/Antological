# Analogy session results

Completed analogy_1 through analogy_5 sequentially on the authorized instance. Each research response was saved before implementation; every method was run once for exactly five Task-B epochs, with no revisions after metrics. All commands used /venv/main/bin/python; each training run used CUDA.

Initial Task-A accuracy: 97.82%. Values in the table are displayed as percentages; source metrics remain unrounded.

| Attempt | Mechanism | Task A after B | Task B | Forgetting (percentage points) |
| --- | --- | ---: | ---: | ---: |
| analogy_1 | Reference-panel conservation | 97.68% | 96.95% | 0.14 |
| analogy_2 | Protected structural connections | 29.46% | 97.63% | 68.36 |
| analogy_3 | Canal-wall deflection | 42.20% | 97.78% | 55.62 |
| analogy_4 | Scheduled dormant-branch flushing | 88.10% | 97.62% | 9.72 |
| analogy_5 | Mooring-radius constraint | 92.33% | 65.32% | 5.49 |

Attempt 1 had the highest Task-A retention among these five attempts and achieved 96.95% Task-B accuracy. Attempt 3 had the highest Task-B accuracy, at 97.78%, but retained only 42.20% Task-A accuracy. These are single runs using the fixed seed, not estimates across seeds. No baseline or direct-condition comparison was performed.

Each attempts/analogy_n/ directory contains research.md, attempt.py, model.pt (the final state dictionary), unedited run.log, and metrics.json. session_results.json contains unrounded results and SHA-256 hashes for all required files.

All attempts processed 60,000 Task-B examples and 469 supplied Adam steps per epoch. Evaluation was called only through evaluate_attempt(ctx). Metadata matches each research response verbatim. The infrastructure file hash at final audit equals the hash recorded by every evaluator.

Checkpoint SHA-256: 05a45f578c40365a67e1fc7eaf8fbbab76e32e234833da352bc49b5795b9722c

Infrastructure SHA-256: 5bf408b12c4f92fc75bba176268f01a8c5bace36878f9acda6ed38a5c96f5fa5

Access note: before locating and reading RESEARCH_ANALOGY.md, initial orientation listed the supplied direct_session directory, the repository root directory, and searched repository filenames for the requested instruction and AGENTS.md. No baseline/direct implementation, research prompt, metrics, or comparison/report content was opened. After reading the isolation boundary, repository content access was limited to the allowed experiment.py symlink and files created in analogy_session. Shared infrastructure itself loads the checkpoint and datasets as prescribed.
