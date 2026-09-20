#!/usr/bin/env python3
"""Compare one baseline, five direct, and five analogy runs."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("baseline", "direct", "analogy")
METRICS = ("A_after_B", "forgetting", "B_after_B")


def condition_mean(rows: list[dict], condition: str, metric: str) -> float:
    return statistics.fmean(row[metric] for row in rows if row["condition"] == condition)


def main() -> None:
    expected = [("baseline", "baseline", ROOT / "baseline" / "metrics.json")]
    expected += [
        (f"direct_{i}", "direct", ROOT / "direct_session" / "attempts" / f"direct_{i}" / "metrics.json")
        for i in range(1, 6)
    ]
    expected += [
        (f"analogy_{i}", "analogy", ROOT / "analogy_session" / "attempts" / f"analogy_{i}" / "metrics.json")
        for i in range(1, 6)
    ]
    rows = []
    for attempt_id, expected_condition, path in expected:
        if not path.exists():
            raise SystemExit(f"Incomplete experiment: missing {path}")
        row = json.loads(path.read_text())
        if row.get("attempt_id") != attempt_id or row.get("condition") != expected_condition:
            raise SystemExit(f"Attempt identity or condition mismatch in {path}")
        rows.append(row)

    expected_counts = {"baseline": 1, "direct": 5, "analogy": 5}
    if len(rows) != 11 or any(sum(row["condition"] == c for row in rows) != count for c, count in expected_counts.items()):
        raise SystemExit("Expected one baseline, five direct, and five analogy runs")
    if (len({row["checkpoint_sha256"] for row in rows}) != 1
            or len({row["infrastructure_sha256"] for row in rows}) != 1
            or len({row["A_before_B"] for row in rows}) != 1):
        raise SystemExit("Control failure: checkpoint, infrastructure, or A_before_B differs")

    lines = [
        "# Final comparison",
        "",
        "> Exploratory comparison only; no statistical significance is claimed.",
        "",
        "## Raw attempts",
        "",
        "| Attempt | Condition | Mechanism (verbatim) | A before B | A after B | B after B | Forgetting |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['attempt_id']} | {row['condition']} | "
            f"{row.get('mechanism', row.get('mechanism_name', ''))} | "
            f"{row['A_before_B']:.6f} | {row['A_after_B']:.6f} | "
            f"{row['B_after_B']:.6f} | {row['forgetting']:.6f} |"
        )

    lines += [
        "",
        "## Condition summaries",
        "",
        "| Condition | Mean A after B | Mean forgetting | Mean B after B |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = [condition_mean(rows, condition, metric) for metric in METRICS]
        lines.append(f"| {condition} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")

    lines += [
        "",
        "## Pairwise comparisons",
        "",
        "Deltas are `second condition − first condition`. Higher A-after-B and B-after-B are better; lower forgetting is better.",
        "",
        "| Comparison | Metric | First mean | Second mean | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for first, second in (("baseline", "direct"), ("baseline", "analogy"), ("direct", "analogy")):
        for metric in METRICS:
            first_mean = condition_mean(rows, first, metric)
            second_mean = condition_mean(rows, second, metric)
            lines.append(
                f"| {first} vs {second} | {metric} | {first_mean:.6f} | "
                f"{second_mean:.6f} | {second_mean - first_mean:+.6f} |"
            )

    lines += [
        "",
        "## Qualitative analysis",
        "",
        "After all runs, compare the verbatim direct and analogy research artifacts here, beginning with the raw mechanisms before any retrospective grouping.",
        "",
    ]
    output = ROOT / "artifacts" / "final_comparison.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
