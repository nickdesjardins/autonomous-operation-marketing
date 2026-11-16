"""The cold-start defaults have evidence from this project's own cases."""

import json
from pathlib import Path

from app.models import cases, rules, testing


def test_the_current_defaults_passed_the_saved_project_specific_checks() -> None:
    path = Path(__file__).resolve().parents[1] / "benchmarks/2026-11-16-default-review.json"
    evidence = json.loads(path.read_text(encoding="utf-8-sig"))
    primary, fallback = testing.default_chain()
    passed = {primary: 0, fallback: 0}
    for run in evidence:
        rows = rules.judge(
            [primary, fallback],
            run["calls"],
            cases.case_ids(),
            run["timeout_seconds"],
            finished=True,
        )
        for row in rows:
            passed[row["model"]] += row["verdict"] == "passed"
    assert passed == {primary: 3, fallback: 2}
    assert testing.DEFAULT_MEASURED_ON == max(run["tested"][:10] for run in evidence)
