"""Profile-first checkpoint selection with a declared no-material-regression gate."""
import math
from .config import GUARDS


def _finite(value):
    return value is not None and math.isfinite(float(value))


def assess(summary, cases, reference, reference_cases, cfg):
    """No oracle, unknown test record, or hidden weighted score is used here."""
    failures = []
    if set(cases) != set(reference_cases):
        return {"eligible": False, "failures": ["case_set_mismatch"]}
    rows = [("equal_case", summary, reference)] + [
        (case, cases[case], reference_cases[case]) for case in sorted(cases)
    ]
    for label, current, initial in rows:
        for metric in GUARDS:
            x, y = current.get(metric), initial.get(metric)
            if not _finite(x) or not _finite(y):
                failures.append(label + ":missing:" + metric)
                continue
            limit = float(y) * (1 + cfg.guard_relative_tolerance + cfg.numerical_relative_slack)
            if float(x) > limit:
                failures.append(label + ":guard:" + metric)
        # The main x-percentage objective must not worsen in either case.
        x, y = current.get("x_mean_pct"), initial.get("x_mean_pct")
        if not _finite(x) or not _finite(y) or float(x) > float(y) + cfg.min_x_improvement_percentage_points:
            failures.append(label + ":x_percentage_regression")
    return {"eligible": not failures, "failures": failures}


def improves_primary(current, selected, cfg):
    return (_finite(current.get("x_mean_pct")) and
            float(current["x_mean_pct"]) < float(selected["x_mean_pct"]) - cfg.min_x_improvement_percentage_points)
