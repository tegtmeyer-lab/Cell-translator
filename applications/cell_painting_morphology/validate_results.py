"""Compare reproduced morphology metrics with notebook reference values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml


def validate(metrics_path, expected_path, workflow, transition, atol=0.01):
    metrics = json.loads(Path(metrics_path).read_text())
    expected = yaml.safe_load(Path(expected_path).read_text())[workflow]["translations"][transition]
    checks = {}
    for key, expected_value in expected.items():
        if not isinstance(expected_value, (int, float)) or key not in metrics:
            continue
        observed = float(metrics[key])
        diff = abs(observed - float(expected_value))
        checks[key] = {"observed": observed, "expected": float(expected_value), "abs_diff": diff, "pass": diff <= atol}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--workflow", required=True, choices=["control_line24", "deletion_line45"])
    parser.add_argument("--transition", required=True)
    parser.add_argument("--atol", type=float, default=0.01)
    args = parser.parse_args()
    checks = validate(args.metrics, args.expected, args.workflow, args.transition, args.atol)
    for name, item in checks.items():
        print(f"{name}: observed={item['observed']:.6f}, expected={item['expected']:.6f}, diff={item['abs_diff']:.6f}, pass={item['pass']}")
    if checks and not all(item["pass"] for item in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
