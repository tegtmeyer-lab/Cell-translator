"""Validate pancreatic reproduction metrics against reference values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--expected", required=True)
    args = ap.parse_args()

    with open(args.metrics, encoding="utf-8") as f:
        observed = json.load(f)
    with open(args.expected, encoding="utf-8") as f:
        expected = yaml.safe_load(f)["ngn3_to_endocrine"]["translation_metrics"]

    failures = []
    for key, spec in expected.items():
        if not isinstance(spec, dict) or "value" not in spec:
            continue
        if key not in observed:
            failures.append(f"{key}: missing from metrics JSON")
            continue
        value = float(spec["value"])
        tolerance = float(spec.get("tolerance", 0.0))
        obs = float(observed[key])
        delta = abs(obs - value)
        status = "PASS" if delta <= tolerance else "FAIL"
        print(f"{status:4s} {key}: observed={obs:.6g}, expected={value:.6g}, tolerance={tolerance:.6g}")
        if status == "FAIL":
            failures.append(key)

    if failures:
        raise SystemExit("Validation failed: " + ", ".join(failures))
    print("All configured pancreatic translation metrics passed.")


if __name__ == "__main__":
    main()
