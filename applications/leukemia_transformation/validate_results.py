"""Validate acute-leukemia reproduction metrics against references."""
from __future__ import annotations
import argparse
import json
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--workflow", choices=["healthy_cd34_to_bmmc", "disease_cd34_to_mpal"], required=True)
    args = ap.parse_args()
    with open(args.metrics, encoding="utf-8") as f:
        observed = json.load(f)
    with open(args.expected, encoding="utf-8") as f:
        expected = yaml.safe_load(f)[args.workflow]["translation_metrics"]

    failures = []
    for key, spec in expected.items():
        if key not in observed:
            failures.append(f"{key}: missing")
            continue
        value = float(spec["value"])
        tol = float(spec.get("tolerance", 0.0))
        obs = float(observed[key])
        ok = abs(obs - value) <= tol
        print(f"{'PASS' if ok else 'FAIL':4s} {key}: observed={obs:.6g}, expected={value:.6g}, tolerance={tol:.6g}")
        if not ok:
            failures.append(key)
    if failures:
        raise SystemExit("Validation failed: " + ", ".join(failures))
    print("All configured acute-leukemia translation metrics passed.")


if __name__ == "__main__":
    main()
