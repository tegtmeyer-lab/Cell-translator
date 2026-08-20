"""Compare reproduced primary-organoid metrics with reference values."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--dataset", default="primary_scp1756")
    args = p.parse_args()
    observed = json.loads(Path(args.metrics).read_text())
    expected = yaml.safe_load(Path(args.expected).read_text())[args.dataset]
    failed = []
    for key, rule in expected.get("metrics", {}).items():
        if key not in observed:
            failed.append(f"missing {key}")
            continue
        val = float(observed[key]); ref = float(rule["value"]); tol = float(rule["tolerance"])
        ok = abs(val-ref) <= tol
        print(f"{key}: observed={val:.6g} expected={ref:.6g} tol={tol:.3g} [{'OK' if ok else 'FAIL'}]")
        if not ok: failed.append(key)
    if failed:
        raise SystemExit("Reference validation failed: " + ", ".join(failed))

if __name__ == "__main__":
    main()
