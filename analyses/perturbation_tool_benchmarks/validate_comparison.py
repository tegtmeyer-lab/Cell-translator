"""Validate the final cross-tool comparison table against archived notebook reference values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    expected = yaml.safe_load(Path(args.expected).read_text())
    rows = expected["final_comparison"]
    cols = expected["columns"]

    failures = []
    for row in rows:
        ref = dict(zip(cols, row))
        hit = df[(df["perturbation"] == ref["perturbation"]) & (df["model"] == ref["model"])]
        if len(hit) != 1:
            failures.append(f"missing/duplicate row {ref['perturbation']}/{ref['model']}")
            continue
        got = hit.iloc[0]
        for key in ("poor_count", "poor_percent", "detection_r", "effect_top5000_r"):
            if key not in got:
                failures.append(f"CSV missing column {key}")
                continue
            if not np.isclose(float(got[key]), float(ref[key]), atol=args.atol, rtol=0):
                failures.append(
                    f"{ref['perturbation']}/{ref['model']} {key}: got {got[key]}, expected {ref[key]}"
                )

    if failures:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(failures))
    print(f"Validated {len(rows)} benchmark rows against archived reference values.")


if __name__ == "__main__":
    main()
