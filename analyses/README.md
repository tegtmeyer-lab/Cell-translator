# Validation and benchmark analyses

This directory contains analyses that benchmark or validate Cell Translator but are not independent biological applications.

- `baseline_comparison/`: simple perturbation baselines evaluated against archived Cell Translator predictions.
- `leave_one_line_out/`: control-line generalization validation for Cell Painting morphology.

These modules do not redefine or replace the canonical model implementations unless an analysis historically used a deliberately reduced validation-specific architecture; such definitions remain local to that analysis and are documented explicitly.
