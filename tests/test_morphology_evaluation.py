import importlib.util
from pathlib import Path
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "applications" / "cell_painting_morphology" / "evaluation.py"
spec = importlib.util.spec_from_file_location("morphology_evaluation", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_corrected_nn_reference_contains_source_and_target():
    rng = np.random.default_rng(42)
    source = rng.normal(-5, 0.1, size=(30, 4)).astype(np.float32)
    target = rng.normal(5, 0.1, size=(30, 4)).astype(np.float32)
    translated = rng.normal(5, 0.1, size=(30, 4)).astype(np.float32)
    metrics, ks = mod.evaluate_translation(source, target, translated, seed=42)
    assert metrics["target_nn_percent"] > 90.0
    assert len(ks) == 4
