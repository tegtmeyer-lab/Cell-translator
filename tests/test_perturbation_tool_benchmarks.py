import numpy as np
import yaml
from pathlib import Path

from analyses.perturbation_tool_benchmarks.common import (
    core_cell_id,
    delta_to_absolute_control_mean,
    map_to_reference_gene_order,
)
from analyses.perturbation_tool_benchmarks.cellot_export import reconstruct_full_gene_space


def test_core_cell_id_matches_notebook_rule():
    assert core_cell_id("control_neuron_CAACAGTAGGATGTTA-control_neuron") == "CAACAGTAGGATGTTA"


def test_scgpt_delta_to_absolute_control_mean():
    control = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    delta = np.array([[0.5, -5.0]], dtype=np.float32)
    out = delta_to_absolute_control_mean(delta, control, clip_nonnegative=True)
    np.testing.assert_allclose(out, [[2.5, 0.0]])


def test_gene_reordering():
    x = np.array([[10, 20, 30]])
    out = map_to_reference_gene_order(x, np.array(["B", "A", "C"]), np.array(["A", "B", "C"]))
    np.testing.assert_array_equal(out, [[20, 10, 30]])


def test_cellot_full_gene_fill():
    x_hvg = np.array([[7.0, 8.0]], dtype=np.float32)
    out = reconstruct_full_gene_space(
        x_hvg,
        np.array(["A", "C"]),
        np.array(["A", "B", "C"]),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(out, [[7.0, 2.0, 8.0]])


def test_benchmark_config_records_source_parameters():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "benchmarks" / "perturbation_tools.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["dataset"]["expected_shape"] == [21971, 18614]
    assert cfg["dataset"]["expected_test_control_cells"] == 2106
    assert cfg["cellot"]["n_hvg"] == 2000
    assert cfg["cellot"]["n_pcs"] == 50
    assert cfg["cellot"]["n_iters"] == 100000
    assert cfg["scgen"]["max_epochs"] == 50
    assert cfg["scgen"]["recorded_scgen_version"] == "2.1.1"
    assert cfg["scgpt"]["recorded_scgpt_version"] == "0.2.4"
    assert cfg["scgpt"]["max_seq_len"] is None
