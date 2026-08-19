from pathlib import Path


def test_validation_forwards_masked_smooth_l1_eps_mode():
    path = Path(__file__).resolve().parents[1] / "cell_translator" / "training" / "hurdle_ae_training.py"
    text = path.read_text(encoding="utf-8")
    validation_section = text.split("# ---- validate ----", 1)[1].split("val_loss =", 1)[0]
    assert "masked_smooth_l1_eps_mode=masked_smooth_l1_eps_mode" in validation_section
