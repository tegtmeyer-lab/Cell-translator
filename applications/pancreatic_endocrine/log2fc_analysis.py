"""Fold-change analyses for NGN3-high-EP to Endocrine translation.

Two analysis definitions are retained: the filtered analysis used for the reported Pearson/sign-accuracy
summary, and a later all-gene linear-space visualization using a smaller
pseudocount.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config_utils import load_config


def _linear_mean(x, clip_max: float) -> np.ndarray:
    return np.expm1(np.clip(np.asarray(x, np.float32), -clip_max, clip_max)).mean(0)


def _detect(x, threshold: float = 1e-6) -> np.ndarray:
    return (np.asarray(x, np.float32) > threshold).mean(0)


def filtered_log2fc(source, real_target, translated, genes, cfg: dict) -> tuple[pd.DataFrame, dict]:
    fc = cfg["log2fc"]["filtered_reported"]
    clip_max = float(fc["linear_clip_max"])
    mean_src = _linear_mean(source, clip_max)
    mean_real = _linear_mean(real_target, clip_max)
    mean_fake = _linear_mean(translated, clip_max)
    det_src, det_real = _detect(source), _detect(real_target)

    finite = np.isfinite(mean_src) & np.isfinite(mean_real) & np.isfinite(mean_fake)
    expr_keep = (mean_src > float(fc["expression_threshold"])) | (mean_real > float(fc["expression_threshold"]))
    det_keep = (det_src > float(fc["detection_fraction_threshold"])) | (det_real > float(fc["detection_fraction_threshold"]))
    keep = finite & expr_keep & det_keep

    pc = float(fc["pseudocount"])
    cap = float(fc["log2fc_cap"])
    real_fc = np.clip(np.log2((mean_real[keep] + pc) / (mean_src[keep] + pc)), -cap, cap)
    fake_fc = np.clip(np.log2((mean_fake[keep] + pc) / (mean_src[keep] + pc)), -cap, cap)
    ok = np.isfinite(real_fc) & np.isfinite(fake_fc)
    genes_keep = np.asarray(genes)[keep][ok]

    df = pd.DataFrame({
        "gene": genes_keep,
        "mean_source": mean_src[keep][ok],
        "mean_target_real": mean_real[keep][ok],
        "mean_target_translated": mean_fake[keep][ok],
        "detection_source": det_src[keep][ok],
        "detection_target_real": det_real[keep][ok],
        "log2fc_real": real_fc[ok],
        "log2fc_translated": fake_fc[ok],
    })
    df["abs_log2fc_real"] = np.abs(df["log2fc_real"])
    df["log2fc_error"] = np.abs(df["log2fc_real"] - df["log2fc_translated"])
    df["sign_match"] = np.sign(df["log2fc_real"]) == np.sign(df["log2fc_translated"])
    summary = {
        "total_genes": int(len(genes)),
        "finite_genes": int(finite.sum()),
        "genes_after_expression_filter": int((finite & expr_keep).sum()),
        "genes_after_detection_filter": int(keep.sum()),
        "pearson_r": float(pearsonr(df["log2fc_real"], df["log2fc_translated"]).statistic),
        "sign_accuracy": float(df["sign_match"].mean()),
    }
    return df, summary


def all_gene_log2fc(source, real_target, translated, genes, cfg: dict) -> pd.DataFrame:
    fc = cfg["log2fc"]["all_gene_secondary"]
    pc = float(fc["pseudocount"])
    src_lin = np.expm1(np.asarray(source, np.float32))
    real_lin = np.expm1(np.asarray(real_target, np.float32))
    fake_lin = np.expm1(np.asarray(translated, np.float32))
    src_mean, real_mean, fake_mean = src_lin.mean(0), real_lin.mean(0), fake_lin.mean(0)
    return pd.DataFrame({
        "gene": genes,
        "source_mean_linear": src_mean,
        "target_real_mean_linear": real_mean,
        "target_translated_mean_linear": fake_mean,
        "true_log2fc": np.log2((real_mean + pc)/(src_mean + pc)),
        "predicted_log2fc": np.log2((fake_mean + pc)/(src_mean + pc)),
        "source_detection": _detect(source, float(fc["detection_threshold"])),
        "target_real_detection": _detect(real_target, float(fc["detection_threshold"])),
        "target_translated_detection": _detect(translated, float(fc["detection_threshold"])),
    })


def _target_validation(ae_npz: Path, target_matrix: Path, target_state: str) -> np.ndarray:
    ae = np.load(ae_npz, allow_pickle=False)
    names = [str(x) for x in ae["state_names"].tolist()]
    slices = {n:(int(s),int(e)) for n,s,e in zip(names, ae["slice_starts"], ae["slice_ends"])}
    s,e = slices[target_state]
    val_idx = ae["val_idx"] if "val_idx" in ae.files else ae["test_idx"]
    local = np.intersect1d(np.arange(s,e), val_idx) - s
    return np.load(target_matrix).astype(np.float32)[local]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    prepared = Path(cfg["paths"]["prepared_dir"])
    out = Path(cfg["paths"]["output_dir"])
    slug = f"{cfg['source_state'].lower()}_to_{cfg['target_state'].lower()}".replace(" ", "_")

    source = np.load(prepared / f"{cfg['source_state']}_X_hvg.npy").astype(np.float32)
    target_path = prepared / f"{cfg['target_state']}_X_hvg.npy"
    target_val = _target_validation(out / f"{cfg['run_name']}_outputs.npz", target_path, cfg["target_state"])
    translated = np.load(out / slug / "translated_source_outputs.npz", allow_pickle=False)["X_translated"].astype(np.float32)
    genes = np.load(prepared / "hvg_genes.npy", allow_pickle=False).astype(str)

    reported, summary = filtered_log2fc(source, target_val, translated, genes, cfg)
    reported.to_csv(out / "log2fc_filtered_reported.csv", index=False)
    pd.Series(summary).to_json(out / "log2fc_filtered_reported_summary.json", indent=2)
    print(summary)

    secondary = all_gene_log2fc(source, target_val, translated, genes, cfg)
    secondary.to_csv(out / "log2fc_all_gene_secondary.csv", index=False)


if __name__ == "__main__":
    main()
