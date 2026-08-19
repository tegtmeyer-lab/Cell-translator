"""Evaluate the perturbation HurdleAE reconstruction for control and target cells."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cell_translator.inference.perturbation import load_perturbation_ae_checkpoint, load_perturbation_latents
from cell_translator.training.hurdle_ae_training import NumpyExpressionDataset, hard_decode
from config_utils import load_config


def pearson_corr(a, b, eps=1e-12):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + eps))


def reconstruct(model, X, device, domain, batch_size=256):
    dl = DataLoader(NumpyExpressionDataset(X), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    soft, probs, mags = [], [], []
    model.eval()
    with torch.no_grad():
        for xb in dl:
            _, x_soft, _, p, mag = model(xb.to(device), domain)
            soft.append(x_soft.cpu().numpy()); probs.append(p.cpu().numpy()); mags.append(mag.cpu().numpy())
    return tuple(np.concatenate(v).astype(np.float32) for v in (soft, probs, mags))


def gene_reconstruction_metrics(X_true, X_hat, gene_names, threshold=1e-6, eps=1e-8):
    mask_nz = (X_true > threshold).astype(np.float32)
    mask_z = 1.0 - mask_nz
    absdiff = np.abs(X_hat - X_true)
    nz_count, z_count = mask_nz.sum(0) + eps, mask_z.sum(0) + eps
    return pd.DataFrame({
        "gene": np.asarray(gene_names).astype(str),
        "mae_nz": (absdiff * mask_nz).sum(0) / nz_count,
        "fp_mae": (np.abs(X_hat) * mask_z).sum(0) / z_count,
        "mean_true": X_true.mean(0), "mean_hat": X_hat.mean(0),
        "det_true": mask_nz.mean(0), "det_hat": (X_hat > threshold).mean(0),
        "mean_abs_err": np.abs(X_hat.mean(0) - X_true.mean(0)),
        "det_abs_err": np.abs((X_hat > threshold).mean(0) - mask_nz.mean(0)),
    })


def summarize(X_true, X_hat, threshold=1e-6):
    return {
        "gene_mean_correlation": pearson_corr(X_true.mean(0), X_hat.mean(0)),
        "detection_rate_correlation": pearson_corr((X_true > threshold).mean(0), (X_hat > threshold).mean(0)),
        "real_zero_fraction": float((X_true <= threshold).mean()),
        "reconstructed_zero_fraction": float((X_hat <= threshold).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gene", required=True, choices=["APOE", "MSI2", "NR1H3", "LRRC4C", "ANKRD12"])
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    if cfg["gene"] != args.gene: raise ValueError("Gene/config mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xc = np.load(cfg["data"]["control_expression"]).astype(np.float32)
    Xt = np.load(cfg["data"]["target_expression"]).astype(np.float32)
    genes = np.load(cfg["data"]["gene_names"], allow_pickle=False)
    model, _ = load_perturbation_ae_checkpoint(cfg["archived_artifacts"]["ae_checkpoint"], args.gene, Xc.shape[1], device)
    lat = load_perturbation_latents(cfg["archived_artifacts"]["ae_outputs"], args.gene)
    threshold = float(cfg["evaluation"]["threshold"])
    out = Path(cfg["outputs"]["directory"]) / "ae_evaluation"; out.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, X, domain, tau in [("control", Xc, "control", lat.get("tau_ctrl")), (args.gene, Xt, "target", lat["tau_target"])]:
        soft, P, MAG = reconstruct(model, X, device, domain)
        hard = hard_decode(P, MAG, tau) if tau is not None else None
        report[label] = {"soft": summarize(X, soft, threshold)}
        gene_reconstruction_metrics(X, soft, genes, threshold).to_csv(out / f"{label.lower()}_soft_gene_metrics.csv", index=False)
        if hard is not None:
            report[label]["hard"] = summarize(X, hard, threshold)
            gene_reconstruction_metrics(X, hard, genes, threshold).to_csv(out / f"{label.lower()}_hard_gene_metrics.csv", index=False)
    (out / "ae_reconstruction_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
