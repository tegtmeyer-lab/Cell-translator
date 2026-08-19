"""Input validation and diagnostic summaries."""

from typing import Optional
import numpy as np

from cell_translator.hyperparameters import THR

def validate_expression_matrix(
    X: np.ndarray,
    name: str,
    expected_n_genes: Optional[int] = None,
) -> np.ndarray:
    """Validate the expression-matrix assumptions required by HurdleAE."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array; received shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one cell and one gene.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN/Inf values.")
    if np.any(X < 0):
        raise ValueError(
            f"{name} contains negative values. This implementation expects "
            "non-negative log1p-normalized expression values."
        )
    if expected_n_genes is not None and X.shape[1] != expected_n_genes:
        raise ValueError(
            f"{name} has {X.shape[1]} genes, but previous states have "
            f"{expected_n_genes}. All states must use the same HVGs/order."
        )
    return X
def basic_stats(X: np.ndarray, name: str = "X", thr: float = 1e-6):
    X = np.asarray(X, dtype=np.float32)
    zfrac = float((X <= thr).mean())
    gene_z = (X <= thr).mean(axis=0)
    row_sum = X.sum(axis=1)
    print(f"\n{name}: shape={X.shape}")
    print(f"  zero_frac={zfrac:.4f} | median_gene_zero={float(np.median(gene_z)):.4f}")
    print(f"  frac_genes_allzero={float((gene_z >= 1.0).mean()):.4f} | frac_genes_p0<0.2={float((gene_z < 0.2).mean()):.4f}")
    print(f"  median_rowsum={float(np.median(row_sum)):.3f} | mean_rowsum={float(np.mean(row_sum)):.3f}")
    print(f"  min/median/max={float(X.min()):.4f} / {float(np.median(X)):.4f} / {float(X.max()):.4f}")
def check_mag_scale(MAG_all: np.ndarray, X_all: np.ndarray, thr: float = 1e-6):
    """Verify mag scale matches data scale on nonzero entries."""
    nz_mask = X_all > thr
    print(f"\n[Mag Scale Check]")
    print(f"  mag  nonzero mean: {MAG_all[nz_mask].mean():.4f}")
    print(f"  x_true nonzero mean: {X_all[nz_mask].mean():.4f}")
    print(f"  ratio (should be ~1.0): {MAG_all[nz_mask].mean() / max(X_all[nz_mask].mean(), 1e-8):.4f}")
def check_latent_separation(Z_all: np.ndarray, slices: dict, state_names: list):
    """Verify latent space has variance across states (no collapse)."""
    print(f"\n[Latent Space Separation Check]")
    means = []
    for k in state_names:
        s, e = slices[k]
        Z_k = Z_all[s:e]
        mean_norm = float(np.linalg.norm(Z_k.mean(axis=0)))
        var = float(Z_k.var(axis=0).mean())
        print(f"  {k}: mean_norm={mean_norm:.4f} | latent_var={var:.4f}")
        means.append(Z_k.mean(axis=0))

    print(f"\n  Pairwise mean distances (should be > 0):")
    for i, ki in enumerate(state_names):
        for j, kj in enumerate(state_names):
            if j > i:
                dist = float(np.linalg.norm(means[i] - means[j]))
                print(f"    {ki} <-> {kj}: {dist:.4f}")
def check_hard_decode_quality(
    X_hard: np.ndarray, X_real: np.ndarray, name: str, thr: float = 1e-6
):
    """Compare zero fractions between hard decode and real data per state."""
    real_zfrac = float((X_real <= thr).mean())
    hard_zfrac = float((X_hard <= thr).mean())
    real_allz = float(((X_real <= thr).mean(axis=0) >= 1.0).mean())
    hard_allz = float(((X_hard <= thr).mean(axis=0) >= 1.0).mean())

    real_detect = (X_real > thr).mean(axis=0)
    hard_detect = (X_hard > thr).mean(axis=0)
    corr = float(np.corrcoef(real_detect, hard_detect)[0, 1])

    print(f"\n[Hard Decode Quality -- {name}]")
    print(f"  zero_frac  | real={real_zfrac:.4f} | hard={hard_zfrac:.4f}")
    print(f"  allzero_genes | real={real_allz:.4f} | hard={hard_allz:.4f}")
    print(f"  detection_rate_corr: {corr:.4f}  (target > 0.85)")
