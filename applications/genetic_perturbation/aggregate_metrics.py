"""Aggregate gene-mean prediction metrics across the five perturbations.

This reproduces the final combined analysis that compares the distribution of
per-gene mean expression values.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    rows=[]; top=int(cfg.get('top_de_genes',5000)); thr=float(cfg.get('threshold',1e-6))
    for pert,item in cfg['perturbations'].items():
        Xc=np.load(item['control']).astype(np.float32)
        Xr=np.load(item['real']).astype(np.float32)
        Xp=np.load(item['predictions']['CellTranslator']).astype(np.float32)
        mr,mp=Xr.mean(0),Xp.mean(0)
        dr,dp=(Xr>thr).mean(0),(Xp>thr).mean(0)
        er=mr-Xc.mean(0); ep=mp-Xc.mean(0)
        idx=np.argsort(np.abs(er))[::-1][:top]
        rows.append({
            'perturbation':pert,
            'gene_mean_distribution_ks':float(ks_2samp(mr,mp).statistic),
            'gene_mean_distribution_wasserstein':float(wasserstein_distance(mr,mp)),
            'gene_mean_pearson_r':float(pearsonr(mr,mp)[0]),
            'detection_rate_pearson_r':float(pearsonr(dr,dp)[0]),
            'effect_top5000_pearson_r':float(pearsonr(er[idx],ep[idx])[0]),
        })
    out=Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/'celltranslator_aggregate_gene_mean_metrics.csv',index=False)
    print(df.to_string(index=False))

if __name__=='__main__': main()
