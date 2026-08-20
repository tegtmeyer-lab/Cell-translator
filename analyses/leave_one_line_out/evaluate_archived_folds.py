#!/usr/bin/env python
"""Recompute the final five-line LOLO metrics from archived fold matrices."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml
from analyses.leave_one_line_out.evaluation import evaluate_archived_transition


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); args=p.parse_args()
    with open(args.config,"r",encoding="utf-8") as h: cfg=yaml.safe_load(h)
    root=cfg["archived_folds_root"]
    rows=[]
    for transition in cfg["transitions"]:
        selected=[int(x) for x in cfg["selected_heldout_lines"][transition]]
        df=evaluate_archived_transition(root,transition,selected_lines=selected,pca_log_base=cfg["evaluation"]["final_pca_log_base"])
        rows.append(df)
    out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    outdir=Path(cfg["archived_evaluation_output_dir"]); outdir.mkdir(parents=True,exist_ok=True)
    out.to_csv(outdir/"lodo_five_line_corrected_metrics.csv",index=False)
    med=out.groupby("transition",as_index=False).agg(
        n_heldout_lines=("heldout_line","nunique"),
        median_effect_size_pearson_r=("effect_size_pearson_r","median"),
        median_featurewise_ks=("median_ks","median"),
        median_pca_variance_log_error=("median_abs_log_pca_variance_ratio","median"),
        median_auc_distance_from_0p5=("auc_distance_from_0p5","median"),
    )
    med.to_csv(outdir/"lodo_five_line_transition_medians.csv",index=False)
    print(out.to_string(index=False)); print(f"\nSaved to {outdir}")

if __name__=="__main__": main()
