"""Validate reproduced perturbation outputs against notebook reference values."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import math
import pandas as pd
import yaml
from config_utils import load_config


def close(a,b,tol): return math.isfinite(float(a)) and abs(float(a)-float(b)) <= tol

def check_gene(config_path, expected_path, tolerance=5e-4):
    cfg=load_config(config_path); gene=cfg['gene']
    exp=yaml.safe_load(Path(expected_path).read_text())
    failures=[]
    # Stage 1 checkpoint/history is deterministic under the recorded PyTorch seed.
    ckpt_path=Path(cfg['archived_artifacts']['ae_checkpoint'])
    if ckpt_path.exists():
        import torch
        try: ck=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        except TypeError: ck=torch.load(ckpt_path,map_location='cpu')
        ref=exp['stage1_hurdle_ae'][gene]; hist=ck['history']
        if int(hist['best_epoch']) != int(ref['best_epoch']): failures.append(f"Stage1 best_epoch: {hist['best_epoch']} != {ref['best_epoch']}")
        if not close(hist['best_val'],ref['best_validation_loss'],tolerance): failures.append(f"Stage1 best_val: {hist['best_val']} != {ref['best_validation_loss']}")
    else:
        print(f"SKIP Stage1 checkpoint (not found): {ckpt_path}")

    metrics_path=Path(cfg['outputs']['directory'])/'evaluation'/'heldout_metrics.json'
    ref=exp.get('heldout_translation',{}).get(gene)
    if ref and metrics_path.exists():
        obs=json.loads(metrics_path.read_text())
        for key,want in ref.items():
            tol=1e-3 if key.startswith('fraction_') else tolerance
            if key not in obs or not close(obs[key],want,tol): failures.append(f"{key}: {obs.get(key)} != {want} (tol={tol})")
    elif ref:
        print(f"SKIP held-out metrics (not found): {metrics_path}")
    else:
        print(f"No single-perturbation printed reference suite is available for {gene}; use comparison validation.")

    if failures:
        print('FAIL'); [print(' -',x) for x in failures]; return 1
    print(f"PASS: available {gene} outputs match notebook references within tolerance."); return 0


def check_comparison(csv_path, expected_path, tolerance=5e-6):
    exp=yaml.safe_load(Path(expected_path).read_text())
    df=pd.read_csv(csv_path); failures=[]
    cols=['perturbation','model','poorly_predicted_gene_count_KS_gt_0_2','poorly_predicted_gene_percent_KS_gt_0_2','detection_rate_pearson_r','logFC_top5000_DE_pearson_r']
    for pert,model,count,pct,det,lfc in exp['full_model_comparison']:
        sub=df[(df.perturbation==pert)&(df.model==model)]
        if len(sub)!=1: failures.append(f"Missing/duplicate row {pert}/{model}"); continue
        r=sub.iloc[0]
        if int(r[cols[2]])!=int(count): failures.append(f"{pert}/{model} poor count")
        for col,want in [(cols[3],pct),(cols[4],det),(cols[5],lfc)]:
            if not close(r[col],want,tolerance): failures.append(f"{pert}/{model} {col}: {r[col]} != {want}")
    if failures:
        print('FAIL'); [print(' -',x) for x in failures]; return 1
    print('PASS: model-comparison summary matches notebook reference values.'); return 0



def check_aggregate(csv_path, expected_path, tolerance=5e-4):
    exp = yaml.safe_load(Path(expected_path).read_text())
    df = pd.read_csv(csv_path)
    failures = []
    mapping = {
        "gene_mean_distribution_ks": "ks",
        "gene_mean_distribution_wasserstein": "wasserstein",
        "gene_mean_pearson_r": "gene_mean_r",
        "detection_rate_pearson_r": "detection_r",
        "effect_top5000_pearson_r": "effect_top5000_r",
    }
    for pert, ref in exp["aggregate_gene_mean_metrics"].items():
        sub = df[df.perturbation == pert]
        if len(sub) != 1:
            failures.append(f"Missing/duplicate aggregate row {pert}")
            continue
        row = sub.iloc[0]
        for col, key in mapping.items():
            if not close(row[col], ref[key], tolerance):
                failures.append(f"{pert} {col}: {row[col]} != {ref[key]}")
    if failures:
        print("FAIL")
        [print(" -", x) for x in failures]
        return 1
    print("PASS: aggregate gene-mean summary matches notebook reference values.")
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected', default='configs/genetic_perturbation/expected_results.yaml')
    p.add_argument('--gene-config')
    p.add_argument('--comparison-csv')
    p.add_argument('--aggregate-csv')
    args=p.parse_args()
    if not args.gene_config and not args.comparison_csv and not args.aggregate_csv: p.error('Provide --gene-config, --comparison-csv, and/or --aggregate-csv')
    rc=0
    if args.gene_config: rc=max(rc,check_gene(args.gene_config,args.expected))
    if args.comparison_csv: rc=max(rc,check_comparison(args.comparison_csv,args.expected))
    if args.aggregate_csv: rc=max(rc,check_aggregate(args.aggregate_csv,args.expected))
    raise SystemExit(rc)

if __name__=='__main__': main()
