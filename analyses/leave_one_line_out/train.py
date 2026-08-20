#!/usr/bin/env python
"""Retrain the validation-specific reduced LOLO model on selected control lines."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from analyses.leave_one_line_out.training import (
    fit_lodo_ae, transform_with_lodo_scaler, encode_lodo, decode_lodo,
    train_lodo_cyclegan,
)
from analyses.leave_one_line_out.evaluation import evaluate_fold_arrays


def load_config(path):
    with open(path,"r",encoding="utf-8") as h: return yaml.safe_load(h)


def prepare_control_dataframe(df,cfg):
    meta=cfg["metadata"]; out=df[df[meta["genotype_column"]].astype(str).str.strip().str.lower()==str(meta["control_label"]).lower()].copy()
    line_col=meta["line_column"]; internal="__LODO_LINE_ID__"
    out[internal]=pd.to_numeric(out[line_col],errors="coerce"); out=out[out[internal].notna()].copy(); out[internal]=out[internal].astype(int)
    out=out[~out[internal].isin(set(cfg["excluded_line_ids"]))].copy()
    return out,internal


def get_feature_columns(a,b,cfg,internal):
    drop=tuple(s.lower() for s in cfg["feature_selection"]["drop_substrings"])
    threshold=float(cfg["feature_selection"]["min_nonmissing_fraction"])
    cols=[]
    for c in sorted(set(a.columns)&set(b.columns)):
        if c==internal or any(p in c.lower() for p in drop): continue
        if not (pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])): continue
        if pd.to_numeric(a[c],errors="coerce").notna().mean()>=threshold and pd.to_numeric(b[c],errors="coerce").notna().mean()>=threshold:
            cols.append(c)
    return cols


def sample_df(df,max_n,seed):
    if max_n is None or len(df)<=max_n: return df.copy()
    return df.sample(n=int(max_n),random_state=seed).copy()


def run_fold(transition_key,tconf,source_all,target_all,internal,feature_cols,heldout,cfg,outdir):
    seed=int(cfg["random_state"]); sampling=cfg["sampling"]
    source_train=source_all[source_all[internal]!=heldout].copy(); target_train=target_all[target_all[internal]!=heldout].copy()
    source_hold=source_all[source_all[internal]==heldout].copy(); target_hold=target_all[target_all[internal]==heldout].copy()
    if source_hold.empty or target_hold.empty: raise ValueError(f"Line {heldout} must contain both domains")
    source_train=sample_df(source_train,sampling["max_train_cells_per_domain"],seed)
    target_train=sample_df(target_train,sampling["max_train_cells_per_domain"],seed+1)
    source_hold=sample_df(source_hold,sampling["max_heldout_source_cells"],seed+2)
    target_hold=sample_df(target_hold,sampling["max_heldout_target_cells"],seed+3)
    fold=Path(outdir)/transition_key/f"heldout_line_{heldout}"; fold.mkdir(parents=True,exist_ok=True)

    src_ae=fit_lodo_ae(source_train,feature_cols,fold/"source_train_lines",cfg["autoencoder"],seed)
    tgt_ae=fit_lodo_ae(target_train,feature_cols,fold/"target_train_lines",cfg["autoencoder"],seed+1)
    Xs_scaled,Xs_orig=transform_with_lodo_scaler(source_hold,feature_cols,src_ae["train_median"],src_ae["scaler"])
    Xt_scaled,Xt_orig=transform_with_lodo_scaler(target_hold,feature_cols,tgt_ae["train_median"],tgt_ae["scaler"])
    Zs_hold=encode_lodo(src_ae["model"],Xs_scaled,src_ae["device"])
    gan=train_lodo_cyclegan(src_ae["Z_all"],tgt_ae["Z_all"],fold,tconf["source_state"],tconf["target_state"],cfg["cyclegan"],seed)
    Zs_std=(Zs_hold-gan["mean_src"])/gan["std_src"]
    Zpred_std=gan["G_forward"](Zs_std.astype(np.float32),training=False).numpy()
    Zpred=Zpred_std*gan["std_tgt"]+gan["mean_tgt"]
    Xpred_scaled=decode_lodo(tgt_ae["model"],Zpred.astype(np.float32),tgt_ae["device"])
    Xpred=tgt_ae["scaler"].inverse_transform(Xpred_scaled).astype(np.float32)
    np.save(fold/"heldout_source_original.npy",Xs_orig); np.save(fold/"heldout_measured_target_original.npy",Xt_orig); np.save(fold/"predicted_target_from_heldout_source.npy",Xpred)
    metrics=evaluate_fold_arrays(Xs_orig,Xt_orig,Xpred,pca_components=int(cfg["evaluation"]["pca_components"]),pca_log_base=cfg["evaluation"]["training_summary_pca_log_base"],seed=seed)
    row={"transition":transition_key,"transition_label":f"{tconf['source_state']}→{tconf['target_state']}","heldout_line":heldout,
         "n_train_source":len(source_train),"n_train_target":len(target_train),"n_heldout_source":len(source_hold),"n_heldout_target":len(target_hold),"n_features":len(feature_cols),
         "source_ae_best_epoch":src_ae["best_epoch"],"source_ae_best_val":src_ae["best_val"],"target_ae_best_epoch":tgt_ae["best_epoch"],"target_ae_best_val":tgt_ae["best_val"],**metrics}
    pd.DataFrame([row]).to_csv(fold/"heldout_summary.csv",index=False)
    return row


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--transition",default="all"); args=p.parse_args()
    cfg=load_config(args.config); data={k:pd.read_parquet(v) for k,v in cfg["data"].items()}
    outdir=Path(cfg["training_output_dir"]); outdir.mkdir(parents=True,exist_ok=True); rows=[]
    transition_keys=list(cfg["transitions"]) if args.transition=="all" else [args.transition]
    for key in transition_keys:
        tc=cfg["transitions"][key]; src,internal=prepare_control_dataframe(data[tc["source_dataset"]],cfg); tgt,_=prepare_control_dataframe(data[tc["target_dataset"]],cfg)
        features=get_feature_columns(src,tgt,cfg,internal)
        tdir=outdir/key; tdir.mkdir(parents=True,exist_ok=True); pd.Series(features,name="feature").to_csv(tdir/"lodo_common_feature_columns.csv",index=False)
        eligible=sorted(set(src[internal])&set(tgt[internal])); selected=cfg["selected_heldout_lines"].get(key,eligible)
        selected=[int(x) for x in selected if int(x) in eligible]
        for heldout in selected: rows.append(run_fold(key,tc,src,tgt,internal,features,heldout,cfg,outdir))
        pd.DataFrame(rows).to_csv(outdir/"all_transitions_selected_LOLO_summary.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
