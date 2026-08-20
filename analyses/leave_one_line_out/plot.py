#!/usr/bin/env python
"""Plot the final five-line LOLO validation summary."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDER=["stem_to_npc","stem_to_neuron","stem_to_astrocyte","npc_to_neuron","npc_to_astrocyte"]
LABELS={"stem_to_npc":"Stem→NPC","stem_to_neuron":"Stem→Neuron","stem_to_astrocyte":"Stem→Astrocyte","npc_to_neuron":"NPC→Neuron","npc_to_astrocyte":"NPC→Astrocyte"}
METRICS=[("effect_size_pearson_r","A   Developmental effect recovery","Effect-size Pearson r"),
         ("median_ks","B   Feature-level distributional agreement","Median feature-wise KS"),
         ("median_abs_log_pca_variance_ratio","C   PCA variance recovery","Median |log2 variance ratio|")]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",required=True); p.add_argument("--outdir",required=True); args=p.parse_args()
    df=pd.read_csv(args.results); order=[x for x in ORDER if x in df.transition.unique()]
    fig,axes=plt.subplots(1,3,figsize=(12,5),dpi=200)
    for ax,(metric,title,ylabel) in zip(axes,METRICS):
        for x,t in enumerate(order):
            tmp=df[df.transition.eq(t)].copy().sort_values("heldout_line")
            vals=tmp[metric].to_numpy(float); jitter=np.linspace(-0.12,0.12,len(vals)) if len(vals)>1 else np.zeros(len(vals))
            ax.scatter(np.full(len(vals),x)+jitter,vals,s=30)
            med=np.nanmedian(vals); ax.plot([x-0.22,x+0.22],[med,med],lw=2)
            for xx,yy,line in zip(np.full(len(vals),x)+jitter,vals,tmp.heldout_line): ax.text(xx,yy,str(int(line)),fontsize=7,ha="center",va="bottom")
        ax.set_xticks(range(len(order)),[LABELS[t] for t in order],rotation=30,ha="right"); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.25)
        if metric=="effect_size_pearson_r": ax.axhline(0,ls="--",lw=1)
    fig.tight_layout(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/"lodo_five_line_summary.pdf",bbox_inches="tight"); fig.savefig(out/"lodo_five_line_summary.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    print(f"Saved figures to {out}")
if __name__=="__main__": main()
