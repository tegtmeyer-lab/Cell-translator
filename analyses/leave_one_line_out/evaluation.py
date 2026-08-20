"""Evaluation metrics for leave-one-line-out Cell Painting validation."""
from __future__ import annotations
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-8


def cohens_d_featurewise(source, target, eps=1e-8):
    source = np.asarray(source, np.float32); target = np.asarray(target, np.float32)
    n1, n2 = source.shape[0], target.shape[0]
    if n1 < 2 or n2 < 2: raise ValueError("Cohen's d requires >=2 cells/domain")
    pooled = ((n1-1)*source.var(0, ddof=1) + (n2-1)*target.var(0, ddof=1))/(n1+n2-2)
    return (target.mean(0)-source.mean(0))/np.sqrt(np.maximum(pooled, eps))


def corr_and_sign(x, y):
    x=np.asarray(x); y=np.asarray(y); m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<2: return np.nan,np.nan,np.nan,np.nan,np.nan,len(x)
    if np.std(x)==0 or np.std(y)==0: pr=pp=np.nan
    else: pr,pp=pearsonr(x,y)
    sr,sp=spearmanr(x,y)
    sign=100*np.mean(np.sign(x)==np.sign(y))
    return float(pr),float(pp),float(sr),float(sp),float(sign),len(x)


def pca_variance_error(measured, predicted, n_components=50, log_base="e", eps=1e-8):
    """PCA fitted on measured target; lower is better, 0 is perfect."""
    measured=np.asarray(measured,np.float32); predicted=np.asarray(predicted,np.float32)
    n=min(n_components, measured.shape[0]-1, measured.shape[1])
    if n<1: return np.nan
    pca=PCA(n_components=n, random_state=42)
    zr=pca.fit_transform(measured); zp=pca.transform(predicted)
    ratio=(np.var(zp,0,ddof=1)+eps)/(np.var(zr,0,ddof=1)+eps)
    if log_base == 2 or str(log_base).lower() == "2": err=np.abs(np.log2(ratio))
    else: err=np.abs(np.log(ratio))
    return float(np.median(err))


def balanced_subsample(a,b,max_rows=1000,seed=42):
    n=min(len(a),len(b),max_rows); rng=np.random.default_rng(seed)
    ia=rng.choice(len(a),n,replace=False) if len(a)>n else np.arange(len(a))
    ib=rng.choice(len(b),n,replace=False) if len(b)>n else np.arange(len(b))
    return a[ia], b[ib]


def discriminator_auc_distance(measured,predicted,max_cells=1000,seed=42):
    n=min(len(measured),len(predicted),max_cells)
    if n<30: return np.nan,np.nan
    m,p=balanced_subsample(measured,predicted,n,seed)
    X=np.vstack([m,p]).astype(np.float32); y=np.r_[np.zeros(n,int),np.ones(n,int)]
    n_splits=min(5,n); train_n=int(len(X)*(n_splits-1)/n_splits)
    n_pc=min(30,X.shape[1],train_n-1)
    pipe=Pipeline([("scaler",StandardScaler()),("pca",PCA(n_components=n_pc,random_state=seed)),
                   ("clf",LogisticRegression(max_iter=3000,class_weight="balanced",solver="lbfgs",random_state=seed))])
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    scores=cross_val_score(pipe,X,y,cv=cv,scoring="roc_auc",n_jobs=1)
    auc=float(np.mean(scores)); return auc,float(abs(auc-0.5))


def evaluate_fold_arrays(source, measured, predicted, ks_max_cells=1000, pca_components=50,
                         pca_log_base=2, seed=42):
    source=np.asarray(source,np.float32); measured=np.asarray(measured,np.float32); predicted=np.asarray(predicted,np.float32)
    if not (source.ndim==measured.ndim==predicted.ndim==2): raise ValueError("All matrices must be 2D")
    if not (source.shape[1]==measured.shape[1]==predicted.shape[1]): raise ValueError("Feature mismatch")
    d_real=cohens_d_featurewise(source,measured); d_pred=cohens_d_featurewise(source,predicted)
    pr,pp,sr,sp,sign,nf=corr_and_sign(d_real,d_pred)
    m,p=balanced_subsample(measured,predicted,ks_max_cells,seed)
    ks=np.array([ks_2samp(m[:,j],p[:,j]).statistic for j in range(m.shape[1])])
    auc,aucd=discriminator_auc_distance(measured,predicted,max_cells=1000,seed=seed)
    return {
        "effect_size_pearson_r":pr,"effect_size_pearson_p":pp,
        "effect_size_spearman_r":sr,"effect_size_spearman_p":sp,
        "effect_size_sign_agreement_percent":sign,"n_effect_features":nf,
        "median_ks":float(np.median(ks)),"mean_ks":float(np.mean(ks)),
        "median_abs_log_pca_variance_ratio":pca_variance_error(measured,predicted,pca_components,pca_log_base),
        "discriminator_auc":auc,"auc_distance_from_0p5":aucd,
        "feature_mean_pearson_r":float(pearsonr(measured.mean(0),predicted.mean(0))[0]),
        "feature_std_pearson_r":float(pearsonr(measured.std(0,ddof=1),predicted.std(0,ddof=1))[0]),
    }


def get_fold_matrix_paths(fold_dir):
    """Support both final generic filenames and legacy stem/neuron aliases."""
    fold=Path(fold_dir)
    generic=(fold/"heldout_source_original.npy", fold/"heldout_measured_target_original.npy", fold/"predicted_target_from_heldout_source.npy")
    if all(p.exists() for p in generic): return generic
    source=[p for p in fold.glob("heldout_*_original.npy") if "measured" not in p.name.lower()]
    measured=list(fold.glob("heldout_measured_*_original.npy")); pred=list(fold.glob("predicted_*_from_heldout_*.npy"))
    if len(source)!=1 or len(measured)!=1 or len(pred)!=1:
        raise FileNotFoundError(f"Could not uniquely identify fold matrices in {fold}")
    return source[0],measured[0],pred[0]


def infer_line(fold):
    m=re.search(r"heldout[_-]?line[_-]?([A-Za-z0-9.-]+)",Path(fold).name,re.I)
    return m.group(1) if m else Path(fold).name


def impute_nonfinite(X):
    X=np.asarray(X,np.float32).copy()
    if np.isfinite(X).all(): return X
    warnings.warn("Non-finite values found; imputing each matrix with its own column median")
    Xm=X.astype(float); med=np.nanmedian(np.where(np.isfinite(Xm),Xm,np.nan),axis=0)
    med[~np.isfinite(med)]=0.0; bad=~np.isfinite(Xm); Xm[bad]=np.take(med,np.where(bad)[1])
    return Xm.astype(np.float32)


def evaluate_archived_transition(root, transition, selected_lines=None, pca_log_base=2):
    root=Path(root)/transition
    folds=sorted([p for p in root.glob("heldout_line_*") if p.is_dir()], key=lambda p:p.name)
    rows=[]
    for fold in folds:
        line=infer_line(fold)
        try: numeric=int(float(line))
        except Exception: numeric=line
        if selected_lines is not None and numeric not in selected_lines: continue
        sp,mp,pp=get_fold_matrix_paths(fold)
        source,measured,predicted=map(lambda p: impute_nonfinite(np.load(p)),(sp,mp,pp))
        metrics=evaluate_fold_arrays(source,measured,predicted,pca_log_base=pca_log_base)
        rows.append({"transition":transition,"heldout_line":numeric,"n_features":source.shape[1],**metrics})
    return pd.DataFrame(rows)
