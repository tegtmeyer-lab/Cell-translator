"""Morphology Cell Translator: developmental and drug perturbation vectors.

"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

BASE_DIR = Path(os.environ.get('CELL_TRANSLATOR_DATA_DIR', str(Path.cwd() / 'data'))).expanduser().resolve()
RUN_ROOT = Path(os.environ.get('CELL_TRANSLATOR_RUN_DIR', str(Path.cwd() / 'outputs' / 'run_70_15_15_qc_v3'))).expanduser().resolve()  
AE_DIR = RUN_ROOT / 'autoencoders'
DEV_DIR = RUN_ROOT / 'cyclegan_ipsc_astrocyte_drugs'
CTRL_DIR = RUN_ROOT / 'cyclegan_control_to_perturbation'
OUT = RUN_ROOT / 'evaluation' / 'morphology_transition_vectors'
DRUGS = ['Latrunculin_B','GW3965','Clozapine']
CONDITIONS = ['DMSO'] + DRUGS
SEED = 42
MAX_CELLS_PER_GROUP = 800
SOURCE, TARGET, PRED = '#7E8D9C', '#287D79', '#C46A52'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13,
                     'axes.labelsize':15, 'axes.titlesize':16,
                     'xtick.labelsize':13, 'ytick.labelsize':13,
                     'legend.fontsize':13,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.facecolor':'white'})


def _check_frame(array, features, title):
    a = np.asarray(array)
    cols = list(map(str, features))
    if a.ndim != 2 or len(a) < 2 or a.shape[1] != len(cols):
        raise ValueError(f'{title}: invalid shape {a.shape} for {len(cols)} names')
    if len(set(cols)) != len(cols):
        raise ValueError(f'{title}: duplicate features')
    if not np.isfinite(a).all():
        raise ValueError(f'{title}: non-finite values')
    return pd.DataFrame(a.astype(np.float32, copy=False),columns=cols)


def load_real(condition, split='test'):
    prefix=AE_DIR / condition
    stats=pd.read_csv(str(prefix)+'_scaler_stats.csv')
    X=np.load(str(prefix)+f'_X_{split}_scaled.npy')
    names=stats['feature'].astype(str).tolist()
    original=X.astype(np.float64)*stats['scaler_scale'].to_numpy()[None,:]+stats['scaler_mean'].to_numpy()[None,:]
    return _check_frame(original,names,f'{condition}/{split}')


def load_developmental_prediction(drug):
    folder=DEV_DIR / drug
    X=np.load(folder/'translated_astro_morphology_original.npy')
    cols=pd.read_csv(folder/'translated_astro_feature_names.csv')['feature'].astype(str).tolist()
    return _check_frame(X,cols,f'developmental/{drug}')


def load_control_to_drug_prediction(cell_type,drug):
    folder=CTRL_DIR / f'{cell_type}_DMSO_to_{drug}'
    X=np.load(folder/'X_control_to_drug_original.npy')
    cols=pd.read_csv(folder/'control_to_drug_feature_names.csv')['feature'].astype(str).tolist()
    return _check_frame(X,cols,f'{cell_type}/DMSO_to_{drug}')


def align_frames(frames):
    # Retain source feature order, not alphabetical, and never compare columns by position alone.
    keys=list(frames)
    first=frames[keys[0]]
    common=[c for c in first.columns if all(c in frames[k].columns for k in keys[1:])]
    if len(common)<3:
        raise ValueError('Fewer than three shared named morphology features')
    return {k:frames[k].loc[:,common] for k in keys},common


def sample(a,n,rng):
    ix=rng.choice(len(a),size=min(n,len(a)),replace=False)
    return a[ix]


def prepare_space(source_train,target_train,frames,seed=SEED):
    # Reference standardization is fitted on measured TRAIN only (no leakage or predicted data).
    everything={'source_training':source_train,'target_training':target_train,**frames}
    aligned,features=align_frames(everything)
    rng=np.random.default_rng(seed)
    n=min(MAX_CELLS_PER_GROUP,len(aligned['source_training']),len(aligned['target_training']))
    source_ref=sample(aligned['source_training'].to_numpy(np.float64),n,rng)
    target_ref=sample(aligned['target_training'].to_numpy(np.float64),n,rng)
    reference=np.vstack([source_ref,target_ref])
    mean=reference.mean(axis=0)
    scale=reference.std(axis=0,ddof=0)
    good=np.isfinite(scale)&(scale>1e-8)
    if good.sum()<3:
        raise ValueError('Too few variable reference features')
    features=np.asarray(features)[good].tolist()
    scaled={key:(frame.loc[:,features].to_numpy(np.float64)-mean[good])/scale[good]
            for key,frame in aligned.items()}
    pca=PCA(n_components=2,svd_solver='randomized',random_state=seed)
    pca.fit((reference[:,good]-mean[good])/scale[good])
    return scaled,pca,features


def vector_metrics(space,condition,kind):
    mu={k:x.mean(axis=0) for k,x in space.items() if k in ('source','target','predicted')}
    real=mu['target']-mu['source']; pred=mu['predicted']-mu['source']
    nr=np.linalg.norm(real); npred=np.linalg.norm(pred)
    cosine=float(np.dot(real,pred)/(nr*npred)) if nr>1e-12 and npred>1e-12 else np.nan
    cosine=np.clip(cosine,-1,1) if np.isfinite(cosine) else np.nan
    endpoint=np.linalg.norm(mu['target']-mu['predicted'])
    return dict(analysis=kind,condition=condition,representation='shared training-reference standardized decoded morphology',
                n_features=len(real),n_source=len(space['source']),n_target=len(space['target']),
                n_predicted=len(space['predicted']),cosine_similarity=cosine,
                angle_degrees=float(np.degrees(np.arccos(cosine))) if np.isfinite(cosine) else np.nan,
                magnitude_ratio=float(npred/nr) if nr>1e-12 else np.nan,
                relative_endpoint_error=float(endpoint/nr) if nr>1e-12 else np.nan,
                endpoint_mae=float(np.mean(np.abs(mu['target']-mu['predicted']))),
                measured_vector_norm=float(nr),predicted_vector_norm=float(npred))


def plot_panel(ax,space,pca,metrics,title,labels=('Source','Measured target','Translated target'),seed=SEED):
    rng=np.random.default_rng(seed)
    centers={k:pca.transform(x.mean(axis=0,keepdims=True))[0] for k,x in space.items()
             if k in ('source','target','predicted')}
    for key,col in [('source',SOURCE),('target',TARGET),('predicted',PRED)]:
        xy=pca.transform(sample(space[key],MAX_CELLS_PER_GROUP,rng))
        ax.scatter(xy[:,0],xy[:,1],s=9,alpha=.70,color=col,lw=0,rasterized=True,zorder=2)
    src=centers['source']; tar=centers['target']; pr=centers['predicted']
    for point,col in [(tar,TARGET),(pr,PRED)]:
        ax.annotate('',xy=point,xytext=src,arrowprops=dict(arrowstyle='-|>',lw=2.8,color=col,mutation_scale=14))
    ax.plot([tar[0],pr[0]],[tar[1],pr[1]],ls='--',lw=1.0,c='#888888')
    for key,col,mark in [('source',SOURCE,'o'),('target',TARGET,'D'),('predicted',PRED,'D')]:
        ax.scatter(*centers[key],s=85,c=col,marker=mark,edgecolor='white',lw=1.1,zorder=8)
    e=pca.explained_variance_ratio_*100
    ax.set(title=title,xlabel=f'PC1 ({e[0]:.1f}%)',ylabel=f'PC2 ({e[1]:.1f}%)')
    ax.grid(lw=.5,alpha=.2)
    ax.text(.02,.98,f'Cosine = {metrics["cosine_similarity"]:.2f}\n'
                    f'Angle = {metrics["angle_degrees"]:.1f}°\n'
                    f'Magnitude ratio = {metrics["magnitude_ratio"]:.2f}',
            transform=ax.transAxes,ha='left',va='top',fontsize=13,
            bbox=dict(facecolor='white',edgecolor='lightgray',alpha=.9,boxstyle='round,pad=.25'))


def save_figure(fig,path):
    fig.savefig(str(path)+'.png',dpi=300,bbox_inches='tight')
    fig.savefig(str(path)+'.pdf',bbox_inches='tight')
    plt.close(fig)


def run_one(kind,drug,cell_type=None):
    if kind=='developmental':
        src=f'iPSC_{drug}'; tgt=f'Astrocytes_{drug}'
        pred=load_developmental_prediction(drug)
        labels=('Measured iPSC','Measured astrocyte','Translated astrocyte')
    elif kind=='control_to_drug':
        src=f'{cell_type}_DMSO'; tgt=f'{cell_type}_{drug}'
        pred=load_control_to_drug_prediction(cell_type,drug)
        labels=('Measured DMSO',f'Measured {drug}',f'Translated {drug}')
    else:raise ValueError(kind)
    frames=dict(source=load_real(src),target=load_real(tgt),predicted=pred)
    space,pca,features=prepare_space(load_real(src,'train'),load_real(tgt,'train'),frames)
    metrics=vector_metrics(space,drug,kind if cell_type is None else f'{kind}_{cell_type}')
    metrics['source_condition']=src; metrics['target_condition']=tgt
    stem=kind+'_'+(cell_type+'_' if cell_type else '')+drug
    fig,ax=plt.subplots(figsize=(5,5))
    plot_panel(ax,space,pca,metrics,drug.replace('_',' '),labels)
    fig.tight_layout();save_figure(fig,OUT/stem)
    pd.DataFrame({'feature':features,'measured_vector':space['target'].mean(0)-space['source'].mean(0),
                  'predicted_vector':space['predicted'].mean(0)-space['source'].mean(0)}).to_csv(OUT/(stem+'_vectors.csv'),index=False)
    return metrics,(space,pca,metrics)


def run_drug_effect_vectors(drug):
    # Separate experiment: drug effect in astrocytes, with measured and predicted
    # DMSO anchors. Primary metrics compare *differences of means* in full space.
    frames=dict(real_control=load_real('Astrocytes_DMSO'),
                real_drug=load_real('Astrocytes_'+drug),
                pred_control=load_developmental_prediction('DMSO'),
                pred_drug=load_developmental_prediction(drug),
                direct_drug=load_control_to_drug_prediction('Astrocytes',drug))
    source_train=load_real('Astrocytes_DMSO','train')
    target_train=load_real('Astrocytes_'+drug,'train')
    aligned,pca,features=prepare_space(source_train,target_train,frames)
    mu={key:val.mean(0) for key,val in aligned.items() if key in frames}
    actual=mu['real_drug']-mu['real_control']
    dev=mu['pred_drug']-mu['pred_control']
    direct=mu['direct_drug']-mu['real_control']
    def compare(pred,method):
        a=np.linalg.norm(actual);b=np.linalg.norm(pred)
        cosine=float(np.dot(actual,pred)/(a*b)) if a>1e-12 and b>1e-12 else np.nan
        cosine=np.clip(cosine,-1,1) if np.isfinite(cosine) else np.nan
        return {'drug':drug,'model':method,'n_features':len(features),
                'cosine_similarity':cosine,
                'angle_degrees':float(np.degrees(np.arccos(cosine))) if np.isfinite(cosine) else np.nan,
                'magnitude_ratio':float(b/a) if a>1e-12 else np.nan,
                'vector_mae':float(np.mean(np.abs(pred-actual)))}
    table=pd.DataFrame({'feature':features,'measured_drug_shift':actual,
                        'developmental_pred_shift':dev,'direct_ctrl_to_drug_shift':direct})
    table.to_csv(OUT/f'drug_effect_{drug}_vectors.csv',index=False)
    # Single PCA per drug, trained only on measured TRAIN control + drug.
    centers={key:pca.transform(value[None,:])[0] for key,value in mu.items()}
    fig,ax=plt.subplots(figsize=(5,5))
    for key,color in [('real_control',SOURCE),('real_drug',TARGET),('pred_control','#a48baf'),
                      ('pred_drug',PRED),('direct_drug','#b78c43')]:
        xy=pca.transform(sample(aligned[key],MAX_CELLS_PER_GROUP,np.random.default_rng(SEED)))
        ax.scatter(xy[:,0],xy[:,1],s=9,alpha=.70,color=color,rasterized=True,zorder=2)
        ax.scatter(*centers[key],s=75,color=color,zorder=8,edgecolors='white',label=key.replace('_',' '))
    for start,end,col in [('real_control','real_drug',TARGET),('pred_control','pred_drug',PRED),
                          ('real_control','direct_drug','#b78c43')]:
        ax.annotate('',xy=centers[end],xytext=centers[start],
                    arrowprops=dict(arrowstyle='-|>',lw=2,color=col,mutation_scale=12))
    evr=pca.explained_variance_ratio_*100
    ax.set(xlabel=f'PC1 ({evr[0]:.1f}%)',ylabel=f'PC2 ({evr[1]:.1f}%)',
           title=f'Astrocyte drug effects — {drug.replace("_"," ")}')
    ax.legend(loc='best',fontsize=13,frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout();save_figure(fig,OUT/f'drug_effect_{drug}')
    return [compare(dev,'developmental'),compare(direct,'control_to_drug')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    main_rows=[]; panels={'developmental':[], 'control_to_drug_iPSC':[], 'control_to_drug_Astrocytes':[]}
    for drug in CONDITIONS:
        row,info=run_one('developmental',drug)
        main_rows.append(row);panels['developmental'].append((drug,info))
    for cell_type in ['iPSC','Astrocytes']:
        for drug in DRUGS:
            row,info=run_one('control_to_drug',drug,cell_type)
            main_rows.append(row);panels['control_to_drug_'+cell_type].append((drug,info))
    pd.DataFrame(main_rows).to_csv(OUT/'transition_vector_metrics.csv',index=False)
    for group,entries in panels.items():
        fig,axes=plt.subplots(1,len(entries),figsize=(4*len(entries),4),squeeze=False)
        for ax,(drug,(space,pca,metrics)) in zip(axes.ravel(),entries):
            plot_panel(ax,space,pca,metrics,drug.replace('_',' '))
        fig.tight_layout();save_figure(fig,OUT/f'combined_{group}')
    effect_rows=[]
    for drug in DRUGS:
        effect_rows.extend(run_drug_effect_vectors(drug))
    pd.DataFrame(effect_rows).to_csv(OUT/'astrocyte_drug_effect_vector_metrics.csv',index=False)
    print('Saved outputs to',OUT)
    return pd.DataFrame(main_rows),pd.DataFrame(effect_rows)

if __name__=='__main__':
    main()
