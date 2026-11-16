"""Cell Translator morphology: top-K feature-ranking recovery.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_ROOT = Path(os.environ.get('CELL_TRANSLATOR_RUN_DIR', str(Path.cwd() / 'outputs' / 'run_70_15_15_qc_v3'))).expanduser().resolve()
AE_DIR = RUN_ROOT / 'autoencoders'
DEV_DIR = RUN_ROOT / 'cyclegan_ipsc_astrocyte_drugs'
CTRL_DIR = RUN_ROOT / 'cyclegan_control_to_perturbation'
EFFECT_DIR = RUN_ROOT / 'evaluation' / 'category_effect_ig'
OUT_DIR = RUN_ROOT / 'evaluation' / 'top_feature_recovery_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRUGS = ['Latrunculin_B', 'GW3965', 'Clozapine']
STRATEGIES = ['iPSC_control_to_drug', 'Astrocyte_control_to_drug', 'developmental']
LABELS = {
 'iPSC_control_to_drug': 'iPSC DMSO → drug',
 'Astrocyte_control_to_drug': 'Astrocyte DMSO → drug',
 'developmental': 'Drug iPSC → astrocyte',
}
TOP_KS = (20, 50, 100)
COLORS = {'iPSC_control_to_drug':'#587e9d', 'Astrocyte_control_to_drug':'#d28c59', 'developmental':'#608b79'}

def original_test(condition):
    prefix = AE_DIR / condition
    stats = pd.read_csv(str(prefix) + '_scaler_stats.csv')
    X = np.load(str(prefix) + '_X_test_scaled.npy', allow_pickle=False)
    assert X.ndim == 2 and X.shape[1] == len(stats), f'{condition}: dimension mismatch'
    if stats['feature'].duplicated().any(): raise ValueError(f'{condition}: duplicated features')
    X = X.astype(np.float64) * stats['scaler_scale'].to_numpy(dtype=float) + stats['scaler_mean'].to_numpy(dtype=float)
    return pd.DataFrame(X, columns=stats.feature.astype(str))


def predicted_control(cell_type, drug):
    folder = CTRL_DIR / f'{cell_type}_DMSO_to_{drug}'
    X = np.load(folder / 'X_control_to_drug_original.npy', allow_pickle=False)
    features = pd.read_csv(folder / 'control_to_drug_feature_names.csv')['feature'].astype(str)
    if X.ndim != 2 or X.shape[1] != len(features) or features.duplicated().any():
        raise ValueError(f'Bad predicted control-to-drug features: {folder}')
    return pd.DataFrame(X, columns=features)


def predicted_development(drug):
    folder = DEV_DIR / drug
    X = np.load(folder / 'translated_astro_morphology_original.npy', allow_pickle=False)
    features = pd.read_csv(folder / 'translated_astro_feature_names.csv')['feature'].astype(str)
    if X.ndim != 2 or X.shape[1] != len(features) or features.duplicated().any():
        raise ValueError(f'Bad developmental features: {folder}')
    return pd.DataFrame(X, columns=features)


def cohens_d(a, b, min_sd=1e-8):
    """Signed (mean(b)-mean(a)) / pooled sample SD, one value per feature."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(len(a),len(b)) < 2:
        raise ValueError('Expected 2D matrices with equal feature counts and >=2 cells')
    n0,n1=len(a),len(b)
    pooled = np.sqrt(((n0-1)*np.var(a,axis=0,ddof=1) + (n1-1)*np.var(b,axis=0,ddof=1))/(n0+n1-2))
    out = np.full(a.shape[1], np.nan)
    ok = np.isfinite(pooled)&(pooled>min_sd)
    out[ok] = (np.mean(b,axis=0)[ok]-np.mean(a,axis=0)[ok])/pooled[ok]
    return out


def effect_table(control_real, drug_real, control_pred, drug_pred, label):
    mats = (control_real,drug_real,control_pred,drug_pred)
    for m in mats:
        if m.columns.duplicated().any(): raise ValueError(f'{label}: duplicated features')
    common = [f for f in drug_real.columns if all(f in m.columns for m in mats)]
    if len(common) < 100: raise ValueError(f'{label}: only {len(common)} common features')
    r0,r1,p0,p1 = (m.loc[:,common].to_numpy(dtype=float) for m in mats)
    for mat in (r0,r1,p0,p1):
        if not np.isfinite(mat).all(): raise ValueError(f'{label}: nonfinite values')
    measured = cohens_d(r0,r1)
    predicted = cohens_d(p0,p1)
    result = pd.DataFrame({'feature':common,'d_measured':measured,'d_predicted':predicted})
    result = result.replace([np.inf,-np.inf],np.nan).dropna(subset=['d_measured','d_predicted'])
    if len(result) < max(TOP_KS): raise ValueError(f'{label}: fewer than {max(TOP_KS)} valid features')
    result['abs_d_measured'] = result.d_measured.abs()
    result['abs_d_predicted'] = result.d_predicted.abs()
    return result


def load_effect(strategy, drug):

    if strategy in ('Astrocyte_control_to_drug','developmental'):
        label = 'developmental' if strategy=='developmental' else 'control_to_drug'
        file = EFFECT_DIR / f'{label}__{drug}_cohens_d_features.csv'
        if file.exists():
            t = pd.read_csv(file)
            pred_col = 'd_predicted_own_pooled_sd' if 'd_predicted_own_pooled_sd' in t else 'd_predicted'
            req = ['feature','d_measured',pred_col]
            if not set(req).issubset(t.columns): raise ValueError(f'{file} needs {req}')
            tab = t[req].rename(columns={pred_col:'d_predicted'}).dropna().copy()
            if tab.feature.duplicated().any(): raise ValueError(f'Duplicate features in {file}')
            tab['abs_d_measured']=tab.d_measured.abs()
            tab['abs_d_predicted']=tab.d_predicted.abs()
            return tab
    if strategy == 'iPSC_control_to_drug':
        return effect_table(original_test('iPSC_DMSO'),original_test('iPSC_'+drug),
                            original_test('iPSC_DMSO'),predicted_control('iPSC',drug), strategy+'_'+drug)
    if strategy == 'Astrocyte_control_to_drug':
        return effect_table(original_test('Astrocytes_DMSO'),original_test('Astrocytes_'+drug),
                            original_test('Astrocytes_DMSO'),predicted_control('Astrocytes',drug),strategy+'_'+drug)
    if strategy == 'developmental':
        return effect_table(original_test('Astrocytes_DMSO'),original_test('Astrocytes_'+drug),
                            predicted_development('DMSO'),predicted_development(drug),strategy+'_'+drug)
    raise ValueError(strategy)


def recover_top_k(tab, k):
    measured = set(tab.nlargest(k,'abs_d_measured').feature)
    predicted = set(tab.nlargest(k,'abs_d_predicted').feature)
    n = len(measured & predicted)
    return {'overlap':n,'recall_at_k':n/k,'jaccard':n/(2*k-n),
            'expected_random_overlap': k*k/len(tab), 'n_features':len(tab)}


def save(fig, stem):
    fig.savefig(OUT_DIR / (stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT_DIR / (stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def plot_top100(summary):
    d = summary[summary.top_k==100]
    fig, ax=plt.subplots(figsize=(8.2,4.4))
    x=np.arange(3); width=.24
    for i,s in enumerate(STRATEGIES):
        a=d.set_index(['strategy','drug']).loc[[(s,drug) for drug in DRUGS]]
        vals=a.recall_at_k.to_numpy()*100
        xpos=x+(i-1)*width
        bars=ax.bar(xpos,vals,width,color=COLORS[s],label=LABELS[s])
        for bar,n in zip(bars,a.overlap):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.3,str(int(n)),ha='center',fontsize=9)
    ax.set_xticks(x,['Latrunculin B','GW3965','Clozapine'])
    ax.set_ylim(0,105);ax.set_ylabel('Top-100 recovery (%)')
    ax.set_title('Recovery of the 100 strongest drug-associated morphology features',pad=14)
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'top100_recovery_three_strategies')


def plot_topk(summary):
    fig, axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    x=np.arange(3); width=.23
    for ax,k in zip(axes,TOP_KS):
        sub=summary[summary.top_k==k]
        for i,s in enumerate(STRATEGIES):
            a=sub.set_index(['strategy','drug']).loc[[(s,drug) for drug in DRUGS]]
            bars=ax.bar(x+(i-1)*width,a.recall_at_k.to_numpy()*100,width,color=COLORS[s],label=LABELS[s])
            for bar,n in zip(bars,a.overlap):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,str(int(n)),ha='center',fontsize=7)
        ax.set_title(f'Top {k}');ax.set_xticks(x,['Latrunculin B','GW3965','Clozapine'],rotation=22,ha='right')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        ax.set_ylim(0,108)
    axes[0].set_ylabel('Recall@K (%)')
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.03),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'topK_recovery_three_strategies')


def dmso_developmental_recovery():
    src=original_test('iPSC_DMSO')
    tgt=original_test('Astrocytes_DMSO')
    pred=predicted_development('DMSO')
    tab=effect_table(src,tgt,src,pred,'DMSO developmental differentiation')
    rows=[]
    for k in TOP_KS:
        rows.append({'comparison':'DMSO_iPSC_to_astrocyte_differentiation', 'top_k':k,**recover_top_k(tab,k)})
    pd.DataFrame(rows).to_csv(OUT_DIR/'DMSO_developmental_differentiation_topK.csv',index=False)
    tab.to_csv(OUT_DIR/'DMSO_developmental_differentiation_effects.csv',index=False)
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    ax.bar([str(k) for k in TOP_KS],[r['recall_at_k']*100 for r in rows],color=COLORS['developmental'])
    for i,r in enumerate(rows):ax.text(i,r['recall_at_k']*100+1.5,str(r['overlap']),ha='center')
    ax.set_ylim(0,105);ax.set_ylabel('Top-K recovery (%)');ax.set_xlabel('Top K')
    ax.set_title('DMSO: iPSC → astrocyte differentiation (not a drug effect)')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save(fig,'DMSO_developmental_differentiation_top100')
    return pd.DataFrame(rows)


def main():
    rows=[]
    for s in STRATEGIES:
        for drug in DRUGS:
            tab=load_effect(s,drug)
            for k in TOP_KS:
                rows.append({'strategy':s,'drug':drug,'top_k':k,**recover_top_k(tab,k)})
            print(f'{LABELS[s]:27s} {drug:16s} top100={rows[-1]["overlap"]:3d}/100, features={len(tab)}')
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT_DIR/'topK_recovery_summary.csv',index=False)
    summary[summary.top_k==100].to_csv(OUT_DIR/'top100_recovery_summary.csv',index=False)
    plot_top100(summary)
    plot_topk(summary)
    try:
        dmso=dmso_developmental_recovery()
        print('\nDMSO developmental differentiation (separate endpoint):')
        print(dmso[['top_k','overlap','recall_at_k']].to_string(index=False))
    except FileNotFoundError as exc:
        print('DMSO differentiation plot skipped: missing',exc)
    print('\nSaved figures and tables under:',OUT_DIR)
    return summary

if __name__=='__main__':
    topk_results=main()
