
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


RUN_ROOT = Path(os.environ.get('CELL_TRANSLATOR_RUN_DIR', str(Path.cwd() / 'outputs' / 'run_70_15_15_qc_v3'))).expanduser().resolve()
IN_DIR = RUN_ROOT / 'evaluation' / 'category_effect_ig'
OUT_DIR = IN_DIR / 'combined_panels'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Edit these to resize the final exported figures.
FIGSIZE_GROUPED = (18, 10)  
FIGSIZE_IG = (13, 8)       
FIGSIZE_SIGNED = (15, 7)   
DPI = 300

DRUGS = ['Latrunculin_B', 'GW3965', 'Clozapine']
DRUG_LABELS = ['Latrunculin B', 'GW3965', 'Clozapine']
STRATEGIES = ['developmental', 'control_to_drug']
STRATEGY_LABELS = [
    'Perturbed iPSC → perturbed astrocyte',
    'Astrocyte DMSO → drug',
]
FAMILIES = [
    'AreaShape', 'Granularity', 'RadialDistribution',
    'Texture', 'Intensity', 'Correlation',
]
COLORS = {'measured_abs': '#477b9e', 'predicted_abs': '#df8847'}
Q_CUTOFF = 0.05

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})


def load_enrichment(filename):
    path = IN_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}. Run the category-effect/IG notebook first.'
        )
    tab = pd.read_csv(path)
    required = {'analysis', 'family', 'obs_exp', 'q_direction'}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f'{filename}: missing {sorted(missing)}')
    if tab.duplicated(['analysis', 'family']).any():
        raise ValueError(f'{filename}: duplicated analysis-family rows')
    return tab


def get_family_rows(tab, analysis):
    sub = tab.loc[tab['analysis'].eq(analysis)].set_index('family')
    if sub.empty:
        raise ValueError(f'No rows for analysis {analysis!r}')
    absent = set(FAMILIES) - set(sub.index)
    if absent:
        raise ValueError(f'{analysis}: missing families {sorted(absent)}')
    return sub.reindex(FAMILIES)


def save_fig(fig, stem):
    """Preserve exact figsize × DPI; never use bbox_inches='tight'."""
    fig.set_facecolor('white')
    png = OUT_DIR / f'{stem}.png'
    pdf = OUT_DIR / f'{stem}.pdf'
    fig.savefig(png, dpi=DPI, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    expected = tuple(np.rint(fig.get_size_inches() * DPI).astype(int))
    try:
        from PIL import Image
        with Image.open(png) as im:
            actual = im.size
        print(f'Saved {png.name} | pixels: {actual} | expected: {expected}')
    except ImportError:
        print(f'Saved {png.name} | expected pixels: {expected}')
    print(f'Saved {pdf.name}')
    plt.close(fig)


def plot_grouped_effects(tab):
    """2 × 3 grouped bars: measured vs predicted O/E per morphology family."""
    prepared = {}
    xmax = 1.0
    for strategy in STRATEGIES:
        for drug in DRUGS:
            for kind in ('measured_abs', 'predicted_abs'):
                sub = get_family_rows(tab, f'{strategy}__{drug}__{kind}')
                prepared[(strategy, drug, kind)] = sub
                xmax = max(xmax, float(np.nanmax(sub['obs_exp'])))
    xmax = np.ceil((xmax + 0.28) * 2) / 2

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GROUPED, sharex=True, sharey=True)
    y = np.arange(len(FAMILIES))
    height = 0.35

    for i, strategy in enumerate(STRATEGIES):
        for j, drug in enumerate(DRUGS):
            ax = axes[i, j]
            for offset, kind in [(-height / 2, 'measured_abs'),
                                 (height / 2, 'predicted_abs')]:
                sub = prepared[(strategy, drug, kind)]
                values = sub['obs_exp'].to_numpy(float)
                qvals = sub['q_direction'].to_numpy(float)
                yy = y + offset
                ax.barh(yy, values, height=height, color=COLORS[kind],
                        alpha=1.0, edgecolor='none', zorder=2)
                for yp, value, q in zip(yy, values, qvals):
                    if np.isfinite(value) and np.isfinite(q) and q < Q_CUTOFF:
                        ax.text(value + 0.035, yp, '*', fontsize=15,
                                ha='left', va='center')

            ax.axvline(1, color='#424242', lw=1.1, ls='--', zorder=3)
            ax.set_title(DRUG_LABELS[j], fontsize=16, pad=12)
            ax.set_xlim(0, xmax)
            ax.set_yticks(y)
            ax.set_yticklabels(FAMILIES)
            ax.invert_yaxis()
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(axis='x', lw=0.6, alpha=0.15)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(STRATEGY_LABELS[i] + '\nMorphology family',
                              fontsize=13, labelpad=13)
            if i == 1:
                ax.set_xlabel('Observed / expected representation', fontsize=13)

    fig.suptitle('Morphology-family representation among strongest drug effects',
                 fontsize=19, y=0.975)
    fig.legend(
        handles=[
            Patch(color=COLORS['measured_abs'], label='Measured top 10%'),
            Patch(color=COLORS['predicted_abs'], label='Predicted top 10%'),
        ],
        loc='upper center', bbox_to_anchor=(0.5, 0.925),
        ncol=2, frameon=False, fontsize=14,
    )
    fig.text(0.5, 0.03,
             'Dashed line: expected representation (O/E = 1). '
             '* q < 0.05 (direction-specific BH-FDR).',
             ha='center', fontsize=12)
    # Manual layout instead of tight_layout; these are fractions of the fixed canvas.
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.12,
                        top=0.82, wspace=0.20, hspace=0.29)
    save_fig(fig, 'Figure_effect_family_enrichment_grouped_2x3')


def heatmap_signed_log2(ax, tab, analyses, labels, title, max_abs=None,
                        annotation_fontsize=11):
    """Show log2(O/E); significance based on existing direction-specific q values."""
    vals, qs = [], []
    for analysis in analyses:
        sub = get_family_rows(tab, analysis)
        vals.append(np.log2(sub['obs_exp'].clip(lower=1e-9).to_numpy(float)))
        qs.append(sub['q_direction'].to_numpy(float))
    vals, qs = np.asarray(vals), np.asarray(qs)
    if max_abs is None:
        max_abs = max(0.5, float(np.nanmax(np.abs(vals))))

    img = ax.imshow(vals, aspect='auto', cmap='RdBu_r',
                    vmin=-max_abs, vmax=max_abs, interpolation='nearest')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xticks(np.arange(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, rotation=45, ha='right', fontsize=12)
    ax.set_title(title, fontsize=16, pad=13)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            value, q = vals[i, j], qs[i, j]
            mark = '*' if np.isfinite(q) and q < Q_CUTOFF else ''
            label = f'{value:+.2f}{mark}' if np.isfinite(value) else 'NA'
            ink = 'white' if np.isfinite(value) and abs(value) > 0.65 * max_abs else '#202020'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=annotation_fontsize, color=ink)
    return img


def plot_ig_heatmap(tab):
    """9 conditions: 3 drugs × developmental/iPSC-direct/astrocyte-direct."""
    analyses, labels = [], []
    for drug, label in zip(DRUGS, DRUG_LABELS):
        for strategy, cell_type, short in [
            ('developmental', 'iPSC', 'Developmental'),
            ('control_to_drug', 'iPSC', 'iPSC DMSO → drug'),
            ('control_to_drug', 'Astrocytes', 'Astro DMSO → drug'),
        ]:
            analyses.append(f'{strategy}__{cell_type}__{drug}__IG_source')
            labels.append(f'{label} | {short}')

    values = [
        np.log2(get_family_rows(tab, analysis)['obs_exp']
                .clip(lower=1e-9).to_numpy(float))
        for analysis in analyses
    ]
    max_abs = max(0.5, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=FIGSIZE_IG)
    img = heatmap_signed_log2(
        ax, tab, analyses, labels,
        'Source morphology-family IG enrichment (top 10% absolute IG)',
        max_abs=max_abs,
        annotation_fontsize=12,
    )
    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label('log₂(observed / expected)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.text(0.5, 0.035,
             '* q < 0.05. IG attributes source features; '
             'it is not the signed drug effect.',
             ha='center', fontsize=12)
    fig.subplots_adjust(left=0.30, right=0.92, bottom=0.19, top=0.88)
    save_fig(fig, 'Figure_source_IG_enrichment_heatmap')


def plot_signed_measured_heatmaps(tab):
    """Two-panel supplement: strongest positive versus negative measured effects."""
    labels = [
        f'{drug}: {strategy}'
        for strategy in ('Developmental', 'Direct Astro')
        for drug in DRUG_LABELS
    ]
    analyses_by_direction = {
        direction: [
            f'{strategy}__{drug}__{direction}'
            for strategy in STRATEGIES for drug in DRUGS
        ]
        for direction in ('measured_positive', 'measured_negative')
    }
    all_values = [
        np.log2(get_family_rows(tab, analysis)['obs_exp']
                .clip(lower=1e-9).to_numpy(float))
        for analyses in analyses_by_direction.values()
        for analysis in analyses
    ]
    max_abs = max(0.5, float(np.nanmax(np.abs(all_values))))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SIGNED, sharey=True)

    for ax, direction, title in zip(
        axes,
        ('measured_positive', 'measured_negative'),
        ('Strongest positive measured effects',
         'Strongest negative measured effects'),
    ):
        im = heatmap_signed_log2(
            ax, tab, analyses_by_direction[direction], labels, title,
            max_abs=max_abs, annotation_fontsize=10,
        )
    # Reserve room at the right for one shared colorbar without changing figure size.
    fig.subplots_adjust(left=0.19, right=0.86, bottom=0.24, top=0.88, wspace=0.37)
    cax = fig.add_axes([0.89, 0.26, 0.018, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('log₂(observed / expected)', fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    save_fig(fig, 'Supplement_measured_positive_negative_family_enrichment')


def main():
    effects = load_enrichment('category_enrichment_measured_predicted.csv')
    ig = load_enrichment('IG_source_family_enrichment.csv')
    plot_grouped_effects(effects)
    plot_ig_heatmap(ig)
    plot_signed_measured_heatmaps(effects)
    print('All figures saved in:', OUT_DIR)


if __name__ == '__main__':
    main()
