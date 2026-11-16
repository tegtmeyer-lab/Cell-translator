"""Supplementary cross-drug morphology-effect specificity heatmaps.

Reads existing Phase 1 CSVs, without retraining or recomputing signatures.
Rows: predicted drug; columns: measured drug. Correlations use the same
common morphology features within each strategy, as in the Phase 1 analysis.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

RUN_ROOT = Path(os.environ.get('CELL_TRANSLATOR_RUN_DIR', str(Path.cwd() / 'outputs' / 'rerun_70_15_15_qc_v3'))).expanduser().resolve()
INPUT_DIR = RUN_ROOT / 'evaluation' / 'phase1_interpretation'
OUTPUT_DIR = INPUT_DIR / 'supplementary_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUGS = ['Latrunculin_B', 'GW3965', 'Clozapine']
LABELS = ['Latrunculin B', 'GW3965', 'Clozapine']
FILE_MAP = {
    'A': ('developmental_cross_drug_specificity_matched_features.csv',
          'Developmental translation'),
    'B': ('control_to_drug_cross_drug_specificity_matched_features.csv',
          'Astrocyte control → drug'),
}


def load_matrix(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run the cross-drug specificity cell in the Phase 1 notebook.')
    df = pd.read_csv(path, index_col=0)
    # Keep the exact drug ordering for both strategies.
    if set(DRUGS) - set(df.index) or set(DRUGS) - set(df.columns):
        raise ValueError(f'Unexpected labels in {path.name}: rows={list(df.index)}, columns={list(df.columns)}')
    df = df.loc[DRUGS, DRUGS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f'Nonfinite correlations in {path.name}')
    return df


def make_figure(matrices):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.35), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap('RdBu_r')

    im = None
    for ax, (panel, (matrix, heading)) in zip(axs, matrices.items()):
        vals = matrix.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        ax.set_title(f'{panel}. {heading}', fontsize=11, fontweight='semibold', pad=12)
        ax.set_xticks(np.arange(3), LABELS, rotation=25, ha='right')
        ax.set_yticks(np.arange(3), LABELS)
        ax.set_xlabel('Measured astrocyte drug effect')
        ax.set_ylabel('Predicted astrocyte drug effect')
        ax.tick_params(length=0)
        for i in range(3):
            for j in range(3):
                v = vals[i, j]
                color = 'white' if abs(v) >= 0.62 else '#202020'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=12, color=color,
                        fontweight='bold' if i == j else 'normal')
            ax.add_patch(Rectangle((i - .49, i - .49), .98, .98,
                                   fill=False, edgecolor='#111111', linewidth=2.2))
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axs, shrink=.77, fraction=.035, pad=.025)
    cbar.set_label('Pearson correlation (r)')
    return fig


def main():
    matrices = {}
    for panel, (fname, heading) in FILE_MAP.items():
        matrix = load_matrix(INPUT_DIR / fname)
        matrices[panel] = (matrix, heading)
        matched = np.diag(matrix.to_numpy())
        margins = [matrix.iloc[i, i] - matrix.iloc[i].drop(matrix.index[i]).max()
                   for i in range(3)]
        print(f'{heading}: matched r = {np.round(matched, 3)}, '
              f'specificity margins = {np.round(margins, 3)}')

    fig = make_figure(matrices)
    for extension in ('png', 'pdf'):
        path = OUTPUT_DIR / f'Supplementary_cross_drug_specificity.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print('Saved:', path)
    plt.close(fig)


if __name__ == '__main__':
    main()
