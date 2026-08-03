"""
Plotting functions for Onsager transport analysis.
MSD plots, heatmaps, and flux visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition

from onsager_utils import formula_to_latex


# Configure matplotlib defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 48


def custom_plot_correlations(doc_dict, fig_path, size=12):
    """
    Custom plotting function for MSD correlation data from saved TransportDoc.
    """
    fig, ax = plt.subplots(figsize=(size, size))

    species = doc_dict.get('species', [])
    times = doc_dict.get('times', np.array([]))
    msds = doc_dict.get('msds', {})
    time_step = doc_dict.get('time_step', 1)
    step_skip = doc_dict.get('step_skip', 1)

    time_axis = times * time_step * step_skip
    n_species = len(species)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for i in range(n_species):
        for j in range(i, n_species):
            key = (i, j)
            if key in msds:
                msd_data = msds[key]
                label = f'{species[i]}-{species[j]}'

                if isinstance(msd_data, np.ndarray):
                    if msd_data.ndim == 1:
                        ax.plot(time_axis, msd_data, label=label, color=colors[color_idx % 10])
                    elif msd_data.ndim == 2:
                        if len(msd_data) > 4:
                            ax.plot(time_axis, msd_data[4], label=f'{label} (net)', color=colors[color_idx % 10])
                        else:
                            ax.plot(time_axis, msd_data[-1], label=label, color=colors[color_idx % 10])
                color_idx += 1

    ax.set_xlabel('Time (fs)', fontsize=14)
    ax.set_ylabel('Correlation function (1/(cm·eV))', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f"{doc_dict.get('reduced_formula', 'Unknown')} at {doc_dict.get('temperature', '?')}K", fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """Create a heatmap from a numpy array and two lists of labels."""
    if ax is None:
        ax = plt.gca()
    if cbar_kw is None:
        cbar_kw = {}

    # Replace 0s with NaN for plotting
    data_to_plot = np.where(data == 0, np.nan, data)
    im = ax.imshow(data_to_plot, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=40)

    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=0, ha="center", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap_mean_std(im, mean, std, threshold=None,
                             textcolors=("black", "white"), **textkw):
    """Annotate a heatmap with mean +/- std."""
    if threshold is None:
        norm = im.norm
        threshold = norm.vmax - (norm.vmax - norm.vmin) / 2
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)
    texts = []
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if np.isnan(mean[i, j]):
                color = textcolors[0]
                label = "NA"
            else:
                color = textcolors[int(mean[i, j] > threshold)]
                label = f"{mean[i, j]:.2f}\n+/-{std[i, j]:.2f}"
            text = im.axes.text(j, i, label, color=color, **kw)
            texts.append(text)
    return texts


def formula_contains_element(formula, element):
    """Check if a formula contains a specific element."""
    try:
        comp = Composition(formula)
        return element in [str(el) for el in comp.elements]
    except:
        return False


def get_flux(df, formula, temperature, mu):
    """
    Calculate ion flux given Onsager coefficients and chemical potential gradient.
    Uses Monte Carlo sampling for uncertainty estimation.
    """
    formula_data = df[(df['Formula'] == formula) & (df['Temperature'] == temperature)]
    if len(formula_data) == 0:
        return None, None

    fluxes = []
    for n in range(1000):  # Monte Carlo sampling for uncertainty
        Lij = []
        for i in range(3):
            for j in range(3):
                col = f'L{min(i,j)}{max(i,j)}'
                std_col = f'{col}_std'
                val = formula_data[col].values[0] if col in formula_data.columns else 0
                std = formula_data[std_col].values[0] if std_col in formula_data.columns else 0
                Lij.append(np.random.normal(val, std) if std > 0 else val)
        L = np.array(Lij).reshape(3, 3)
        fluxes.append(-np.dot(L, mu))

    return np.mean(fluxes, axis=0), np.std(fluxes, axis=0)
