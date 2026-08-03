"""
Interphase Transport Plotting Module
=====================================
Publication-quality plots for interphase analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional, Tuple


# =============================================================================
# Plot Configuration
# =============================================================================

# Publication-quality settings
FONTSIZE_TITLE = 11
FONTSIZE_LABEL = 10
FONTSIZE_TICK = 9
FONTSIZE_LEGEND = 9

# Species colors (consistent with flux plots)
SPECIES_COLORS = {
    'Na': '#558AD0',  # Blue
    'P': '#BA4743',   # Red
    'S': '#CACB92',   # Yellow-green
}

# =============================================================================
# Unit Conversion Constants
# =============================================================================
# Cell dimensions from LAMMPS dump
X_DIM_ANGSTROM = 136.55117713571758  # Å
Y_DIM_ANGSTROM = 136.55117713571758  # Å
AREA_ANGSTROM2 = X_DIM_ANGSTROM * Y_DIM_ANGSTROM  # Å²
AREA_M2 = AREA_ANGSTROM2 * 1e-20  # m²

# Time parameters
TIME_PER_FRAME_NS = 0.01  # ns per frame
TIME_PER_FRAME_S = TIME_PER_FRAME_NS * 1e-9  # seconds per frame

# Avogadro's number
N_AVOGADRO = 6.02214076e23  # mol⁻¹

# Conversion factors
# atoms/frame -> atoms/(m²·s)
FLUX_CONVERSION_ATOMS = 1.0 / (AREA_M2 * TIME_PER_FRAME_S)
# ≈ 5.363e26 atoms/(m²·s) per (atom/frame)

# atoms/frame -> mol/(m²·s)
FLUX_CONVERSION_MOL = FLUX_CONVERSION_ATOMS / N_AVOGADRO
# ≈ 890.5 mol/(m²·s) per (atom/frame)

# Symlog auto-detection parameters
DYNAMIC_RANGE_THRESHOLD = 100  # Use symlog if max/min > 100×


# =============================================================================
# Scale Detection Helper
# =============================================================================

def check_if_symlog_needed(data_series_list, threshold=DYNAMIC_RANGE_THRESHOLD):
    """
    Check if symlog scale is needed based on dynamic range of data.

    Parameters
    ----------
    data_series_list : list of pd.Series or np.ndarray
        List of data arrays to check (e.g., flux for each species)
    threshold : float
        Dynamic range threshold (default 100×)

    Returns
    -------
    use_symlog : bool
        True if symlog is recommended
    linthresh : float
        Recommended linthresh value (10th percentile of |values|)
    """
    # Combine all data
    all_values = []
    for series in data_series_list:
        if series is not None:
            # Drop NaN and get absolute values
            valid = np.abs(series.dropna().values) if hasattr(series, 'dropna') else np.abs(series[~np.isnan(series)])
            if len(valid) > 0:
                all_values.extend(valid)

    if len(all_values) == 0:
        return False, 1.0

    all_values = np.array(all_values)
    # Filter out zeros for dynamic range calculation
    nonzero = all_values[all_values > 0]

    if len(nonzero) < 2:
        return False, 1.0

    # Calculate dynamic range
    dynamic_range = np.max(nonzero) / np.min(nonzero)

    # Calculate linthresh as 10th percentile
    linthresh = np.percentile(nonzero, 10)

    use_symlog = dynamic_range > threshold

    return use_symlog, linthresh


def setup_publication_style():
    """Apply publication-quality matplotlib settings."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = FONTSIZE_TICK
    plt.rcParams['axes.labelsize'] = FONTSIZE_LABEL
    plt.rcParams['axes.titlesize'] = FONTSIZE_TITLE
    plt.rcParams['legend.fontsize'] = FONTSIZE_LEGEND
    plt.rcParams['xtick.labelsize'] = FONTSIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONTSIZE_TICK


# =============================================================================
# Composition Heatmap
# =============================================================================

def plot_composition_heatmap(
    frames: np.ndarray,
    bin_centers: np.ndarray,
    compositions: np.ndarray,
    time_per_frame: float = 0.01,
    boundaries_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot z-t composition heatmap showing interphase evolution.

    Parameters
    ----------
    frames : np.ndarray
        Frame numbers
    bin_centers : np.ndarray
        Bin center positions (fractional)
    compositions : np.ndarray
        Shape (n_frames, n_bins, 3) with [Na, P, S] fractions
    time_per_frame : float
        Time between frames in ns
    boundaries_df : pd.DataFrame, optional
        Boundary positions to overlay
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    # Calculate P+S fraction (indicator of interphase/electrolyte)
    ps_fraction = compositions[:, :, 1] + compositions[:, :, 2]

    # Time array
    time_ns = frames * time_per_frame

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.pcolormesh(
        time_ns, bin_centers, ps_fraction.T,
        cmap='viridis', shading='auto', vmin=0, vmax=0.7
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='P + S fraction')
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    # Overlay boundary positions if provided
    if boundaries_df is not None:
        valid_df = boundaries_df[boundaries_df['valid']]
        time_valid = valid_df['frame'].values * time_per_frame
        ax.plot(time_valid, valid_df['z_lower'].values, 'w-', linewidth=1.5,
                label='Na/interphase boundary')
        ax.plot(time_valid, valid_df['z_upper'].values, 'w--', linewidth=1.5,
                label='Interphase/Na₃PS₄ boundary')
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND, framealpha=0.9)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('z (fractional)')
    ax.set_title('Composition Evolution: P + S Fraction')

    # Show full cell or focus on interface region if boundaries provided
    if boundaries_df is not None:
        valid_df = boundaries_df[boundaries_df['valid']]
        if len(valid_df) > 0:
            z_min = valid_df['z_lower'].min()
            z_max = valid_df['z_upper'].max()
            # Add padding
            padding = (z_max - z_min) * 0.2
            ax.set_ylim(max(0, z_min - padding), min(1.0, z_max + padding))
        else:
            ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# Boundary Evolution Plot
# =============================================================================

def plot_boundary_evolution(
    boundaries_df: pd.DataFrame,
    lattice_heights: Optional[pd.Series] = None,
    time_per_frame: float = 0.01,
    figsize: Tuple[float, float] = (7, 4),
    save_path: Optional[str] = None
):
    """
    Plot interphase boundary positions over time.

    Parameters
    ----------
    boundaries_df : pd.DataFrame
        Boundary positions from analysis
    lattice_heights : pd.Series, optional
        Lattice height per frame for converting to Angstroms
    time_per_frame : float
        Time between frames in ns
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    valid_df = boundaries_df[boundaries_df['valid']].copy()
    valid_df['time_ns'] = valid_df['frame'] * time_per_frame

    # Ensure numeric types for plotting
    valid_df['z_lower'] = pd.to_numeric(valid_df['z_lower'], errors='coerce')
    valid_df['z_upper'] = pd.to_numeric(valid_df['z_upper'], errors='coerce')

    # Drop any remaining NaN values
    valid_df = valid_df.dropna(subset=['z_lower', 'z_upper'])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(valid_df['time_ns'].values, valid_df['z_lower'].values, '-', color='#2E86AB',
            linewidth=1.5, label='Na/interphase (z_lower)')
    ax.plot(valid_df['time_ns'].values, valid_df['z_upper'].values, '-', color='#A23B72',
            linewidth=1.5, label='Interphase/Na₃PS₄ (z_upper)')

    # Fill between boundaries
    ax.fill_between(valid_df['time_ns'].values, valid_df['z_lower'].values, valid_df['z_upper'].values,
                    alpha=0.3, color='#F18F01', label='Interphase region')

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('z (fractional)')
    ax.set_title('Interphase Boundary Evolution')
    ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# Interphase Thickness Plot
# =============================================================================

def plot_interphase_thickness(
    growth_df: pd.DataFrame,
    lattice_heights: Optional[pd.Series] = None,
    time_per_frame: float = 0.01,
    figsize: Tuple[float, float] = (7, 4),
    save_path: Optional[str] = None
):
    """
    Plot interphase thickness over time.

    Parameters
    ----------
    growth_df : pd.DataFrame
        Growth data from analysis
    lattice_heights : pd.Series, optional
        Lattice height per frame for converting to Angstroms
    time_per_frame : float
        Time between frames in ns
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    valid_df = growth_df[growth_df['valid']].copy()

    # Ensure numeric types
    valid_df['thickness_frac'] = pd.to_numeric(valid_df['thickness_frac'], errors='coerce')
    valid_df = valid_df.dropna(subset=['thickness_frac'])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(valid_df['time_ns'].values, valid_df['thickness_frac'].values * 100, '-',
            color='#2E86AB', linewidth=1.5)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Interphase Thickness (% of cell)')
    ax.set_title('Interphase Growth')
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    # Add linear fit for growth rate
    from scipy import stats
    valid_data = valid_df.dropna(subset=['thickness_frac'])
    if len(valid_data) > 10:
        slope, intercept, r_value, _, _ = stats.linregress(
            valid_data['time_ns'].values, valid_data['thickness_frac'].values * 100
        )
        fit_line = slope * valid_data['time_ns'].values + intercept
        ax.plot(valid_data['time_ns'].values, fit_line, '--', color='red', linewidth=1,
                label=f'Linear fit: {slope:.3f} %/ns (R²={r_value**2:.3f})')
        ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# Interphase Population Plot
# =============================================================================

def plot_interphase_population(
    population_df: pd.DataFrame,
    time_per_frame: float = 0.01,
    figsize: Tuple[float, float] = (7, 4),
    save_path: Optional[str] = None
):
    """
    Plot number of atoms in interphase over time.

    Parameters
    ----------
    population_df : pd.DataFrame
        Population data from analysis
    time_per_frame : float
        Time between frames in ns
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    df = population_df.copy()
    df['time_ns'] = df['frame'] * time_per_frame

    # Ensure numeric types
    for specie in ['Na', 'P', 'S']:
        df[f'N_{specie}'] = pd.to_numeric(df[f'N_{specie}'], errors='coerce')

    fig, ax = plt.subplots(figsize=figsize)

    for specie, color in SPECIES_COLORS.items():
        # Drop NaN values for this species
        valid_mask = df[f'N_{specie}'].notna()
        ax.plot(df.loc[valid_mask, 'time_ns'].values,
                df.loc[valid_mask, f'N_{specie}'].values, '-', color=color,
                linewidth=1.5, label=specie)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Number of atoms in interphase')
    ax.set_title('Interphase Population Evolution')
    ax.legend(loc='upper left', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# Flux Plots
# =============================================================================

def plot_interphase_flux(
    flux_df: pd.DataFrame,
    boundary: str = 'lower',
    time_per_frame: float = 0.01,
    window_size: int = 10,
    figsize: Tuple[float, float] = (7, 8),
    save_path: Optional[str] = None,
    unit_type: str = 'mol',
    auto_scale: bool = True,
    force_scale: Optional[str] = None
):
    """
    Plot flux across a boundary over time.

    Parameters
    ----------
    flux_df : pd.DataFrame
        Flux data from analysis
    boundary : str
        'lower' (Na/interphase) or 'upper' (interphase/Na₃PS₄)
    time_per_frame : float
        Time between frames in ns
    window_size : int
        Window for rolling average
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure
    unit_type : str
        'mol' for mol/(m²·s), 'atoms' for atoms/frame
    auto_scale : bool
        If True, auto-detect if symlog is needed (default True)
    force_scale : str, optional
        'linear' or 'symlog' to override auto-detection
    """
    setup_publication_style()

    df = flux_df.copy()
    df['time_ns'] = df['frame'] * time_per_frame

    # Unit conversion factor based on unit_type
    if unit_type == 'mol':
        flux_factor = FLUX_CONVERSION_MOL
        flux_unit = r'mol$\cdot$m$^{-2}\cdot$s$^{-1}$'
        cumulative_factor = FLUX_CONVERSION_MOL * TIME_PER_FRAME_S  # mol/m² per atom
        cumulative_unit = r'mol$\cdot$m$^{-2}$'
    else:  # atoms/frame
        flux_factor = 1.0
        flux_unit = 'atoms/frame'
        cumulative_factor = 1.0
        cumulative_unit = 'atoms'

    # Collect raw flux data for all species to check dynamic range
    net_flux_series = []
    cumulative_series = []
    for specie in SPECIES_COLORS.keys():
        col = f'flux_{specie}_{boundary}_net'
        if col in df.columns:
            net_flux_series.append(df[col] * flux_factor)
            cumulative_series.append(df[col].cumsum() * cumulative_factor)

    # Determine scale: force_scale overrides auto_scale
    if force_scale is not None:
        use_symlog_net = (force_scale == 'symlog')
        use_symlog_cum = (force_scale == 'symlog')
        # Still compute linthresh for symlog
        _, linthresh_net = check_if_symlog_needed(net_flux_series)
        _, linthresh_cum = check_if_symlog_needed(cumulative_series)
    elif auto_scale:
        use_symlog_net, linthresh_net = check_if_symlog_needed(net_flux_series)
        use_symlog_cum, linthresh_cum = check_if_symlog_needed(cumulative_series)
    else:
        use_symlog_net, linthresh_net = False, 1.0
        use_symlog_cum, linthresh_cum = False, 1.0

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    boundary_name = 'Na/Interphase' if boundary == 'lower' else 'Interphase/Na₃PS₄'
    unit_label = 'mol/(m²·s)' if unit_type == 'mol' else 'atoms/frame'

    # Plot 1: Net flux
    ax1 = axes[0]
    for specie, color in SPECIES_COLORS.items():
        col = f'flux_{specie}_{boundary}_net'
        if col in df.columns:
            raw = df[col] * flux_factor
            rolling = raw.rolling(window=window_size, center=True).mean()
            ax1.plot(df['time_ns'], raw, alpha=0.3, color=color, linewidth=0.5)
            ax1.plot(df['time_ns'], rolling, color=color, linewidth=1.5, label=specie)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_ylabel(f'Net Flux\n({flux_unit})')
    scale_note = ' [symlog]' if use_symlog_net else ''
    ax1.set_title(f'Flux Across {boundary_name} Boundary ({unit_label}){scale_note}')
    ax1.legend(loc='lower right', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    if use_symlog_net:
        ax1.set_yscale('symlog', linthresh=linthresh_net)
    else:
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)

    # Plot 2: Cumulative flux
    ax2 = axes[1]
    for specie, color in SPECIES_COLORS.items():
        col = f'flux_{specie}_{boundary}_net'
        if col in df.columns:
            cumulative = df[col].cumsum() * cumulative_factor
            ax2.plot(df['time_ns'], cumulative, color=color, linewidth=1.5, label=specie)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.set_ylabel(f'Cumulative Flux\n({cumulative_unit})')
    scale_note_cum = ' [symlog]' if use_symlog_cum else ''
    ax2.set_title(f'Cumulative Flux Across {boundary_name} Boundary{scale_note_cum}')
    ax2.legend(loc='lower right', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    if use_symlog_cum:
        ax2.set_yscale('symlog', linthresh=linthresh_cum)
    else:
        ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)

    # Plot 3: Directional flux (always linear)
    ax3 = axes[2]
    for specie, color in SPECIES_COLORS.items():
        col_pos = f'flux_{specie}_{boundary}_pos'
        col_neg = f'flux_{specie}_{boundary}_neg'
        if col_pos in df.columns and col_neg in df.columns:
            pos_raw = df[col_pos] * flux_factor
            neg_raw = df[col_neg] * flux_factor
            pos_rolling = pos_raw.rolling(window=window_size, center=True).mean()
            neg_rolling = neg_raw.rolling(window=window_size, center=True).mean()
            ax3.plot(df['time_ns'], pos_rolling, color=color, linewidth=1.5,
                     linestyle='-', label=f'{specie} (+z)')
            ax3.plot(df['time_ns'], neg_rolling, color=color, linewidth=1.5,
                     linestyle='--', label=f'{specie} (-z)')

    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel(f'Directional Flux\n({flux_unit})')
    ax3.set_title('Directional Flux (+z vs -z)')
    ax3.legend(loc='upper right', ncol=3, fontsize=FONTSIZE_LEGEND-1, framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    # Print scale detection info
    if auto_scale:
        print(f"  Scale detection: Net flux symlog={use_symlog_net} (linthresh={linthresh_net:.2e}), "
              f"Cumulative symlog={use_symlog_cum} (linthresh={linthresh_cum:.2e})")


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary_statistics(results: Dict, time_per_frame: float = 0.01):
    """
    Print summary statistics from analysis results.

    Parameters
    ----------
    results : dict
        Results from run_interphase_analysis()
    time_per_frame : float
        Time between frames in ns
    """
    boundaries = results['boundaries']
    population = results['population']
    flux = results['flux']
    growth = results['growth']

    print("\n" + "="*70)
    print("INTERPHASE ANALYSIS SUMMARY STATISTICS")
    print("="*70)

    # Boundary detection
    n_valid = boundaries['valid'].sum()
    n_total = len(boundaries)
    print(f"\n1. Boundary Detection:")
    print(f"   Valid frames: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # Thickness evolution
    valid_growth = growth[growth['valid']]
    if len(valid_growth) > 0:
        initial_thickness = valid_growth['thickness_frac'].iloc[0]
        final_thickness = valid_growth['thickness_frac'].iloc[-1]
        print(f"\n2. Interphase Thickness:")
        print(f"   Initial: {initial_thickness*100:.3f}% of cell")
        print(f"   Final: {final_thickness*100:.3f}% of cell")
        print(f"   Change: {(final_thickness - initial_thickness)*100:.3f}%")

        # Linear growth rate
        from scipy import stats
        valid_data = valid_growth.dropna(subset=['thickness_frac'])
        if len(valid_data) > 10:
            slope, _, r_value, _, _ = stats.linregress(
                valid_data['time_ns'], valid_data['thickness_frac']
            )
            print(f"   Growth rate: {slope*100:.4f} %/ns (R²={r_value**2:.3f})")

    # Population
    valid_pop = population.dropna()
    if len(valid_pop) > 0:
        print(f"\n3. Interphase Population (Final Frame):")
        final = valid_pop.iloc[-1]
        print(f"   Na: {int(final['N_Na']):,} atoms")
        print(f"   P: {int(final['N_P']):,} atoms")
        print(f"   S: {int(final['N_S']):,} atoms")
        print(f"   Total: {int(final['N_total']):,} atoms")

    # Flux
    if len(flux) > 0:
        print(f"\n4. Cumulative Flux (Total Simulation):")
        cumulative_factor = 1.0 / AREA_M2  # atoms/m²
        for boundary in ['lower', 'upper']:
            boundary_name = 'Na/Interphase' if boundary == 'lower' else 'Interphase/Na₃PS₄'
            print(f"\n   {boundary_name} boundary:")
            for specie in ['Na', 'P', 'S']:
                col = f'flux_{specie}_{boundary}_net'
                if col in flux.columns:
                    total_atoms = flux[col].sum()
                    total_per_area = total_atoms * cumulative_factor
                    print(f"      {specie}: {int(total_atoms):+,} atoms = {total_per_area:+.3e} atoms/m²")

        # Average flux rate
        print(f"\n5. Average Flux Rate:")
        n_frames = len(flux)
        for boundary in ['lower', 'upper']:
            boundary_name = 'Na/Interphase' if boundary == 'lower' else 'Interphase/Na₃PS₄'
            print(f"\n   {boundary_name} boundary:")
            for specie in ['Na', 'P', 'S']:
                col = f'flux_{specie}_{boundary}_net'
                if col in flux.columns:
                    avg_atoms_per_frame = flux[col].mean()
                    avg_flux = avg_atoms_per_frame * FLUX_CONVERSION_ATOMS
                    print(f"      {specie}: {avg_flux:+.3e} atoms/(m²·s)")

    print("\n" + "="*70)


# =============================================================================
# Master Plotting Function
# =============================================================================

def generate_all_plots(
    results: Dict,
    time_per_frame: float = 0.01,
    output_dir: str = 'figures'
):
    """
    Generate all plots from analysis results.

    Generates 12 plots total:
    1. Composition heatmap
    2. Boundary evolution
    3. Interphase thickness
    4. Interphase population
    5-12. Flux plots (2 boundaries × 2 unit types × 2 scales)

    Parameters
    ----------
    results : dict
        Results from run_interphase_analysis()
    time_per_frame : float
        Time between frames in ns
    output_dir : str
        Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating plots...")
    print("="*60)

    # 1. Composition heatmap
    print("\n[1/12] Composition heatmap...")
    plot_composition_heatmap(
        results['frames'], results['bin_centers'], results['compositions'],
        time_per_frame, results['boundaries'],
        save_path=f'{output_dir}/interphase_composition_heatmap.png'
    )

    # 2. Boundary evolution
    print("\n[2/12] Boundary evolution...")
    plot_boundary_evolution(
        results['boundaries'], time_per_frame=time_per_frame,
        save_path=f'{output_dir}/interphase_boundary_evolution.png'
    )

    # 3. Thickness
    print("\n[3/12] Interphase thickness...")
    plot_interphase_thickness(
        results['growth'], time_per_frame=time_per_frame,
        save_path=f'{output_dir}/interphase_thickness.png'
    )

    # 4. Population
    print("\n[4/12] Interphase population...")
    plot_interphase_population(
        results['population'], time_per_frame=time_per_frame,
        save_path=f'{output_dir}/interphase_population.png'
    )

    # Flux plots: 2 boundaries × 2 units × 2 scales = 8 plots
    plot_num = 5

    for boundary in ['lower', 'upper']:
        boundary_name = 'Na/interphase' if boundary == 'lower' else 'interphase/Na₃PS₄'

        for unit_type in ['mol', 'atoms']:
            unit_label = 'mol/(m²·s)' if unit_type == 'mol' else 'atoms/frame'

            for scale in ['linear', 'symlog']:
                print(f"\n[{plot_num}/12] Flux across {boundary_name} [{unit_label}] ({scale})...")
                plot_interphase_flux(
                    results['flux'], boundary=boundary, time_per_frame=time_per_frame,
                    unit_type=unit_type, force_scale=scale,
                    save_path=f'{output_dir}/interphase_flux_{boundary}_{unit_type}_{scale}.png'
                )
                plot_num += 1

    print("\n" + "="*60)
    print("All plots generated!")
    print("="*60)

    # Print summary statistics
    print_summary_statistics(results, time_per_frame)


def plot_interphase_population_log(
    population_df: pd.DataFrame,
    time_per_frame: float = 0.01,
    figsize: Tuple[float, float] = (7, 4),
    save_path: Optional[str] = None
):
    """
    Plot number of atoms in interphase over time with log scale.

    Parameters
    ----------
    population_df : pd.DataFrame
        Population data from analysis
    time_per_frame : float
        Time between frames in ns
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    df = population_df.copy()
    df['time_ns'] = df['frame'] * time_per_frame

    # Ensure numeric types
    for specie in ['Na', 'P', 'S']:
        df[f'N_{specie}'] = pd.to_numeric(df[f'N_{specie}'], errors='coerce')

    fig, ax = plt.subplots(figsize=figsize)

    for specie, color in SPECIES_COLORS.items():
        # Drop NaN values for this species
        valid_mask = df[f'N_{specie}'].notna() & (df[f'N_{specie}'] > 0)
        ax.plot(df.loc[valid_mask, 'time_ns'].values,
                df.loc[valid_mask, f'N_{specie}'].values, '-', color=color,
                linewidth=1.5, label=specie)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Number of atoms in interphase')
    ax.set_title('Interphase Population Evolution (Log Scale)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
