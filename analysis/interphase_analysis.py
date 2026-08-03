"""
Interphase Transport Analysis Module
=====================================
Functions for analyzing Na/Na₃PS₄ interface evolution from MD trajectories.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Optional


# =============================================================================
# Constants
# =============================================================================

# Bulk Na₃PS₄ stoichiometry (atom fractions)
BULK_NA3PS4 = {
    'Na': 3/8,  # 37.5%
    'P': 1/8,   # 12.5%
    'S': 4/8,   # 50.0%
}

# Bulk Na metal stoichiometry
BULK_NA_METAL = {
    'Na': 1.0,
    'P': 0.0,
    'S': 0.0,
}


# =============================================================================
# Composition Profile Functions
# =============================================================================

def compute_composition_profile(
    df: pd.DataFrame,
    frame: int,
    bin_width: float = 0.006,
    z_col: str = 'z-frac_coord',
    bin_offset: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local composition profile along z-axis for a single frame.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory dataframe with columns: site, site_specie, z-frac_coord, frame
    frame : int
        Frame number to analyze
    bin_width : float
        Bin width in fractional coordinates (default ~0.006 ≈ 5 Å at 842 Å cell)
    z_col : str
        Column name for z-coordinate (fractional)
    bin_offset : float
        Offset for bin edges (0 to bin_width), for phase-shifting bins

    Returns
    -------
    bin_centers : np.ndarray
        Center positions of each bin (fractional coordinates)
    composition : np.ndarray
        Shape (n_bins, 3) with columns [Na_frac, P_frac, S_frac]
    counts : np.ndarray
        Shape (n_bins, 3) with columns [Na_count, P_count, S_count]
    """
    # Filter to this frame
    frame_df = df[df['frame'] == frame]

    # Define bins with offset
    # Start from -bin_offset so first full bin starts at ~0
    bin_edges = np.arange(-bin_offset, 1 + bin_width, bin_width)
    # Clip to [0, 1] range
    bin_edges = np.clip(bin_edges, 0, 1)
    # Remove duplicates that might arise from clipping
    bin_edges = np.unique(bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Initialize count arrays
    counts = np.zeros((n_bins, 3))  # Na, P, S

    # Count atoms in each bin by species
    for i, specie in enumerate(['Na', 'P', 'S']):
        specie_z = frame_df[frame_df['site_specie'] == specie][z_col].values
        hist, _ = np.histogram(specie_z, bins=bin_edges)
        counts[:, i] = hist

    # Calculate composition (fraction of each species in bin)
    total_per_bin = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty bins
    with np.errstate(divide='ignore', invalid='ignore'):
        composition = np.where(total_per_bin > 0, counts / total_per_bin, 0)

    return bin_centers, composition, counts


def compute_all_composition_profiles(
    df: pd.DataFrame,
    bin_width: float = 0.006,
    z_col: str = 'z-frac_coord'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute composition profiles for all frames.

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory dataframe
    bin_width : float
        Bin width in fractional coordinates
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    frames : np.ndarray
        Frame numbers
    bin_centers : np.ndarray
        Bin center positions (fractional)
    compositions : np.ndarray
        Shape (n_frames, n_bins, 3) - composition profiles
    counts : np.ndarray
        Shape (n_frames, n_bins, 3) - atom counts
    """
    frames = sorted(df['frame'].unique())
    n_frames = len(frames)

    # Get bin structure from first frame
    bin_centers, _, _ = compute_composition_profile(df, frames[0], bin_width, z_col)
    n_bins = len(bin_centers)

    # Initialize arrays
    compositions = np.zeros((n_frames, n_bins, 3))
    counts = np.zeros((n_frames, n_bins, 3))

    # Process each frame
    for i, frame in enumerate(tqdm(frames, desc="Computing composition profiles")):
        _, comp, cnt = compute_composition_profile(df, frame, bin_width, z_col)
        compositions[i] = comp
        counts[i] = cnt

    return np.array(frames), bin_centers, compositions, counts


# =============================================================================
# Boundary Detection Functions
# =============================================================================

def detect_na_interphase_boundary(
    composition: np.ndarray,
    bin_centers: np.ndarray,
    scan_start: float = 0.0,
    scan_end: float = 0.5
) -> Optional[float]:
    """
    Detect Na/interphase boundary as first bin with any P or S.

    Scans from low z (Na metal) toward high z (Na₃PS₄).

    Parameters
    ----------
    composition : np.ndarray
        Shape (n_bins, 3) with [Na_frac, P_frac, S_frac]
    bin_centers : np.ndarray
        Bin center positions (fractional)
    scan_start : float
        Start of scan region (fractional z)
    scan_end : float
        End of scan region (fractional z)

    Returns
    -------
    z_boundary : float or None
        Fractional z-coordinate of boundary, or None if not found
    """
    # Get bins in scan region
    mask = (bin_centers >= scan_start) & (bin_centers <= scan_end)

    for i, z in enumerate(bin_centers):
        if not mask[i]:
            continue
        # Check if any P or S present (composition > 0)
        p_frac = composition[i, 1]
        s_frac = composition[i, 2]
        if p_frac > 0 or s_frac > 0:
            return z

    return None  # Boundary not found


def detect_interphase_na3ps4_boundary(
    composition: np.ndarray,
    bin_centers: np.ndarray,
    tolerance: float = 0.02,
    scan_start: float = 0.0,
    scan_end: float = 1.0,
    na_interphase_boundary: Optional[float] = None
) -> Optional[float]:
    """
    Detect interphase/Na₃PS₄ boundary where composition returns to bulk stoichiometry.

    Scans from the Na/interphase boundary toward high z.

    Parameters
    ----------
    composition : np.ndarray
        Shape (n_bins, 3) with [Na_frac, P_frac, S_frac]
    bin_centers : np.ndarray
        Bin center positions (fractional)
    tolerance : float
        Tolerance for matching bulk stoichiometry (default ±2%)
    scan_start : float
        Start of scan region (fractional z)
    scan_end : float
        End of scan region (fractional z)
    na_interphase_boundary : float, optional
        Position of Na/interphase boundary to start scanning from

    Returns
    -------
    z_boundary : float or None
        Fractional z-coordinate of boundary, or None if not found
    """
    # Start scanning from Na/interphase boundary if provided
    if na_interphase_boundary is not None:
        scan_start = max(scan_start, na_interphase_boundary)

    # Get bins in scan region
    mask = (bin_centers >= scan_start) & (bin_centers <= scan_end)

    for i, z in enumerate(bin_centers):
        if not mask[i]:
            continue

        # Check if composition matches bulk Na₃PS₄ within tolerance
        na_frac = composition[i, 0]
        p_frac = composition[i, 1]
        s_frac = composition[i, 2]

        na_match = abs(na_frac - BULK_NA3PS4['Na']) <= tolerance
        p_match = abs(p_frac - BULK_NA3PS4['P']) <= tolerance
        s_match = abs(s_frac - BULK_NA3PS4['S']) <= tolerance

        if na_match and p_match and s_match:
            return z

    return None  # Boundary not found


def detect_boundaries_all_frames(
    compositions: np.ndarray,
    bin_centers: np.ndarray,
    frames: np.ndarray,
    tolerance: float = 0.02,
    interface_region: Tuple[float, float] = (0.0, 0.15)
) -> pd.DataFrame:
    """
    Detect interphase boundaries for all frames.

    Parameters
    ----------
    compositions : np.ndarray
        Shape (n_frames, n_bins, 3) - composition profiles
    bin_centers : np.ndarray
        Bin center positions (fractional)
    frames : np.ndarray
        Frame numbers
    tolerance : float
        Tolerance for bulk stoichiometry matching
    interface_region : tuple
        (z_min, z_max) fractional region to search for Interface A

    Returns
    -------
    boundaries_df : pd.DataFrame
        Columns: frame, z_lower, z_upper, thickness_frac, valid
    """
    results = []

    for i, frame in enumerate(tqdm(frames, desc="Detecting boundaries")):
        comp = compositions[i]

        # Detect Na/interphase boundary
        z_lower = detect_na_interphase_boundary(
            comp, bin_centers,
            scan_start=interface_region[0],
            scan_end=interface_region[1]
        )

        # Detect interphase/Na₃PS₄ boundary
        z_upper = detect_interphase_na3ps4_boundary(
            comp, bin_centers,
            tolerance=tolerance,
            scan_start=interface_region[0],
            scan_end=0.95,  # Scan up to 95% of cell to find bulk Na₃PS₄
            na_interphase_boundary=z_lower
        )

        # Check validity
        valid = (z_lower is not None) and (z_upper is not None)
        if valid:
            valid = valid and (z_upper > z_lower)

        # Calculate thickness - convert None to np.nan for numeric compatibility
        z_lower_val = z_lower if z_lower is not None else np.nan
        z_upper_val = z_upper if z_upper is not None else np.nan
        thickness = (z_upper_val - z_lower_val) if valid else np.nan

        results.append({
            'frame': frame,
            'z_lower': z_lower_val,
            'z_upper': z_upper_val,
            'thickness_frac': thickness,
            'valid': valid
        })

    return pd.DataFrame(results)


# =============================================================================
# Multi-Bin Averaging Functions
# =============================================================================

# Default configurations for multi-bin averaging
DEFAULT_BIN_WIDTHS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012]
DEFAULT_N_PHASE_SHIFTS = 5


def generate_bin_configurations(
    bin_widths: list = None,
    n_phase_shifts: int = None
) -> list:
    """
    Generate all (bin_width, bin_offset) configurations for multi-bin averaging.

    Parameters
    ----------
    bin_widths : list
        List of bin widths to use (default: 10 widths from 0.003 to 0.012)
    n_phase_shifts : int
        Number of phase shifts per bin width (default: 5)

    Returns
    -------
    configs : list of tuples
        List of (bin_width, bin_offset) pairs
    """
    if bin_widths is None:
        bin_widths = DEFAULT_BIN_WIDTHS
    if n_phase_shifts is None:
        n_phase_shifts = DEFAULT_N_PHASE_SHIFTS

    configs = []
    for bw in bin_widths:
        for i in range(n_phase_shifts):
            offset = (i / n_phase_shifts) * bw
            configs.append((bw, offset))

    return configs


def detect_boundaries_for_frame_single_config(
    frame_df: pd.DataFrame,
    bin_width: float,
    bin_offset: float,
    tolerance: float,
    interface_region: Tuple[float, float],
    z_col: str = 'z-frac_coord'
) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect boundaries for a single frame using one bin configuration.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Dataframe for a single frame
    bin_width : float
        Bin width
    bin_offset : float
        Bin offset for phase shifting
    tolerance : float
        Tolerance for bulk stoichiometry matching
    interface_region : tuple
        (z_min, z_max) to search for Interface A
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    z_lower, z_upper : tuple of floats or None
        Boundary positions, or None if not found
    """
    # Define bins with offset
    bin_edges = np.arange(-bin_offset, 1 + bin_width, bin_width)
    bin_edges = np.clip(bin_edges, 0, 1)
    bin_edges = np.unique(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Compute composition
    counts = np.zeros((n_bins, 3))
    for i, specie in enumerate(['Na', 'P', 'S']):
        specie_z = frame_df[frame_df['site_specie'] == specie][z_col].values
        hist, _ = np.histogram(specie_z, bins=bin_edges)
        counts[:, i] = hist

    total_per_bin = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        composition = np.where(total_per_bin > 0, counts / total_per_bin, 0)

    # Detect boundaries
    z_lower = detect_na_interphase_boundary(
        composition, bin_centers,
        scan_start=interface_region[0],
        scan_end=interface_region[1]
    )

    z_upper = detect_interphase_na3ps4_boundary(
        composition, bin_centers,
        tolerance=tolerance,
        scan_start=interface_region[0],
        scan_end=0.95,
        na_interphase_boundary=z_lower
    )

    return z_lower, z_upper


def detect_boundaries_averaged_all_frames(
    df: pd.DataFrame,
    bin_widths: list = None,
    n_phase_shifts: int = None,
    tolerance: float = 0.02,
    interface_region: Tuple[float, float] = (0.0, 0.15),
    z_col: str = 'z-frac_coord'
) -> pd.DataFrame:
    """
    Detect interphase boundaries for all frames using multi-bin averaging.

    For each frame, boundaries are detected using multiple bin configurations
    (varying bin widths and phase offsets), then averaged to reduce binning artifacts.

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory dataframe
    bin_widths : list
        List of bin widths to average over (default: [0.004, 0.005, 0.006, 0.007, 0.008])
    n_phase_shifts : int
        Number of phase shifts per bin width (default: 3)
    tolerance : float
        Tolerance for bulk stoichiometry matching
    interface_region : tuple
        (z_min, z_max) fractional region to search for Interface A
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    boundaries_df : pd.DataFrame
        Columns: frame, z_lower, z_upper, thickness_frac, valid,
                 z_lower_std, z_upper_std, n_valid_configs
    """
    configs = generate_bin_configurations(bin_widths, n_phase_shifts)
    n_configs = len(configs)

    frames = sorted(df['frame'].unique())
    results = []

    print(f"      Using {n_configs} bin configurations "
          f"({len(bin_widths or DEFAULT_BIN_WIDTHS)} widths × {n_phase_shifts} phase shifts)")

    for frame in tqdm(frames, desc="Detecting boundaries (multi-bin averaged)"):
        frame_df = df[df['frame'] == frame]

        # Collect boundaries from all configurations
        z_lowers = []
        z_uppers = []

        for bin_width, bin_offset in configs:
            z_lower, z_upper = detect_boundaries_for_frame_single_config(
                frame_df, bin_width, bin_offset, tolerance, interface_region, z_col
            )

            # Only include valid detections
            if z_lower is not None and z_upper is not None and z_upper > z_lower:
                z_lowers.append(z_lower)
                z_uppers.append(z_upper)

        # Average valid detections
        n_valid_configs = len(z_lowers)

        if n_valid_configs > 0:
            z_lower_avg = np.mean(z_lowers)
            z_upper_avg = np.mean(z_uppers)
            z_lower_std = np.std(z_lowers) if n_valid_configs > 1 else 0.0
            z_upper_std = np.std(z_uppers) if n_valid_configs > 1 else 0.0
            thickness = z_upper_avg - z_lower_avg
            valid = True
        else:
            z_lower_avg = np.nan
            z_upper_avg = np.nan
            z_lower_std = np.nan
            z_upper_std = np.nan
            thickness = np.nan
            valid = False

        results.append({
            'frame': frame,
            'z_lower': z_lower_avg,
            'z_upper': z_upper_avg,
            'thickness_frac': thickness,
            'valid': valid,
            'z_lower_std': z_lower_std,
            'z_upper_std': z_upper_std,
            'n_valid_configs': n_valid_configs
        })

    return pd.DataFrame(results)


# =============================================================================
# Interphase Population Tracking
# =============================================================================

def count_interphase_population(
    df: pd.DataFrame,
    frame: int,
    z_lower: float,
    z_upper: float,
    z_col: str = 'z-frac_coord'
) -> Dict[str, int]:
    """
    Count atoms of each species within the interphase region.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory dataframe
    frame : int
        Frame number
    z_lower : float
        Lower boundary (fractional z)
    z_upper : float
        Upper boundary (fractional z)
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    counts : dict
        {'Na': n_na, 'P': n_p, 'S': n_s, 'total': n_total}
    """
    frame_df = df[df['frame'] == frame]

    # Filter to interphase region
    interphase_mask = (frame_df[z_col] >= z_lower) & (frame_df[z_col] < z_upper)
    interphase_df = frame_df[interphase_mask]

    # Count by species
    species_counts = interphase_df['site_specie'].value_counts()

    return {
        'Na': species_counts.get('Na', 0),
        'P': species_counts.get('P', 0),
        'S': species_counts.get('S', 0),
        'total': len(interphase_df)
    }


def track_interphase_population_all_frames(
    df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    z_col: str = 'z-frac_coord'
) -> pd.DataFrame:
    """
    Track interphase population for all frames.

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory dataframe
    boundaries_df : pd.DataFrame
        Boundary positions from detect_boundaries_all_frames()
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    population_df : pd.DataFrame
        Columns: frame, N_Na, N_P, N_S, N_total
    """
    results = []

    for _, row in tqdm(boundaries_df.iterrows(), total=len(boundaries_df),
                       desc="Tracking interphase population"):
        frame = row['frame']

        if row['valid']:
            counts = count_interphase_population(
                df, frame, row['z_lower'], row['z_upper'], z_col
            )
        else:
            counts = {'Na': np.nan, 'P': np.nan, 'S': np.nan, 'total': np.nan}

        results.append({
            'frame': frame,
            'N_Na': counts['Na'],
            'N_P': counts['P'],
            'N_S': counts['S'],
            'N_total': counts['total']
        })

    return pd.DataFrame(results)


# =============================================================================
# Flux Calculation Functions
# =============================================================================

def calculate_boundary_flux(
    df: pd.DataFrame,
    frame: int,
    prev_frame: int,
    z_boundary: float,
    prev_z_boundary: float,
    z_col: str = 'z-frac_coord'
) -> Dict[str, Dict[str, int]]:
    """
    Calculate flux across a boundary between two frames.

    Counts both:
    - True crossings: atoms physically moving across boundary
    - Boundary sweep: boundary moving past atoms

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory dataframe
    frame : int
        Current frame number
    prev_frame : int
        Previous frame number
    z_boundary : float
        Current boundary position (fractional z)
    prev_z_boundary : float
        Previous boundary position (fractional z)
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    flux : dict
        {species: {'positive': n, 'negative': n, 'net': n}}
        Positive = crossing toward higher z
        Negative = crossing toward lower z
    """
    # Get data for both frames
    curr_df = df[df['frame'] == frame][['site', 'site_specie', z_col]].copy()
    prev_df = df[df['frame'] == prev_frame][['site', 'site_specie', z_col]].copy()

    # Merge on site to track individual atoms
    merged = prev_df.merge(
        curr_df,
        on=['site', 'site_specie'],
        suffixes=('_prev', '_curr')
    )

    # Determine position relative to boundary at each time
    merged['above_prev'] = merged[f'{z_col}_prev'] >= prev_z_boundary
    merged['above_curr'] = merged[f'{z_col}_curr'] >= z_boundary

    # Detect crossings (change in above/below status)
    merged['crossed_up'] = (~merged['above_prev']) & (merged['above_curr'])
    merged['crossed_down'] = (merged['above_prev']) & (~merged['above_curr'])

    # Count by species
    flux = {}
    for specie in ['Na', 'P', 'S']:
        specie_df = merged[merged['site_specie'] == specie]
        positive = specie_df['crossed_up'].sum()
        negative = specie_df['crossed_down'].sum()
        flux[specie] = {
            'positive': int(positive),
            'negative': int(negative),
            'net': int(positive - negative)
        }

    return flux


def calculate_flux_all_frames(
    df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    z_col: str = 'z-frac_coord'
) -> pd.DataFrame:
    """
    Calculate flux across both boundaries for all frames.

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory dataframe
    boundaries_df : pd.DataFrame
        Boundary positions from detect_boundaries_all_frames()
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    flux_df : pd.DataFrame
        Flux data for each frame and boundary
    """
    frames = boundaries_df['frame'].values
    results = []

    for i in tqdm(range(1, len(frames)), desc="Calculating flux"):
        frame = frames[i]
        prev_frame = frames[i-1]

        curr_row = boundaries_df[boundaries_df['frame'] == frame].iloc[0]
        prev_row = boundaries_df[boundaries_df['frame'] == prev_frame].iloc[0]

        # Skip if boundaries invalid
        if not (curr_row['valid'] and prev_row['valid']):
            results.append({
                'frame': frame,
                'flux_Na_lower_pos': np.nan, 'flux_Na_lower_neg': np.nan, 'flux_Na_lower_net': np.nan,
                'flux_P_lower_pos': np.nan, 'flux_P_lower_neg': np.nan, 'flux_P_lower_net': np.nan,
                'flux_S_lower_pos': np.nan, 'flux_S_lower_neg': np.nan, 'flux_S_lower_net': np.nan,
                'flux_Na_upper_pos': np.nan, 'flux_Na_upper_neg': np.nan, 'flux_Na_upper_net': np.nan,
                'flux_P_upper_pos': np.nan, 'flux_P_upper_neg': np.nan, 'flux_P_upper_net': np.nan,
                'flux_S_upper_pos': np.nan, 'flux_S_upper_neg': np.nan, 'flux_S_upper_net': np.nan,
            })
            continue

        # Calculate flux across lower boundary (Na/interphase)
        flux_lower = calculate_boundary_flux(
            df, frame, prev_frame,
            curr_row['z_lower'], prev_row['z_lower'],
            z_col
        )

        # Calculate flux across upper boundary (interphase/Na₃PS₄)
        flux_upper = calculate_boundary_flux(
            df, frame, prev_frame,
            curr_row['z_upper'], prev_row['z_upper'],
            z_col
        )

        result = {'frame': frame}
        for specie in ['Na', 'P', 'S']:
            result[f'flux_{specie}_lower_pos'] = flux_lower[specie]['positive']
            result[f'flux_{specie}_lower_neg'] = flux_lower[specie]['negative']
            result[f'flux_{specie}_lower_net'] = flux_lower[specie]['net']
            result[f'flux_{specie}_upper_pos'] = flux_upper[specie]['positive']
            result[f'flux_{specie}_upper_neg'] = flux_upper[specie]['negative']
            result[f'flux_{specie}_upper_net'] = flux_upper[specie]['net']

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# Growth Rate Calculation
# =============================================================================

def compute_growth_rate(
    boundaries_df: pd.DataFrame,
    time_per_frame: float = 0.01  # ns
) -> pd.DataFrame:
    """
    Compute interphase growth rate from boundary positions.

    Parameters
    ----------
    boundaries_df : pd.DataFrame
        Boundary positions with thickness_frac column
    time_per_frame : float
        Time between frames in ns

    Returns
    -------
    growth_df : pd.DataFrame
        Growth rate data
    """
    df = boundaries_df.copy()
    df['time_ns'] = df['frame'] * time_per_frame

    # Compute derivative of thickness (central difference)
    df['d_thickness_dt'] = df['thickness_frac'].diff() / time_per_frame

    return df


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_interphase_analysis(
    df: pd.DataFrame,
    bin_width: float = 0.006,
    tolerance: float = 0.02,
    interface_region: Tuple[float, float] = (0.0, 0.15),
    time_per_frame: float = 0.01,
    z_col: str = 'z-frac_coord'
) -> Dict:
    """
    Run complete interphase analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory dataframe
    bin_width : float
        Bin width for composition profiles (fractional)
    tolerance : float
        Tolerance for bulk stoichiometry matching
    interface_region : tuple
        (z_min, z_max) to search for Interface A
    time_per_frame : float
        Time between frames in ns
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    results : dict
        Contains all analysis results:
        - 'frames': frame numbers
        - 'bin_centers': bin positions
        - 'compositions': composition profiles (n_frames, n_bins, 3)
        - 'boundaries': boundary positions DataFrame
        - 'population': interphase population DataFrame
        - 'flux': flux DataFrame
        - 'growth': growth rate DataFrame
    """
    print("="*60)
    print("INTERPHASE ANALYSIS PIPELINE")
    print("="*60)

    # Step 1: Compute composition profiles
    print("\n[1/5] Computing composition profiles...")
    frames, bin_centers, compositions, counts = compute_all_composition_profiles(
        df, bin_width, z_col
    )

    # Step 2: Detect boundaries
    print("\n[2/5] Detecting interphase boundaries...")
    boundaries_df = detect_boundaries_all_frames(
        compositions, bin_centers, frames, tolerance, interface_region
    )

    # Report boundary detection success
    n_valid = boundaries_df['valid'].sum()
    n_total = len(boundaries_df)
    print(f"      Boundary detection: {n_valid}/{n_total} frames valid ({100*n_valid/n_total:.1f}%)")

    # Step 3: Track population
    print("\n[3/5] Tracking interphase population...")
    population_df = track_interphase_population_all_frames(df, boundaries_df, z_col)

    # Step 4: Calculate flux
    print("\n[4/5] Calculating flux across boundaries...")
    flux_df = calculate_flux_all_frames(df, boundaries_df, z_col)

    # Step 5: Compute growth rate
    print("\n[5/5] Computing growth rate...")
    growth_df = compute_growth_rate(boundaries_df, time_per_frame)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return {
        'frames': frames,
        'bin_centers': bin_centers,
        'compositions': compositions,
        'counts': counts,
        'boundaries': boundaries_df,
        'population': population_df,
        'flux': flux_df,
        'growth': growth_df
    }


def run_interphase_analysis_averaged(
    df: pd.DataFrame,
    bin_widths: list = None,
    n_phase_shifts: int = None,
    tolerance: float = 0.02,
    interface_region: Tuple[float, float] = (0.0, 0.15),
    time_per_frame: float = 0.01,
    z_col: str = 'z-frac_coord'
) -> Dict:
    """
    Run complete interphase analysis pipeline with multi-bin averaging.

    This version averages boundary detection across multiple bin configurations
    (varying bin widths and phase offsets) to reduce binning artifacts that
    can cause spurious spikes in flux and population data.

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory dataframe
    bin_widths : list
        List of bin widths to average over (default: [0.004, 0.005, 0.006, 0.007, 0.008])
    n_phase_shifts : int
        Number of phase shifts per bin width (default: 3)
    tolerance : float
        Tolerance for bulk stoichiometry matching
    interface_region : tuple
        (z_min, z_max) to search for Interface A
    time_per_frame : float
        Time between frames in ns
    z_col : str
        Column name for z-coordinate

    Returns
    -------
    results : dict
        Contains all analysis results:
        - 'frames': frame numbers
        - 'boundaries': boundary positions DataFrame (with averaged boundaries)
        - 'population': interphase population DataFrame
        - 'flux': flux DataFrame
        - 'growth': growth rate DataFrame
        - 'bin_configs': list of (bin_width, offset) configurations used
    """
    if bin_widths is None:
        bin_widths = DEFAULT_BIN_WIDTHS
    if n_phase_shifts is None:
        n_phase_shifts = DEFAULT_N_PHASE_SHIFTS

    configs = generate_bin_configurations(bin_widths, n_phase_shifts)

    print("="*60)
    print("INTERPHASE ANALYSIS PIPELINE (Multi-Bin Averaged)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Bin widths: {bin_widths}")
    print(f"  Phase shifts per width: {n_phase_shifts}")
    print(f"  Total configurations: {len(configs)}")

    # Step 1: Detect boundaries with multi-bin averaging
    print("\n[1/4] Detecting interphase boundaries (multi-bin averaged)...")
    boundaries_df = detect_boundaries_averaged_all_frames(
        df, bin_widths, n_phase_shifts, tolerance, interface_region, z_col
    )

    # Report boundary detection success
    n_valid = boundaries_df['valid'].sum()
    n_total = len(boundaries_df)
    print(f"      Boundary detection: {n_valid}/{n_total} frames valid ({100*n_valid/n_total:.1f}%)")

    # Report averaging statistics
    avg_configs = boundaries_df['n_valid_configs'].mean()
    avg_lower_std = boundaries_df['z_lower_std'].mean()
    avg_upper_std = boundaries_df['z_upper_std'].mean()
    print(f"      Avg valid configs per frame: {avg_configs:.1f}/{len(configs)}")
    print(f"      Avg boundary std: z_lower={avg_lower_std:.4f}, z_upper={avg_upper_std:.4f}")

    # Step 2: Track population using averaged boundaries
    print("\n[2/4] Tracking interphase population...")
    population_df = track_interphase_population_all_frames(df, boundaries_df, z_col)

    # Step 3: Calculate flux using averaged boundaries
    print("\n[3/4] Calculating flux across boundaries...")
    flux_df = calculate_flux_all_frames(df, boundaries_df, z_col)

    # Step 4: Compute growth rate
    print("\n[4/4] Computing growth rate...")
    growth_df = compute_growth_rate(boundaries_df, time_per_frame)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    frames = sorted(df['frame'].unique())

    return {
        'frames': np.array(frames),
        'boundaries': boundaries_df,
        'population': population_df,
        'flux': flux_df,
        'growth': growth_df,
        'bin_configs': configs
    }
