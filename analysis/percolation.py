#!/usr/bin/env python3
"""
Percolation analysis for Na/Na3PS4 interphase region.

Two analyses:
1) P-P only percolation: direct P-P bonds (reduced phosphorus species)
2) Mixed Na-P network percolation: P-P and Na-P edges

Outputs:
- Percolation results per frame
- Percolating cluster atom indices + LAMMPS dump for OVITO visualization
- Percolation probability vs cutoff distance (independent and joint sweeps)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================

DUMP_FILE = ""  # path to master_coordinates.dump (download from MPContribs — see top-level README.md)
OUTPUT_DIR = "percolation_results"

# Fractional coordinate bounds for interphase region in z
Z_FRAC_MIN = 0.54
Z_FRAC_MAX = 0.60

# Default cutoffs (Angstroms)
PP_CUTOFF = 2.5
NAP_CUTOFF = 3.2

# Source: z_frac > Z_FRAC_MAX (bulk Na3PS4 side)
# Sink:   z_frac < Z_FRAC_MIN (Na metal side)
# Within the interphase, source atoms are near z_frac ~ 0.60, sink atoms near z_frac ~ 0.54

# For the interphase region, define "source layer" and "sink layer"
# Source layer: top 10% of interphase (near Na3PS4 side)
# Sink layer: bottom 10% of interphase (near Na side)
SOURCE_FRAC_MIN = 0.59  # atoms near z=0.60 boundary
SINK_FRAC_MAX = 0.55    # atoms near z=0.54 boundary

# Cutoff sweep ranges
PP_SWEEP = np.arange(2.0, 3.6, 0.1)
NAP_SWEEP = np.arange(2.5, 4.1, 0.1)

# Joint sweep: use same ranges
PP_JOINT_SWEEP = np.arange(2.0, 3.6, 0.25)
NAP_JOINT_SWEEP = np.arange(2.5, 4.1, 0.25)


# =============================================================================
# PARSING
# =============================================================================

def parse_dump(filename):
    """
    Generator that yields one frame at a time from a LAMMPS dump file.
    
    Yields:
        dict with keys:
            'timestep': int
            'natoms': int
            'box_bounds': np.array of shape (3,2) [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
            'ids': np.array of int
            'elements': np.array of str
            'coords': np.array of shape (N,3) - Cartesian coordinates
    """
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                
                f.readline()  # ITEM: NUMBER OF ATOMS
                natoms = int(f.readline().strip())
                
                f.readline()  # ITEM: BOX BOUNDS
                box_bounds = np.zeros((3, 2))
                for dim in range(3):
                    parts = f.readline().strip().split()
                    box_bounds[dim, 0] = float(parts[0])
                    box_bounds[dim, 1] = float(parts[1])
                
                f.readline()  # ITEM: ATOMS header
                
                ids = np.zeros(natoms, dtype=int)
                elements = []
                coords = np.zeros((natoms, 3))
                
                for i in range(natoms):
                    parts = f.readline().strip().split()
                    ids[i] = int(parts[0])
                    elements.append(parts[1])
                    coords[i, 0] = float(parts[2])
                    coords[i, 1] = float(parts[3])
                    coords[i, 2] = float(parts[4])
                
                yield {
                    'timestep': timestep,
                    'natoms': natoms,
                    'box_bounds': box_bounds,
                    'ids': ids,
                    'elements': np.array(elements),
                    'coords': coords
                }


def filter_interphase_atoms(frame):
    """
    Filter atoms to only those within the interphase z-region.
    Returns filtered arrays for P and Na atoms separately, plus combined.
    """
    box_z_lo = frame['box_bounds'][2, 0]
    box_z_hi = frame['box_bounds'][2, 1]
    box_z_len = box_z_hi - box_z_lo
    
    # Convert z to fractional
    z_frac = (frame['coords'][:, 2] - box_z_lo) / box_z_len
    
    # Interphase mask
    interphase_mask = (z_frac >= Z_FRAC_MIN) & (z_frac <= Z_FRAC_MAX)
    
    elements = frame['elements']
    
    # P atoms in interphase
    p_mask = interphase_mask & (elements == 'P')
    # Na atoms in interphase
    na_mask = interphase_mask & (elements == 'Na')
    
    return {
        'p_indices': np.where(p_mask)[0],
        'na_indices': np.where(na_mask)[0],
        'all_indices': np.where(interphase_mask & ((elements == 'P') | (elements == 'Na')))[0],
        'p_coords': frame['coords'][p_mask],
        'na_coords': frame['coords'][na_mask],
        'p_ids': frame['ids'][p_mask],
        'na_ids': frame['ids'][na_mask],
        'p_z_frac': z_frac[p_mask],
        'na_z_frac': z_frac[na_mask],
        'box_bounds': frame['box_bounds'],
        'z_frac_all': z_frac,
        'box_z_len': box_z_len,
        'box_z_lo': box_z_lo,
    }


# =============================================================================
# NEIGHBOR SEARCH WITH PBC IN X,Y ONLY
# =============================================================================

def build_pbc_tree(coords, box_bounds):
    """
    Build a KD-tree that handles PBC in x and y directions.
    We replicate atoms in x and y to handle periodic boundaries.
    
    For efficiency, we only replicate atoms near the edges.
    """
    box_lengths = box_bounds[:, 1] - box_bounds[:, 0]
    lx, ly = box_lengths[0], box_lengths[1]
    
    # Instead of replicating, we use the minimum image convention
    # by wrapping coordinates. scipy cKDTree doesn't natively support PBC
    # in select dimensions, so we handle this manually.
    
    # Wrap coords into box
    wrapped = coords.copy()
    wrapped[:, 0] = (wrapped[:, 0] - box_bounds[0, 0]) % lx + box_bounds[0, 0]
    wrapped[:, 1] = (wrapped[:, 1] - box_bounds[1, 0]) % ly + box_bounds[1, 0]
    
    return wrapped, box_lengths


def find_neighbors_pbc_xy(coords_a, coords_b, cutoff, box_bounds):
    """
    Find all pairs (i in A, j in B) within cutoff, with PBC in x and y.
    
    Returns list of (i, j) pairs.
    """
    box_lengths = box_bounds[:, 1] - box_bounds[:, 0]
    lx, ly = box_lengths[0], box_lengths[1]
    
    if len(coords_a) == 0 or len(coords_b) == 0:
        return []
    
    # Wrap coordinates
    a_wrapped = coords_a.copy()
    a_wrapped[:, 0] = (a_wrapped[:, 0] - box_bounds[0, 0]) % lx
    a_wrapped[:, 1] = (a_wrapped[:, 1] - box_bounds[1, 0]) % ly
    a_wrapped[:, 2] = a_wrapped[:, 2] - box_bounds[2, 0]  # shift but no wrap
    
    b_wrapped = coords_b.copy()
    b_wrapped[:, 0] = (b_wrapped[:, 0] - box_bounds[0, 0]) % lx
    b_wrapped[:, 1] = (b_wrapped[:, 1] - box_bounds[1, 0]) % ly
    b_wrapped[:, 2] = b_wrapped[:, 2] - box_bounds[2, 0]
    
    # Create replicas of B for PBC in x,y
    shifts = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shifts.append([dx * lx, dy * ly, 0.0])
    shifts = np.array(shifts)
    
    # Build replicated B
    n_b = len(b_wrapped)
    b_replicated = np.vstack([b_wrapped + s for s in shifts])
    
    tree = cKDTree(b_replicated)
    pairs = []
    
    # Query all A atoms at once
    results = tree.query_ball_point(a_wrapped, cutoff)
    
    for i, neighbors in enumerate(results):
        for j_rep in neighbors:
            j_orig = j_rep % n_b
            pairs.append((i, j_orig))
    
    # Remove duplicates (same pair can appear from multiple images)
    pairs = list(set(pairs))
    
    return pairs


def find_self_neighbors_pbc_xy(coords, cutoff, box_bounds):
    """
    Find all pairs (i, j) with i < j within the same set of coords,
    with PBC in x and y.
    """
    pairs = find_neighbors_pbc_xy(coords, coords, cutoff, box_bounds)
    # Remove self-pairs and keep only i < j
    filtered = set()
    for i, j in pairs:
        if i != j:
            filtered.add((min(i, j), max(i, j)))
    return list(filtered)


# =============================================================================
# GRAPH CONSTRUCTION AND PERCOLATION CHECK
# =============================================================================

def check_pp_percolation(filtered, cutoff=PP_CUTOFF):
    """
    Analysis 1: P-P only percolation.
    
    Build a graph with only P atoms as nodes, P-P edges within cutoff.
    Check if any connected component spans from source layer to sink layer.
    
    Returns:
        percolates: bool
        largest_z_span: float (fractional z-span of the largest cluster)
        min_z_reached: float (minimum z_frac reached by any cluster touching source)
        cluster_atom_ids: list of atom IDs in the percolating cluster (or largest)
        cluster_info: dict with additional details
    """
    p_coords = filtered['p_coords']
    p_z_frac = filtered['p_z_frac']
    p_ids = filtered['p_ids']
    n_p = len(p_coords)
    
    if n_p == 0:
        return False, 0.0, Z_FRAC_MAX, [], {}
    
    # Find P-P neighbors
    pp_pairs = find_self_neighbors_pbc_xy(p_coords, cutoff, filtered['box_bounds'])
    
    # Build adjacency matrix
    adj = lil_matrix((n_p, n_p), dtype=bool)
    for i, j in pp_pairs:
        adj[i, j] = True
        adj[j, i] = True
    
    adj_csr = adj.tocsr()
    
    # Find connected components
    n_components, labels = connected_components(adj_csr, directed=False)
    
    # Define source and sink atoms
    source_mask = p_z_frac >= SOURCE_FRAC_MIN
    sink_mask = p_z_frac <= SINK_FRAC_MAX
    
    # Check each component for percolation
    percolates = False
    best_cluster_ids = []
    best_z_span = 0.0
    min_z_reached = Z_FRAC_MAX
    
    for comp_id in range(n_components):
        comp_mask = labels == comp_id
        comp_z = p_z_frac[comp_mask]
        
        if len(comp_z) == 0:
            continue
        
        z_span = comp_z.max() - comp_z.min()
        
        has_source = np.any(comp_mask & source_mask)
        has_sink = np.any(comp_mask & sink_mask)
        
        if has_source and has_sink:
            percolates = True
            if z_span > best_z_span:
                best_z_span = z_span
                best_cluster_ids = p_ids[comp_mask].tolist()
                min_z_reached = comp_z.min()
        elif has_source:
            if comp_z.min() < min_z_reached:
                min_z_reached = comp_z.min()
        
        if z_span > best_z_span and not percolates:
            best_z_span = z_span
            best_cluster_ids = p_ids[comp_mask].tolist()
    
    # If percolation found, re-identify the percolating cluster
    if percolates:
        pass  # already set above
    
    cluster_info = {
        'n_p_interphase': n_p,
        'n_pp_edges': len(pp_pairs),
        'n_components': n_components,
        'largest_component_size': max(np.bincount(labels)) if n_p > 0 else 0,
    }
    
    return percolates, best_z_span, min_z_reached, best_cluster_ids, cluster_info


def check_nap_percolation(filtered, pp_cutoff=PP_CUTOFF, nap_cutoff=NAP_CUTOFF):
    """
    Analysis 2: Mixed Na-P network percolation.
    
    Nodes: all Na and P atoms in the interphase.
    Edges: P-P within pp_cutoff, Na-P within nap_cutoff. No Na-Na edges.
    
    Source: atoms (Na or P) near z_frac ~ 0.60 (Na3PS4 side)
    Sink: atoms (Na or P) near z_frac ~ 0.54 (Na side)
    
    Returns same format as check_pp_percolation.
    """
    p_coords = filtered['p_coords']
    na_coords = filtered['na_coords']
    p_z_frac = filtered['p_z_frac']
    na_z_frac = filtered['na_z_frac']
    p_ids = filtered['p_ids']
    na_ids = filtered['na_ids']
    
    n_p = len(p_coords)
    n_na = len(na_coords)
    n_total = n_p + n_na
    
    if n_total == 0:
        return False, 0.0, Z_FRAC_MAX, [], {}
    
    # Combined arrays: indices 0..n_p-1 are P, n_p..n_total-1 are Na
    all_z_frac = np.concatenate([p_z_frac, na_z_frac])
    all_ids = np.concatenate([p_ids, na_ids])
    all_elements = np.array(['P'] * n_p + ['Na'] * n_na)
    
    # Build adjacency matrix
    adj = lil_matrix((n_total, n_total), dtype=bool)
    
    # P-P edges
    if n_p > 0:
        pp_pairs = find_self_neighbors_pbc_xy(p_coords, pp_cutoff, filtered['box_bounds'])
        for i, j in pp_pairs:
            adj[i, j] = True
            adj[j, i] = True
    
    # Na-P edges
    if n_p > 0 and n_na > 0:
        nap_pairs = find_neighbors_pbc_xy(na_coords, p_coords, nap_cutoff, filtered['box_bounds'])
        for na_local, p_local in nap_pairs:
            na_idx = n_p + na_local  # offset by n_p
            p_idx = p_local
            adj[na_idx, p_idx] = True
            adj[p_idx, na_idx] = True
    
    adj_csr = adj.tocsr()
    
    # Find connected components
    n_components, labels = connected_components(adj_csr, directed=False)
    
    # Source and sink
    source_mask = all_z_frac >= SOURCE_FRAC_MIN
    sink_mask = all_z_frac <= SINK_FRAC_MAX
    
    percolates = False
    best_cluster_ids = []
    best_cluster_elements = []
    best_z_span = 0.0
    min_z_reached = Z_FRAC_MAX
    
    for comp_id in range(n_components):
        comp_mask = labels == comp_id
        comp_z = all_z_frac[comp_mask]
        
        if len(comp_z) == 0:
            continue
        
        z_span = comp_z.max() - comp_z.min()
        
        has_source = np.any(comp_mask & source_mask)
        has_sink = np.any(comp_mask & sink_mask)
        
        if has_source and has_sink:
            percolates = True
            if z_span > best_z_span:
                best_z_span = z_span
                best_cluster_ids = all_ids[comp_mask].tolist()
                best_cluster_elements = all_elements[comp_mask].tolist()
                min_z_reached = comp_z.min()
        elif has_source:
            if comp_z.min() < min_z_reached:
                min_z_reached = comp_z.min()
        
        if z_span > best_z_span and not percolates:
            best_z_span = z_span
            best_cluster_ids = all_ids[comp_mask].tolist()
            best_cluster_elements = all_elements[comp_mask].tolist()
    
    n_pp = len(pp_pairs) if n_p > 0 else 0
    n_nap = len(nap_pairs) if (n_p > 0 and n_na > 0) else 0
    
    cluster_info = {
        'n_p_interphase': n_p,
        'n_na_interphase': n_na,
        'n_pp_edges': n_pp,
        'n_nap_edges': n_nap,
        'n_components': n_components,
        'largest_component_size': max(np.bincount(labels)) if n_total > 0 else 0,
    }
    
    return percolates, best_z_span, min_z_reached, best_cluster_ids, cluster_info


# =============================================================================
# VISUALIZATION OUTPUT
# =============================================================================

def write_percolation_dump(filename, frame, cluster_atom_ids, analysis_type, 
                           filtered, cluster_elements=None):
    """
    Write a LAMMPS dump file for OVITO visualization.
    Contains all atoms in the interphase region, with a custom column
    'percolation' = 1 for atoms in the percolating/largest cluster, 0 otherwise.
    Also 'atom_type_flag': 1=P, 2=Na, 3=S (for coloring in OVITO).
    """
    box_bounds = frame['box_bounds']
    
    # Get all interphase atoms
    box_z_lo = box_bounds[2, 0]
    box_z_len = box_bounds[2, 1] - box_z_lo
    z_frac = (frame['coords'][:, 2] - box_z_lo) / box_z_len
    interphase_mask = (z_frac >= Z_FRAC_MIN) & (z_frac <= Z_FRAC_MAX)
    
    interphase_indices = np.where(interphase_mask)[0]
    
    cluster_id_set = set(cluster_atom_ids)
    
    with open(filename, 'a') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{frame['timestep']}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(interphase_indices)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for dim in range(3):
            f.write(f"{box_bounds[dim, 0]:.10e} {box_bounds[dim, 1]:.10e}\n")
        f.write("ITEM: ATOMS id element x y z z_frac percolation\n")
        
        for idx in interphase_indices:
            atom_id = frame['ids'][idx]
            element = frame['elements'][idx]
            x, y, z = frame['coords'][idx]
            zf = z_frac[idx]
            perc = 1 if atom_id in cluster_id_set else 0
            f.write(f"{atom_id} {element} {x:.6f} {y:.6f} {z:.6f} {zf:.6f} {perc}\n")


# =============================================================================
# CUTOFF SWEEP
# =============================================================================

def sweep_pp_cutoff(filtered, pp_range):
    """Sweep P-P cutoff for P-P only percolation."""
    results = []
    for pp_cut in pp_range:
        perc, z_span, min_z, _, info = check_pp_percolation(filtered, cutoff=pp_cut)
        results.append({
            'pp_cutoff': pp_cut,
            'percolates': perc,
            'z_span': z_span,
            'min_z_reached': min_z,
            'n_pp_edges': info.get('n_pp_edges', 0),
            'largest_component': info.get('largest_component_size', 0),
        })
    return results


def sweep_nap_cutoff_independent(filtered, pp_range, nap_range, fixed_pp=PP_CUTOFF, fixed_nap=NAP_CUTOFF):
    """
    Sweep P-P and Na-P cutoffs independently for the mixed network.
    """
    results_pp_sweep = []
    results_nap_sweep = []
    
    # Sweep P-P with fixed Na-P
    for pp_cut in pp_range:
        perc, z_span, min_z, _, info = check_nap_percolation(filtered, pp_cutoff=pp_cut, nap_cutoff=fixed_nap)
        results_pp_sweep.append({
            'pp_cutoff': pp_cut,
            'nap_cutoff': fixed_nap,
            'percolates': perc,
            'z_span': z_span,
            'min_z_reached': min_z,
            'n_pp_edges': info.get('n_pp_edges', 0),
            'n_nap_edges': info.get('n_nap_edges', 0),
            'largest_component': info.get('largest_component_size', 0),
        })
    
    # Sweep Na-P with fixed P-P
    for nap_cut in nap_range:
        perc, z_span, min_z, _, info = check_nap_percolation(filtered, pp_cutoff=fixed_pp, nap_cutoff=nap_cut)
        results_nap_sweep.append({
            'pp_cutoff': fixed_pp,
            'nap_cutoff': nap_cut,
            'percolates': perc,
            'z_span': z_span,
            'min_z_reached': min_z,
            'n_pp_edges': info.get('n_pp_edges', 0),
            'n_nap_edges': info.get('n_nap_edges', 0),
            'largest_component': info.get('largest_component_size', 0),
        })
    
    return results_pp_sweep, results_nap_sweep


def sweep_joint(filtered, pp_range, nap_range):
    """Joint 2D sweep of both cutoffs."""
    results = []
    for pp_cut in pp_range:
        for nap_cut in nap_range:
            perc, z_span, min_z, _, info = check_nap_percolation(
                filtered, pp_cutoff=pp_cut, nap_cutoff=nap_cut
            )
            results.append({
                'pp_cutoff': pp_cut,
                'nap_cutoff': nap_cut,
                'percolates': perc,
                'z_span': z_span,
                'min_z_reached': min_z,
                'largest_component': info.get('largest_component_size', 0),
            })
    return results


# =============================================================================
# SINGLE-FRAME PROCESSING (for parallelization)
# =============================================================================

def process_single_frame(frame_data, pp_cutoff=PP_CUTOFF, nap_cutoff=NAP_CUTOFF):
    """
    Process a single frame for both P-P and Na-P percolation.
    Designed for use with multiprocessing.

    Parameters
    ----------
    frame_data : dict
        Frame data from parse_dump generator
    pp_cutoff : float
        P-P cutoff distance
    nap_cutoff : float
        Na-P cutoff distance

    Returns
    -------
    dict with all results for this frame
    """
    timestep = frame_data['timestep']
    filtered = filter_interphase_atoms(frame_data)

    # P-P percolation
    pp_perc, pp_zspan, pp_minz, pp_cluster_ids, pp_info = check_pp_percolation(
        filtered, cutoff=pp_cutoff
    )

    # Na-P percolation
    nap_perc, nap_zspan, nap_minz, nap_cluster_ids, nap_info = check_nap_percolation(
        filtered, pp_cutoff=pp_cutoff, nap_cutoff=nap_cutoff
    )

    return {
        'timestep': timestep,
        'filtered': filtered,
        'frame_data': frame_data,
        'pp': {
            'percolates': pp_perc,
            'z_span': pp_zspan,
            'min_z_reached': pp_minz,
            'cluster_ids': pp_cluster_ids,
            'info': pp_info,
        },
        'nap': {
            'percolates': nap_perc,
            'z_span': nap_zspan,
            'min_z_reached': nap_minz,
            'cluster_ids': nap_cluster_ids,
            'info': nap_info,
        }
    }


def load_all_frames(dump_file, max_frames=None, progress=True):
    """
    Load all frames from dump file into memory.

    Parameters
    ----------
    dump_file : str
        Path to LAMMPS dump file
    max_frames : int, optional
        Maximum number of frames to load (None for all)
    progress : bool
        Whether to print progress

    Returns
    -------
    list of frame dicts
    """
    frames = []
    for i, frame in enumerate(parse_dump(dump_file)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame)
        if progress and (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1} frames...")
    if progress:
        print(f"  Total frames loaded: {len(frames)}")
    return frames


def process_frames_parallel(frames, n_workers=None, pp_cutoff=PP_CUTOFF, nap_cutoff=NAP_CUTOFF):
    """
    Process multiple frames in parallel using multiprocessing.

    Parameters
    ----------
    frames : list
        List of frame dicts from load_all_frames
    n_workers : int, optional
        Number of worker processes (default: cpu_count)
    pp_cutoff : float
        P-P cutoff distance
    nap_cutoff : float
        Na-P cutoff distance

    Returns
    -------
    list of result dicts (same order as input frames)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), len(frames))

    print(f"Processing {len(frames)} frames with {n_workers} workers...")

    # Create partial function with fixed cutoffs
    process_func = partial(process_single_frame, pp_cutoff=pp_cutoff, nap_cutoff=nap_cutoff)

    with Pool(n_workers) as pool:
        results = pool.map(process_func, frames)

    return results


def run_cutoff_sweep_multiframe(frames, pp_range, nap_range, n_workers=None,
                                 sweep_frames=None, analysis_type='both'):
    """
    Run cutoff sweep on multiple frames for robust statistics.

    Parameters
    ----------
    frames : list
        List of frame dicts
    pp_range : array-like
        P-P cutoff values to sweep
    nap_range : array-like
        Na-P cutoff values to sweep
    n_workers : int, optional
        Number of workers for parallel processing
    sweep_frames : list, optional
        Indices of frames to use for sweep (default: every 100th frame)
    analysis_type : str
        'pp' for P-P only, 'nap' for mixed, 'both' for both analyses

    Returns
    -------
    dict with sweep results
    """
    if sweep_frames is None:
        # Default: every 100th frame (roughly 10 frames for 1051 total)
        sweep_frames = list(range(0, len(frames), max(1, len(frames) // 10)))

    print(f"Running cutoff sweep on {len(sweep_frames)} frames...")

    results = {
        'pp_sweep': [],           # P-P only analysis
        'nap_pp_sweep': [],       # Mixed analysis, P-P sweep with fixed Na-P
        'nap_nap_sweep': [],      # Mixed analysis, Na-P sweep with fixed P-P
        'joint_sweep': [],        # Joint 2D sweep
    }

    selected_frames = [frames[i] for i in sweep_frames]

    # P-P only sweep
    if analysis_type in ['pp', 'both']:
        print("\n  P-P cutoff sweep (P-P only analysis)...")
        for pp_cut in pp_range:
            perc_count = 0
            z_spans = []
            for frame in selected_frames:
                filtered = filter_interphase_atoms(frame)
                perc, z_span, _, _, info = check_pp_percolation(filtered, cutoff=pp_cut)
                if perc:
                    perc_count += 1
                z_spans.append(z_span)

            results['pp_sweep'].append({
                'pp_cutoff': pp_cut,
                'percolation_fraction': perc_count / len(selected_frames),
                'mean_z_span': np.mean(z_spans),
                'std_z_span': np.std(z_spans),
                'n_frames': len(selected_frames),
            })
            print(f"    PP={pp_cut:.2f} Å: {100*perc_count/len(selected_frames):.1f}% percolate")

    # Mixed Na-P sweeps
    if analysis_type in ['nap', 'both']:
        # P-P sweep with fixed Na-P
        print(f"\n  P-P sweep (mixed analysis, Na-P fixed at {NAP_CUTOFF} Å)...")
        for pp_cut in pp_range:
            perc_count = 0
            z_spans = []
            for frame in selected_frames:
                filtered = filter_interphase_atoms(frame)
                perc, z_span, _, _, info = check_nap_percolation(
                    filtered, pp_cutoff=pp_cut, nap_cutoff=NAP_CUTOFF
                )
                if perc:
                    perc_count += 1
                z_spans.append(z_span)

            results['nap_pp_sweep'].append({
                'pp_cutoff': pp_cut,
                'nap_cutoff': NAP_CUTOFF,
                'percolation_fraction': perc_count / len(selected_frames),
                'mean_z_span': np.mean(z_spans),
                'n_frames': len(selected_frames),
            })
            print(f"    PP={pp_cut:.2f} Å: {100*perc_count/len(selected_frames):.1f}% percolate")

        # Na-P sweep with fixed P-P
        print(f"\n  Na-P sweep (mixed analysis, P-P fixed at {PP_CUTOFF} Å)...")
        for nap_cut in nap_range:
            perc_count = 0
            z_spans = []
            for frame in selected_frames:
                filtered = filter_interphase_atoms(frame)
                perc, z_span, _, _, info = check_nap_percolation(
                    filtered, pp_cutoff=PP_CUTOFF, nap_cutoff=nap_cut
                )
                if perc:
                    perc_count += 1
                z_spans.append(z_span)

            results['nap_nap_sweep'].append({
                'pp_cutoff': PP_CUTOFF,
                'nap_cutoff': nap_cut,
                'percolation_fraction': perc_count / len(selected_frames),
                'mean_z_span': np.mean(z_spans),
                'n_frames': len(selected_frames),
            })
            print(f"    NaP={nap_cut:.2f} Å: {100*perc_count/len(selected_frames):.1f}% percolate")

        # Joint 2D sweep
        print("\n  Joint 2D cutoff sweep...")
        for pp_cut in PP_JOINT_SWEEP:
            for nap_cut in NAP_JOINT_SWEEP:
                perc_count = 0
                for frame in selected_frames:
                    filtered = filter_interphase_atoms(frame)
                    perc, _, _, _, _ = check_nap_percolation(
                        filtered, pp_cutoff=pp_cut, nap_cutoff=nap_cut
                    )
                    if perc:
                        perc_count += 1

                results['joint_sweep'].append({
                    'pp_cutoff': pp_cut,
                    'nap_cutoff': nap_cut,
                    'percolation_fraction': perc_count / len(selected_frames),
                    'n_frames': len(selected_frames),
                })
        print(f"    Completed {len(PP_JOINT_SWEEP)}x{len(NAP_JOINT_SWEEP)} grid")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Output files
    pp_results_file = os.path.join(OUTPUT_DIR, "pp_percolation_results.csv")
    nap_results_file = os.path.join(OUTPUT_DIR, "nap_percolation_results.csv")
    pp_dump_file = os.path.join(OUTPUT_DIR, "pp_percolation_clusters.dump")
    nap_dump_file = os.path.join(OUTPUT_DIR, "nap_percolation_clusters.dump")
    
    # Sweep output files
    pp_sweep_file = os.path.join(OUTPUT_DIR, "pp_cutoff_sweep.csv")
    nap_ind_pp_sweep_file = os.path.join(OUTPUT_DIR, "nap_independent_pp_sweep.csv")
    nap_ind_nap_sweep_file = os.path.join(OUTPUT_DIR, "nap_independent_nap_sweep.csv")
    joint_sweep_file = os.path.join(OUTPUT_DIR, "joint_cutoff_sweep.csv")
    
    # Clear dump files if they exist from previous runs
    for f in [pp_dump_file, nap_dump_file]:
        if os.path.exists(f):
            os.remove(f)
    
    # Write CSV headers
    with open(pp_results_file, 'w') as f:
        f.write("frame,timestep,n_p,n_pp_edges,n_components,largest_component,"
                "percolates,z_span,min_z_reached\n")
    
    with open(nap_results_file, 'w') as f:
        f.write("frame,timestep,n_p,n_na,n_pp_edges,n_nap_edges,n_components,"
                "largest_component,percolates,z_span,min_z_reached\n")
    
    print("=" * 70)
    print("PERCOLATION ANALYSIS: Na/Na3PS4 INTERPHASE")
    print("=" * 70)
    print(f"Interphase region: z_frac = [{Z_FRAC_MIN}, {Z_FRAC_MAX}]")
    print(f"Default P-P cutoff: {PP_CUTOFF} Å")
    print(f"Default Na-P cutoff: {NAP_CUTOFF} Å")
    print(f"Source layer: z_frac >= {SOURCE_FRAC_MIN}")
    print(f"Sink layer:   z_frac <= {SINK_FRAC_MAX}")
    print()
    
    frame_count = 0
    pp_perc_count = 0
    nap_perc_count = 0
    
    sweep_frame_data = None
    sweep_frame_raw = None
    
    t_start = time.time()
    
    # =========================================================================
    # MAIN LOOP: per-frame analysis
    # =========================================================================
    for frame in parse_dump(DUMP_FILE):
        t_frame_start = time.time()
        
        timestep = frame['timestep']
        filtered = filter_interphase_atoms(frame)
        
        # ---- Analysis 1: P-P percolation ----
        pp_perc, pp_zspan, pp_minz, pp_cluster_ids, pp_info = check_pp_percolation(
            filtered, cutoff=PP_CUTOFF
        )
        
        if pp_perc:
            pp_perc_count += 1
        
        with open(pp_results_file, 'a') as f:
            f.write(f"{frame_count},{timestep},{pp_info['n_p_interphase']},"
                    f"{pp_info['n_pp_edges']},{pp_info['n_components']},"
                    f"{pp_info['largest_component_size']},"
                    f"{pp_perc},{pp_zspan:.6f},{pp_minz:.6f}\n")
        
        write_percolation_dump(pp_dump_file, frame, pp_cluster_ids, 'PP', filtered)
        
        # ---- Analysis 2: Mixed Na-P percolation ----
        nap_perc, nap_zspan, nap_minz, nap_cluster_ids, nap_info = check_nap_percolation(
            filtered, pp_cutoff=PP_CUTOFF, nap_cutoff=NAP_CUTOFF
        )
        
        if nap_perc:
            nap_perc_count += 1
        
        with open(nap_results_file, 'a') as f:
            f.write(f"{frame_count},{timestep},{nap_info['n_p_interphase']},"
                    f"{nap_info['n_na_interphase']},{nap_info['n_pp_edges']},"
                    f"{nap_info['n_nap_edges']},{nap_info['n_components']},"
                    f"{nap_info['largest_component_size']},"
                    f"{nap_perc},{nap_zspan:.6f},{nap_minz:.6f}\n")
        
        write_percolation_dump(nap_dump_file, frame, nap_cluster_ids, 'NaP', filtered)
        
        # Save every frame's filtered data for sweep (overwrite; keeps last frame)
        sweep_frame_data = filtered
        sweep_frame_raw = frame
        
        t_frame_end = time.time()
        frame_count += 1
        
        if frame_count % 10 == 0 or frame_count == 1:
            elapsed = t_frame_end - t_start
            rate = frame_count / elapsed
            eta = (1051 - frame_count) / rate if rate > 0 else 0
            print(f"Frame {frame_count:4d} | ts={timestep:>10d} | "
                  f"P-P: {'PERC' if pp_perc else 'no  '} (span={pp_zspan:.4f}, "
                  f"minz={pp_minz:.4f}) | "
                  f"Na-P: {'PERC' if nap_perc else 'no  '} (span={nap_zspan:.4f}, "
                  f"minz={nap_minz:.4f}) | "
                  f"{t_frame_end - t_frame_start:.1f}s/frame | "
                  f"ETA: {eta/60:.1f} min")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print("PER-FRAME ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hr)")
    print(f"P-P  percolation: {pp_perc_count}/{frame_count} frames "
          f"({100*pp_perc_count/frame_count:.1f}%)")
    print(f"Na-P percolation: {nap_perc_count}/{frame_count} frames "
          f"({100*nap_perc_count/frame_count:.1f}%)")
    print()
    
    # =========================================================================
    # CUTOFF SENSITIVITY ANALYSIS (on last frame)
    # =========================================================================
    if sweep_frame_data is None:
        print("ERROR: No frames were processed. Cannot perform cutoff sweep.")
        return
    
    print("=" * 70)
    print(f"CUTOFF SENSITIVITY ANALYSIS (on last frame, ts={sweep_frame_raw['timestep']})")
    print("=" * 70)
    
    # ---- P-P only sweep ----
    print("\n--- P-P cutoff sweep (P-P only analysis) ---")
    pp_sweep_results = sweep_pp_cutoff(sweep_frame_data, PP_SWEEP)
    
    with open(pp_sweep_file, 'w') as f:
        f.write("pp_cutoff,percolates,z_span,min_z_reached,n_pp_edges,largest_component\n")
        for r in pp_sweep_results:
            f.write(f"{r['pp_cutoff']:.2f},{r['percolates']},{r['z_span']:.6f},"
                    f"{r['min_z_reached']:.6f},{r['n_pp_edges']},{r['largest_component']}\n")
            print(f"  PP={r['pp_cutoff']:.2f} Å | "
                  f"{'PERC' if r['percolates'] else 'no  '} | "
                  f"edges={r['n_pp_edges']:>6d} | "
                  f"largest={r['largest_component']:>6d} | "
                  f"span={r['z_span']:.4f} | minz={r['min_z_reached']:.4f}")
    
    print(f"\nSaved to {pp_sweep_file}")
    
    # ---- Independent Na-P sweeps ----
    print("\n--- Independent cutoff sweeps (mixed Na-P analysis) ---")
    results_pp_ind, results_nap_ind = sweep_nap_cutoff_independent(
        sweep_frame_data, PP_SWEEP, NAP_SWEEP,
        fixed_pp=PP_CUTOFF, fixed_nap=NAP_CUTOFF
    )
    
    # Write P-P sweep (Na-P fixed)
    with open(nap_ind_pp_sweep_file, 'w') as f:
        f.write("pp_cutoff,nap_cutoff,percolates,z_span,min_z_reached,"
                "n_pp_edges,n_nap_edges,largest_component\n")
        for r in results_pp_ind:
            f.write(f"{r['pp_cutoff']:.2f},{r['nap_cutoff']:.2f},{r['percolates']},"
                    f"{r['z_span']:.6f},{r['min_z_reached']:.6f},"
                    f"{r['n_pp_edges']},{r['n_nap_edges']},{r['largest_component']}\n")
    
    print(f"  P-P sweep (Na-P fixed at {NAP_CUTOFF} Å):")
    for r in results_pp_ind:
        print(f"    PP={r['pp_cutoff']:.2f} Å | "
              f"{'PERC' if r['percolates'] else 'no  '} | "
              f"PP_edges={r['n_pp_edges']:>6d} | NaP_edges={r['n_nap_edges']:>6d} | "
              f"largest={r['largest_component']:>6d}")
    
    print(f"  Saved to {nap_ind_pp_sweep_file}")
    
    # Write Na-P sweep (P-P fixed)
    with open(nap_ind_nap_sweep_file, 'w') as f:
        f.write("pp_cutoff,nap_cutoff,percolates,z_span,min_z_reached,"
                "n_pp_edges,n_nap_edges,largest_component\n")
        for r in results_nap_ind:
            f.write(f"{r['pp_cutoff']:.2f},{r['nap_cutoff']:.2f},{r['percolates']},"
                    f"{r['z_span']:.6f},{r['min_z_reached']:.6f},"
                    f"{r['n_pp_edges']},{r['n_nap_edges']},{r['largest_component']}\n")
    
    print(f"\n  Na-P sweep (P-P fixed at {PP_CUTOFF} Å):")
    for r in results_nap_ind:
        print(f"    NaP={r['nap_cutoff']:.2f} Å | "
              f"{'PERC' if r['percolates'] else 'no  '} | "
              f"PP_edges={r['n_pp_edges']:>6d} | NaP_edges={r['n_nap_edges']:>6d} | "
              f"largest={r['largest_component']:>6d}")
    
    print(f"  Saved to {nap_ind_nap_sweep_file}")
    
    # ---- Joint 2D sweep ----
    print("\n--- Joint 2D cutoff sweep ---")
    joint_results = sweep_joint(sweep_frame_data, PP_JOINT_SWEEP, NAP_JOINT_SWEEP)
    
    with open(joint_sweep_file, 'w') as f:
        f.write("pp_cutoff,nap_cutoff,percolates,z_span,min_z_reached,largest_component\n")
        for r in joint_results:
            f.write(f"{r['pp_cutoff']:.2f},{r['nap_cutoff']:.2f},{r['percolates']},"
                    f"{r['z_span']:.6f},{r['min_z_reached']:.6f},"
                    f"{r['largest_component']}\n")
    
    # Print joint sweep as a grid
    print(f"\n  Percolation grid (rows=Na-P cutoff, cols=P-P cutoff):")
    header = "  NaP\\PP |" + "|".join(f" {c:.2f} " for c in PP_JOINT_SWEEP) + "|"
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    joint_idx = 0
    for nap_cut in NAP_JOINT_SWEEP:
        row = f"  {nap_cut:.2f}  |"
        for pp_cut in PP_JOINT_SWEEP:
            r = joint_results[joint_idx]
            row += f"  {'Y' if r['percolates'] else '.'}   |"
            joint_idx += 1
        print(row)
    
    print(f"\n  Saved to {joint_sweep_file}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print(f"  Per-frame results:  {pp_results_file}")
    print(f"                      {nap_results_file}")
    print(f"  OVITO dump files:   {pp_dump_file}")
    print(f"                      {nap_dump_file}")
    print(f"  Cutoff sweeps:      {pp_sweep_file}")
    print(f"                      {nap_ind_pp_sweep_file}")
    print(f"                      {nap_ind_nap_sweep_file}")
    print(f"                      {joint_sweep_file}")
    print()
    print(f"P-P  percolation probability: {100*pp_perc_count/frame_count:.1f}% "
          f"({pp_perc_count}/{frame_count} frames)")
    print(f"Na-P percolation probability: {100*nap_perc_count/frame_count:.1f}% "
          f"({nap_perc_count}/{frame_count} frames)")
    print()
    print("Load the .dump files in OVITO and color by the 'percolation' column")
    print("to visualize percolating clusters.")
    print("=" * 70)


if __name__ == "__main__":
    main()