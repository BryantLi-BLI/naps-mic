"""
Processing functions for Onsager transport analysis.
Core functions for creating analyzers and processing XDATCAR files.
"""

import os
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt

from py_oats import OnsagerTransportAnalyzer, plot_correlation_pairwise

from onsager_io import extract_metadata, save_transport_doc


def create_onsager_analyzer(path, temp, species):
    """
    Create OnsagerTransportAnalyzer from XDATCAR file.
    """
    analyzer = OnsagerTransportAnalyzer.from_xdatcar(
        path,
        temperature=temp,
        species=species,
        time_step=1,
        step_skip=1000
    )
    return analyzer


def save_msd_plot(analyzer, fig_path):
    """
    Generate and save MSD correlation plot.
    """
    plot_correlation_pairwise(analyzer, size=20)
    plt.savefig(f'{fig_path}.png', dpi=150, bbox_inches='tight')
    plt.close('all')


def extract_l_tensor_data(analyzer, species, metadata):
    """
    Extract L-tensor values from analyzer for CSV output.
    """
    L = analyzer.L_tensor
    L_self = analyzer.L_tensor_self

    result = {
        'filename': metadata['filename'],
        'Formula': metadata['reduced_formula'],
        'Temperature': f"{metadata['temperature']}K",
        'volume': metadata['volume'],
        'species_present': species,
        'L00': 0.0, 'L00_std': 0.0,
        'L01': 0.0, 'L01_std': 0.0,
        'L02': 0.0, 'L02_std': 0.0,
        'L11': 0.0, 'L11_std': 0.0,
        'L12': 0.0, 'L12_std': 0.0,
        'L22': 0.0, 'L22_std': 0.0,
        'L00_self': 0.0, 'L00_self_std': 0.0,
        'L11_self': 0.0, 'L11_self_std': 0.0,
        'L22_self': 0.0, 'L22_self_std': 0.0,
    }

    species_to_standard = {'Na': 0, 'P': 1, 'S': 2}
    n_species = len(species)

    for i in range(n_species):
        for j in range(n_species):
            std_i = species_to_standard[species[i]]
            std_j = species_to_standard[species[j]]
            if std_i <= std_j:
                key = f'L{std_i}{std_j}'
                result[key] = L[i, j]

    for i in range(n_species):
        std_i = species_to_standard[species[i]]
        key = f'L{std_i}{std_i}_self'
        result[key] = L_self[i, i]

    return result


def process_single_xdatcar(path, idx, figures_dir, output_dir):
    """
    Main function: Process a single XDATCAR file.

    Returns dict with L-tensor data on success, None on failure.
    """
    try:
        metadata = extract_metadata(path)
        analyzer = create_onsager_analyzer(path, metadata['temperature'], metadata['species'])

        output_path = f'{output_dir}/onsager_data_{idx}.json.gz'
        save_transport_doc(analyzer, output_path)

        fig_path = f'{figures_dir}/pairwise_msd_{idx}'
        save_msd_plot(analyzer, fig_path)

        result = extract_l_tensor_data(analyzer, metadata['species'], metadata)
        return result

    except Exception as e:
        print(f'[{idx}] ERROR processing {os.path.basename(path)}: {str(e)}')
        traceback.print_exc()
        return None
