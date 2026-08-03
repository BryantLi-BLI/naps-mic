"""
I/O functions for Onsager transport analysis.
File reading, metadata extraction, and transport document handling.
"""

import os
import ast
import json
import gzip
import numpy as np
from pymatgen.io.vasp import Poscar
from py_oats.core.schema import TransportDoc

from onsager_utils import convert_to_json_serializable


def extract_metadata(path):
    """
    Extract metadata from XDATCAR file path and structure.

    Returns dict with: filename, temperature, species, reduced_formula, volume, structure
    """
    filename = os.path.basename(path)
    temp = float(path.split('_')[-2])

    poscar = Poscar.from_file(path)
    structure = poscar.structure

    all_species = ['Na', 'P', 'S']
    structure_elements = list(structure.composition.as_dict().keys())
    species = [s for s in all_species if s in structure_elements]

    reduced_formula = structure.composition.reduced_formula
    volume = structure.volume

    return {
        'filename': filename,
        'temperature': temp,
        'species': species,
        'reduced_formula': reduced_formula,
        'volume': volume,
        'structure': structure
    }


def save_transport_doc(analyzer, output_path):
    """
    Create TransportDoc from analyzer and save to compressed JSON.
    Uses custom serialization to handle Element keys, tuple keys, etc.
    """
    doc = TransportDoc.from_analyzer(analyzer)
    doc_dict = convert_to_json_serializable(doc)

    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(doc_dict, f)

    return doc


def convert_string_keys_to_tuples(d):
    """Convert string keys that look like tuples back to actual tuples."""
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for k, v in d.items():
        new_key = k
        if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
            try:
                new_key = ast.literal_eval(k)
            except:
                pass

        if isinstance(v, dict):
            new_dict[new_key] = convert_string_keys_to_tuples(v)
        elif isinstance(v, list):
            new_dict[new_key] = [convert_string_keys_to_tuples(item) if isinstance(item, dict) else item for item in v]
        else:
            new_dict[new_key] = v

    return new_dict


def load_transport_doc_dict(json_path):
    """
    Load TransportDoc data from saved JSON file as a dictionary.
    Handles key conversion and numpy array reconstruction.
    """
    with gzip.open(json_path, 'rt', encoding='utf-8') as f:
        doc_dict = json.load(f)

    doc_dict = convert_string_keys_to_tuples(doc_dict)

    if 'times' in doc_dict and isinstance(doc_dict['times'], list):
        doc_dict['times'] = np.array(doc_dict['times'])

    if 'msds' in doc_dict and isinstance(doc_dict['msds'], dict):
        for key in doc_dict['msds']:
            if isinstance(doc_dict['msds'][key], list):
                doc_dict['msds'][key] = np.array(doc_dict['msds'][key])

    return doc_dict


def get_processed_indices(output_dir):
    """
    Get set of already-processed file indices by scanning output directory.
    Used for resume capability.
    """
    from glob import glob

    existing_files = glob(f'{output_dir}/onsager_data_*.json.gz')
    processed_indices = set()

    for f in existing_files:
        filename = os.path.basename(f)
        try:
            idx = int(filename.replace('onsager_data_', '').replace('.json.gz', ''))
            processed_indices.add(idx)
        except ValueError:
            continue

    return processed_indices
