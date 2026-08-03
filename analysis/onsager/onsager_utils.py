"""
Utility functions for Onsager transport analysis.
JSON serialization helpers for pymatgen objects.
"""

import numpy as np
from pymatgen.core import Element


def convert_key_to_str(key):
    """
    Convert any key type to a JSON-compatible string.
    """
    if isinstance(key, str):
        return key
    elif isinstance(key, (int, float, bool)):
        return key  # JSON supports these natively
    elif isinstance(key, tuple):
        # Convert tuple elements recursively
        return str(tuple(convert_key_to_str(k) for k in key))
    elif isinstance(key, Element):
        return str(key)  # Element.__str__ returns element symbol like 'Na'
    elif hasattr(key, 'symbol'):  # Species, Element, etc.
        return str(key.symbol)
    else:
        return str(key)


def convert_to_json_serializable(obj):
    """
    Recursively convert an object to be JSON serializable.
    Handles: tuple keys, Element keys, numpy arrays, pymatgen objects.
    """
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {convert_key_to_str(k): convert_to_json_serializable(v)
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, Element):
        return str(obj)
    elif hasattr(obj, 'as_dict'):  # Pymatgen objects with as_dict method
        return convert_to_json_serializable(obj.as_dict())
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Last resort: convert to string
        return str(obj)


def formula_to_latex(formula):
    """Convert chemical formula to LaTeX format for plotting."""
    latex_formula = ''
    i = 0
    while i < len(formula):
        if formula[i].isdigit():
            number = ''
            while i < len(formula) and formula[i].isdigit():
                number += formula[i]
                i += 1
            latex_formula += f'_{{{number}}}'
        else:
            latex_formula += f'{{{formula[i]}}}'
            i += 1
    return f'${latex_formula}$'
