"""
PA-SSL: Configuration System

Loads YAML configs with defaults, supports CLI overrides, and ensures
reproducibility by logging the complete resolved config with each experiment.
"""

import yaml
import os
import json
import copy
import argparse
from pathlib import Path


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config. If None, loads default.
    
    Returns:
        dict: Complete configuration
    """
    default_path = Path(__file__).parent / 'default.yaml'
    
    # Load defaults
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with user config if provided
    if config_path is not None and config_path != str(default_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        config = deep_merge(config, user_config)
    
    return config


def deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config, output_dir):
    """Save resolved config to experiment directory for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as YAML
    yaml_path = os.path.join(output_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Also save as JSON for programmatic access
    json_path = os.path.join(output_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return yaml_path


def config_to_args(config):
    """Convert nested config dict to a flat argparse Namespace."""
    flat = {}
    
    def flatten(d, prefix=''):
        for key, value in d.items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, f"{full_key}.")
            else:
                flat[full_key] = value
    
    flatten(config)
    return argparse.Namespace(**flat)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
