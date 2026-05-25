"""
Central config loader for gpu_mode_sac. Reads config.yaml once and caches the result.

Usage in any gpu_mode_sac module:
    from config_loader import cfg
    lr = cfg["sac"]["learning_rate"]
"""

import os

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required to load config.yaml. Install it with: pip install pyyaml"
    )

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
_cache: dict | None = None


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load and return the YAML config (cached after first call)."""
    global _cache
    if _cache is None:
        with open(path, "r") as f:
            _cache = yaml.safe_load(f)
    return _cache


cfg = load_config()
