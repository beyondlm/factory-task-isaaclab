"""Robot-independent helpers for human-gated VLA dataset aggregation.

Exports are loaded lazily so task specifications can be inspected in a minimal
Python environment without importing the NumPy-based data validator.
"""

from importlib import import_module
from typing import Any


__all__ = [
    "EpisodeAudit",
    "GripperCommandSpec",
    "PairedOutcomeSummary",
    "VLADAggerTaskSpec",
    "stable_inference_seed",
    "summarize_paired_outcomes",
    "validate_transition_arrays",
]

_EXPORT_MODULES = {
    "EpisodeAudit": ".data_contract",
    "GripperCommandSpec": ".task_spec",
    "PairedOutcomeSummary": ".evaluation",
    "VLADAggerTaskSpec": ".task_spec",
    "stable_inference_seed": ".evaluation",
    "summarize_paired_outcomes": ".evaluation",
    "validate_transition_arrays": ".data_contract",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
