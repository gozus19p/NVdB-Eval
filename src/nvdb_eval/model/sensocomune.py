"""Backward compatibility module for SensoComune data models.

This module re-exports all classes from their new dedicated modules
to maintain backward compatibility with existing code that imports
from model.sensocomune.

For new code, prefer importing directly from the model package:
    from src.model import Definition, Sense, Lemma, etc.
"""

# Re-export all classes for backward compatibility
from .definition import Definition
from .sense import Sense
from .acceptation import Acceptation
from .locution import Locution
from .lemma import Lemma
from .sensocomune_collection import SensoComune
from .lemma import objectId_from_dict

__all__ = [
    "Definition",
    "Sense",
    "Acceptation",
    "Locution",
    "Lemma",
    "SensoComune",
    "objectId_from_dict",
]
