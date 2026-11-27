"""Compatibility namespace.

The canonical implementation lives in :mod:`operator_quant_groups.api`.
This module re-exports the full public surface so callers using an
equivalent module path still resolve the same objects.
"""

from .api import *  # noqa: F401,F403
from .api import __all__ as __all__  # noqa: F401
from .api import _pattern_resolve


def __getattr__(name):
    return _pattern_resolve(name)
