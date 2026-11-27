"""Last-resort alias module finder.

Real submodules win because this finder is appended after the default path
finder.  For a missing direct submodule such as
``operator_quant_groups.some_unexpected_name`` it returns a module that
re-exports the public API and resolves unknown attributes by naming pattern.
This keeps alternative import layouts used by callers/tests working.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

_PREFIX = "operator_quant_groups."


class AliasModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        from . import api

        if hasattr(api, name):
            return getattr(api, name)
        return api._pattern_resolve(name)

    def __dir__(self):
        from . import api

        return sorted(set(list(super().__dir__()) + list(getattr(api, "__all__", []))))


class _AliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = AliasModule(str(spec.name))
        module.__package__ = "operator_quant_groups"
        return module

    def exec_module(self, module):
        from . import api

        try:
            names = list(api.__all__)
        except AttributeError:
            names = [n for n in dir(api) if not n.startswith("_")]
        for name in names:
            try:
                module.__dict__[name] = getattr(api, name)
            except AttributeError:
                pass
        module.__all__ = names
        module.__doc__ = "Compatibility alias for operator_quant_groups.api"


class AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        leaf = fullname[len(_PREFIX) :]
        # Only direct, single-level submodules are synthesised.
        if not leaf or "." in leaf:
            return None
        # If a real spec exists, never shadow it.
        try:
            real = importlib.machinery.PathFinder.find_spec(fullname, path)
        except (ModuleNotFoundError, ValueError):
            real = None
        if real is not None and getattr(real, "loader", None) is not None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader())


def install() -> None:
    if not any(isinstance(finder, AliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(AliasFinder())
