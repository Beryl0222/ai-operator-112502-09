"""Shadow safe entry for the grouped linear operator.

Re-exports :mod:`operator_quant_groups.linear_operator` when imported as a
package module.  If this file shadows the standard-library :mod:`operator`
(because the interpreter was launched from inside this directory), it
executes the real stdlib implementation in place so ``from operator import
eq`` keeps working.
"""

from __future__ import annotations

if __name__ == "operator" and not __package__:  # pragma: no cover - env guard
    import sys as _sys

    _ver = "python%d.%d" % _sys.version_info[:2]
    _candidates = [
        _sys.base_prefix + "/lib/" + _ver + "/operator.py",
        _sys.base_exec_prefix + "/lib/" + _ver + "/operator.py",
        "/usr/lib/" + _ver + "/operator.py",
    ]
    for _base in list(_sys.path):
        if _base:
            _candidates.append(_base.rstrip("/") + "/operator.py")
    for _path in _candidates:
        if _path == __file__:
            continue
        try:
            with open(_path, "rb") as _fh:
                _code = compile(_fh.read(), _path, "exec")
        except OSError:
            continue
        exec(_code, globals())
        break
else:
    from .linear_operator import *  # noqa: F401,F403
    from . import linear_operator as _impl

    def __getattr__(name: str):
        return getattr(_impl, name)
