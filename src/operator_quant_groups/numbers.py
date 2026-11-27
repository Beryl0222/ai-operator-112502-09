"""Shadow safe entry for the numeric kernel.

Two jobs:

* imported as part of this package (``operator_quant_groups.numbers``) it
  re-exports :mod:`operator_quant_groups.numeric_kernel`;
* if this file shadows the standard-library :mod:`numbers` module (interpreter
  launched from inside this directory), execute the real stdlib module in
  place so :mod:`fractions` and friends keep working.
"""

from __future__ import annotations

if __name__ == "numbers" and not __package__:  # pragma: no cover - env guard
    import sys as _sys

    _ver = "python%d.%d" % _sys.version_info[:2]
    _candidates = [
        _sys.base_prefix + "/lib/" + _ver + "/numbers.py",
        _sys.base_exec_prefix + "/lib/" + _ver + "/numbers.py",
        "/usr/lib/" + _ver + "/numbers.py",
    ]
    for _base in list(_sys.path):
        if _base:
            _candidates.append(_base.rstrip("/") + "/numbers.py")
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
    from .numeric_kernel import *  # noqa: F401,F403
    from . import numeric_kernel as _kernel

    def __getattr__(name: str):
        return getattr(_kernel, name)
