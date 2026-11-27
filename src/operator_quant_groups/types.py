"""Shadow safe ``types`` alias / compatibility namespace.

When imported as part of the package this re-exports the public API.  If it
shadows the standard-library :mod:`types` (interpreter launched from this
directory), execute the real stdlib module in place.
"""

from __future__ import annotations

if __name__ == "types" and not __package__:  # pragma: no cover - env guard
    import sys as _sys

    _ver = "python%d.%d" % _sys.version_info[:2]
    _candidates = [
        _sys.base_prefix + "/lib/" + _ver + "/types.py",
        _sys.base_exec_prefix + "/lib/" + _ver + "/types.py",
        "/usr/lib/" + _ver + "/types.py",
    ]
    for _base in list(_sys.path):
        if _base:
            _candidates.append(_base.rstrip("/") + "/types.py")
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
    from .api import *  # noqa: F401,F403
    from .api import __all__ as __all__  # noqa: F401
    from .api import _pattern_resolve

    def __getattr__(name):
        return _pattern_resolve(name)
