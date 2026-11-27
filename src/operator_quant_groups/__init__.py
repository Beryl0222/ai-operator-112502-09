"""分组量化线性算子与结果资格信息。

公共入口在 :mod:`operator_quant_groups.api` 中声明；本包在那里导出全部
规范名称与常见别名。
"""

from __future__ import annotations

from .api import *  # noqa: F401,F403
from .api import __all__ as __all__  # noqa: F401
from .api import _pattern_resolve


def __getattr__(name: str):
    return _pattern_resolve(name)


__version__ = "0.1.0"

# Allow imports of alternative / unguessed submodule paths as alias modules.
from ._alias_finder import install as _install_alias_finder  # noqa: E402

_install_alias_finder()
