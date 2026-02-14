"""根目录兼容层：让 `python -m llm4graphgen.*` 在未安装时可运行。"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__all__ = ["__version__"]
__version__ = "0.1.0"

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
_src_pkg = Path(__file__).resolve().parent.parent / "src" / "llm4graphgen"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[attr-defined]
