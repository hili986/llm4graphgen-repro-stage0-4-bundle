"""Stage0 的最小可运行 CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path


KEY_PATHS = [
    "AGENTS.md",
    "SPEC.md",
    "pyproject.toml",
    "Makefile",
    "src/llm4graphgen/smoke.py",
    "tests/test_smoke.py",
    "runs/stage0/验收报告.md",
]


def run_smoke() -> str:
    """返回最小可运行结果。"""
    return "ok"


def _render_tree(root: Path, max_depth: int = 3) -> list[str]:
    lines: list[str] = []
    ignore_names = {".git", "__pycache__", ".pytest_cache"}

    def walk(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        visible_children = [
            child
            for child in children
            if child.name not in ignore_names and not child.name.startswith("pytest-cache-files-")
        ]
        for idx, child in enumerate(visible_children):
            connector = "`-- " if idx == len(visible_children) - 1 else "|-- "
            rel = child.relative_to(root).as_posix()
            lines.append(f"{prefix}{connector}{rel}")
            if child.is_dir():
                extension = "    " if idx == len(visible_children) - 1 else "|   "
                walk(child, prefix + extension, depth + 1)

    lines.append(".")
    walk(root, "", 1)
    return lines


def build_summary(root: Path) -> str:
    tree_lines = _render_tree(root=root, max_depth=3)
    path_lines = []
    for rel in KEY_PATHS:
        exists = (root / rel).exists()
        mark = "存在" if exists else "缺失"
        path_lines.append(f"- {rel}: {mark}")
    content = [
        "Stage0 结构摘要",
        "",
        "目录树（深度<=3）：",
        *tree_lines,
        "",
        "关键文件检查：",
        *path_lines,
    ]
    return "\n".join(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM4GraphGen Stage0 最小自检命令")
    parser.add_argument("--summary", action="store_true", help="输出 Stage0 结构摘要信息")
    args = parser.parse_args(argv)

    if args.summary:
        print(build_summary(Path.cwd()))
    else:
        print(run_smoke())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
