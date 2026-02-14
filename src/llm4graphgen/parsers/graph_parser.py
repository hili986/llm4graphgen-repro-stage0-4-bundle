"""图结构输出解析器。"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class GraphParseResult:
    """图解析结果。"""

    success: bool
    n: int | None
    edges: list[tuple[int, int]]
    parse_failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        if not self.success:
            return {
                "success": False,
                "n": None,
                "edges": [],
                "parse_failure_reason": self.parse_failure_reason,
            }
        return {
            "success": True,
            "n": self.n,
            "edges": [list(edge) for edge in self.edges],
            "parse_failure_reason": None,
        }


def parse_graph_output(raw_text: str) -> GraphParseResult:
    """解析 SPEC 规定的 `(n, [(u,v), ...])` 格式。"""
    text = raw_text.strip()
    if not text:
        return GraphParseResult(False, None, [], "模型输出为空，无法解析图结构。")

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        return GraphParseResult(False, None, [], f"格式错误，无法解析为 `(n, [(u,v), ...])`：{exc}")

    if not isinstance(parsed, tuple) or len(parsed) != 2:
        return GraphParseResult(False, None, [], "输出必须是长度为 2 的元组：`(n, edges)`。")

    n, raw_edges = parsed
    if not isinstance(n, int) or n < 0:
        return GraphParseResult(False, None, [], "节点数 n 必须是非负整数。")
    if not isinstance(raw_edges, (list, tuple)):
        return GraphParseResult(False, None, [], "边集合必须是列表或元组。")

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for idx, edge in enumerate(raw_edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return GraphParseResult(False, None, [], f"第 {idx + 1} 条边不是二元组 `(u,v)`。")
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int):
            return GraphParseResult(False, None, [], f"第 {idx + 1} 条边包含非整数端点。")
        if u < 0 or v < 0 or u >= n or v >= n:
            return GraphParseResult(False, None, [], f"第 {idx + 1} 条边端点越界（节点范围应为 0..{n - 1}）。")

        canon = (u, v) if u <= v else (v, u)
        if canon not in seen:
            seen.add(canon)
            edges.append(canon)

    return GraphParseResult(True, n, edges, None)
