"""Stage3：Distribution-based 评测 — 对齐论文实验设计。

改进点（vs 原版）：
- 实现 p 值扫描实验（p = 0.2, 0.4, 0.6, 0.8）
- 图分布采样器生成输入图
- 支持 3 种 motif（triangle, house, crane）
- 修正 p_pred/p_gen 为论文定义
- 支持真实 LLM + CoT 策略
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm4graphgen.parsers import GraphParseResult, parse_graph_output
from llm4graphgen.prompts import build_distribution_prompt
from llm4graphgen.stage2_rule_based import (
    Graph, adjacency, components_count, is_cycle, is_tree,
)
from llm4graphgen.graph_samplers import (
    sample_trees_or_cycles, sample_union_of_components, sample_motif_graphs,
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributionTaskConfig:
    task_id: str
    task_name: str
    judge_name: str
    judge_args: dict[str, str]


# ---------------------------------------------------------------------------
# Judge 函数
# ---------------------------------------------------------------------------

def _canonical_edges(graph: Graph) -> set[tuple[int, int]]:
    return set((u, v) if u <= v else (v, u) for u, v in graph.edges)


def _component_nodes(graph: Graph) -> list[list[int]]:
    adj = adjacency(graph)
    visited = [False] * graph.n
    groups: list[list[int]] = []
    for i in range(graph.n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        group: list[int] = []
        while stack:
            u = stack.pop()
            group.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(sorted(group))
    return groups


def _induced_subgraph(graph: Graph, nodes: list[int]) -> Graph:
    mapping = {old: new for new, old in enumerate(nodes)}
    node_set = set(nodes)
    edges: list[tuple[int, int]] = []
    for u, v in graph.edges:
        if u in node_set and v in node_set:
            a, b = mapping[u], mapping[v]
            edges.append((a, b) if a <= b else (b, a))
    edges = sorted(set(edges))
    return Graph(n=len(nodes), edges=tuple(edges))


def judge_trees_or_cycles(graph: Graph) -> bool:
    return is_tree(graph) or is_cycle(graph)


def judge_union_of_components(graph: Graph) -> bool:
    if components_count(graph) < 2:
        return False
    for nodes in _component_nodes(graph):
        sub = _induced_subgraph(graph, nodes)
        if not (is_tree(sub) or is_cycle(sub)):
            return False
    return True


def _has_triangle(graph: Graph) -> bool:
    edges = _canonical_edges(graph)
    for a in range(graph.n):
        for b in range(a + 1, graph.n):
            if (a, b) not in edges:
                continue
            for c in range(b + 1, graph.n):
                if (a, c) in edges and (b, c) in edges:
                    return True
    return False


def _has_house(graph: Graph) -> bool:
    """检查是否包含 house motif（5 节点: 四边形 + 一个三角形顶）。"""
    adj = adjacency(graph)
    for a in range(graph.n):
        for b in adj[a]:
            if b <= a:
                continue
            common = adj[a] & adj[b]
            for c in common:
                for d in adj[c]:
                    if d != a and d != b and d in adj[a] and d not in adj[b]:
                        return True
                    if d != a and d != b and d in adj[b] and d not in adj[a]:
                        return True
    return False


def _has_crane(graph: Graph) -> bool:
    """检查是否包含 crane motif（高度连接的 4-clique 子结构）。"""
    edges = _canonical_edges(graph)
    from itertools import combinations
    for nodes in combinations(range(graph.n), 4):
        count = sum(1 for u, v in combinations(nodes, 2)
                    if (min(u, v), max(u, v)) in edges)
        if count >= 5:  # crane 有 5+ 边
            return True
    return False


def judge_motif(graph: Graph, motif: str) -> bool:
    if graph.n > 50:
        raise RuntimeError("图规模超过 motif 匹配上限（n > 50），本次判别中止。")
    if motif == "triangle":
        return _has_triangle(graph)
    if motif == "house":
        return _has_house(graph)
    if motif == "crane":
        return _has_crane(graph)
    raise ValueError(f"未知 motif：{motif}")


# ---------------------------------------------------------------------------
# 任务配置
# ---------------------------------------------------------------------------

def _task_configs() -> list[DistributionTaskConfig]:
    return [
        DistributionTaskConfig(
            task_id="trees_or_cycles",
            task_name="Trees-or-cycles",
            judge_name="trees_or_cycles",
            judge_args={},
        ),
        DistributionTaskConfig(
            task_id="union_of_components",
            task_name="Union-of-components",
            judge_name="union_of_components",
            judge_args={},
        ),
        DistributionTaskConfig(
            task_id="motif_triangle",
            task_name="Motif-Triangle",
            judge_name="motif",
            judge_args={"motif": "triangle"},
        ),
    ]


def _judge_graph(graph: Graph, task: DistributionTaskConfig) -> bool:
    if task.judge_name == "trees_or_cycles":
        return judge_trees_or_cycles(graph)
    if task.judge_name == "union_of_components":
        return judge_union_of_components(graph)
    if task.judge_name == "motif":
        return judge_motif(graph, motif=task.judge_args["motif"])
    raise ValueError(f"未知判别器：{task.judge_name}")


def _build_graph(parse_result: GraphParseResult) -> Graph:
    assert parse_result.success and parse_result.n is not None
    return Graph(n=parse_result.n, edges=tuple(parse_result.edges))


# ---------------------------------------------------------------------------
# LLM 输出多图解析
# ---------------------------------------------------------------------------

def extract_graphs_from_response(response: str) -> list[str]:
    """从 LLM 响应中提取所有 (n, [...]) 格式的图。"""
    pattern = r'\(\s*\d+\s*,\s*\[.*?\]\s*\)'
    matches = re.findall(pattern, response, re.DOTALL)
    return matches if matches else [response.strip()]


def extract_p_from_response(response: str) -> float | None:
    """从 LLM 响应中提取推断的 p 值。"""
    patterns = [
        r'p\s*=\s*([0-9]*\.?[0-9]+)',
        r'p\s*is\s*(?:approximately\s*)?([0-9]*\.?[0-9]+)',
        r'estimated?\s*p\s*[:=]\s*([0-9]*\.?[0-9]+)',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0 <= val <= 1:
                    return val
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Mock 样本生成
# ---------------------------------------------------------------------------

def _generate_mock_outputs(
    task: DistributionTaskConfig, p: float, n_nodes: int, num_output: int, seed: int,
) -> list[str]:
    """生成 Mock LLM 输出。"""
    if task.task_id == "trees_or_cycles":
        return sample_trees_or_cycles(p, n_nodes, num_output, seed=seed + 1000)
    elif task.task_id == "union_of_components":
        return sample_union_of_components(p, n_nodes // 2, num_output, seed=seed + 2000)
    elif task.task_id.startswith("motif"):
        return sample_motif_graphs(p, n_nodes, num_output, seed=seed + 3000)
    else:
        return sample_trees_or_cycles(0.5, n_nodes, num_output, seed=seed)


# ---------------------------------------------------------------------------
# 核心运行引擎
# ---------------------------------------------------------------------------

def run_stage3(
    output_root: Path,
    run_id: str,
    model: str = "mock-distribution-stage3",
    temperature: float = 0.5,
    provider=None,
    strategy: str = "zero_shot",
    p_values: list[float] | None = None,
    n_nodes: int = 10,
    num_input: int = 10,
    num_output: int = 10,
) -> tuple[int, Path]:
    """运行 Stage3 评测。

    Args:
        p_values: 要扫描的 p 值列表（论文为 [0.2, 0.4, 0.6, 0.8]）
        n_nodes: 每个图的节点数
        num_input: 输入图数量
        num_output: 要求生成的图数量
    """
    if p_values is None:
        p_values = [0.2, 0.4, 0.6, 0.8]

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = _task_configs()
    use_mock = provider is None

    sample_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    llm_rows: list[dict[str, object]] = []

    for task in tasks:
        for p in p_values:
            seed = int(p * 1000) + hash(task.task_id) % 10000

            # 生成输入图
            if task.task_id == "trees_or_cycles":
                input_graphs = sample_trees_or_cycles(p, n_nodes, num_input, seed=seed)
            elif task.task_id == "union_of_components":
                input_graphs = sample_union_of_components(p, n_nodes // 2, num_input, seed=seed)
            else:
                input_graphs = sample_motif_graphs(p, n_nodes, num_input, seed=seed)

            prompt = build_distribution_prompt(task.task_id, input_graphs, num_output, strategy)

            # 获取 LLM 输出
            if use_mock:
                output_graphs = _generate_mock_outputs(task, p, n_nodes, num_output, seed)
                p_pred_val = p + 0.05  # Mock: 推断 p 接近真实值
            else:
                try:
                    resp = provider.generate(prompt, model, temperature)
                    output_graphs = extract_graphs_from_response(resp)
                    p_pred_val = extract_p_from_response(resp)
                except Exception as exc:
                    output_graphs = [f"LLM_ERROR: {exc}"]
                    p_pred_val = None

            # 评估输出图
            total = len(output_graphs)
            parse_success_count = 0
            parse_fail_count = 0
            judge_success_count = 0
            judge_fail_count = 0
            pred_positive_count = 0

            for idx, raw in enumerate(output_graphs, start=1):
                parse_result = parse_graph_output(raw)
                judge_positive: bool | None = None
                judge_failure_reason = ""
                if parse_result.success:
                    parse_success_count += 1
                    graph = _build_graph(parse_result)
                    try:
                        judge_positive = _judge_graph(graph, task)
                        judge_success_count += 1
                        if judge_positive:
                            pred_positive_count += 1
                    except Exception as exc:
                        judge_fail_count += 1
                        judge_failure_reason = str(exc)
                else:
                    parse_fail_count += 1

                sample_rows.append({
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "p_true": p,
                    "sample_id": idx,
                    "raw_output": raw[:200],
                    "parse_success": parse_result.success,
                    "judge_success": judge_positive is not None,
                    "judge_positive": "" if judge_positive is None else bool(judge_positive),
                    "parse_failure_reason": parse_result.parse_failure_reason or "",
                    "judge_failure_reason": judge_failure_reason,
                })

                llm_rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_id": task.task_id,
                    "p_true": p,
                    "sample_id": idx,
                    "provider": "mock" if use_mock else provider.name,
                    "strategy": strategy,
                    "model": model,
                    "temperature": temperature,
                    "raw_output": raw[:500],
                    "parsed_result": parse_result.to_dict() if parse_result.success else None,
                    "parse_failure_reason": parse_result.parse_failure_reason,
                    "judge_result": judge_positive,
                    "judge_failure_reason": judge_failure_reason if judge_positive is None else None,
                })

            # 计算指标 — 对齐论文定义
            p_gen = pred_positive_count / judge_success_count if judge_success_count else 0.0
            parse_fail_rate = parse_fail_count / total if total else 0.0
            judge_fail_rate = judge_fail_count / parse_success_count if parse_success_count else 0.0

            metric_rows.append({
                "task_id": task.task_id,
                "task_name": task.task_name,
                "p_true": p,
                "p_pred": f"{p_pred_val:.4f}" if p_pred_val is not None else "N/A",
                "p_gen": f"{p_gen:.4f}",
                "p_pred_error": f"{abs(p_pred_val - p):.4f}" if p_pred_val is not None else "N/A",
                "p_gen_error": f"{abs(p_gen - p):.4f}",
                "total_samples": total,
                "parse_success_count": parse_success_count,
                "parse_fail_count": parse_fail_count,
                "parse_fail_rate": f"{parse_fail_rate:.4f}",
                "judge_success_count": judge_success_count,
                "judge_fail_count": judge_fail_count,
                "judge_fail_rate": f"{judge_fail_rate:.4f}",
                "pred_positive_count": pred_positive_count,
            })

    _write_csv(run_dir / "distribution_metrics.csv", metric_rows)
    _write_csv(run_dir / "distribution_samples.csv", sample_rows)
    _write_csv(
        run_dir / "failure_cases.csv",
        [row for row in sample_rows if not bool(row["parse_success"]) or not bool(row["judge_success"])],
    )
    _write_jsonl(run_dir / "llm_io.jsonl", llm_rows)
    _write_run_log(run_dir / "run.log", metric_rows, strategy, model, temperature, p_values)
    return 0, run_dir


# ---------------------------------------------------------------------------
# 输出写入工具
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_run_log(
    path: Path,
    metrics: list[dict[str, object]],
    strategy: str,
    model: str,
    temperature: float,
    p_values: list[float],
) -> None:
    lines = [
        "Stage3 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：{model}",
        f"- 策略：{strategy}",
        f"- 温度：{temperature}",
        f"- p 值扫描：{p_values}",
        "- 阶段范围：Distribution-based 任务（p_pred / p_gen / p_error）",
        "- 指标口径（对齐论文）：",
        "  - p_pred = LLM 推断的分布参数",
        "  - p_gen = 从生成图计算的经验正类率",
        "  - p_pred_error = |p_pred - p_true|",
        "  - p_gen_error = |p_gen - p_true|",
        "",
        "任务摘要：",
    ]
    for row in metrics:
        lines.append(
            f"- {row['task_id']}(p={row['p_true']}): p_pred={row['p_pred']}, "
            f"p_gen={row['p_gen']}, parse_fail={row['parse_fail_rate']}, "
            f"judge_fail={row['judge_fail_rate']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage3 Distribution-based 评测（论文对齐版）")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model", default="mock-distribution-stage3")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"])
    parser.add_argument("--p-values", default="0.2,0.4,0.6,0.8",
                        help="逗号分隔的 p 值列表")
    parser.add_argument("--n-nodes", type=int, default=10, help="每图节点数")
    parser.add_argument("--num-input", type=int, default=10, help="输入图数量")
    parser.add_argument("--num-output", type=int, default=10, help="生成图数量")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai"])
    args = parser.parse_args(argv)

    prov = None
    if args.provider == "openai":
        from llm4graphgen.providers import OpenAIProvider
        prov = OpenAIProvider()

    p_vals = [float(x.strip()) for x in args.p_values.split(",")]

    _, run_dir = run_stage3(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
        provider=prov,
        strategy=args.strategy,
        p_values=p_vals,
        n_nodes=args.n_nodes,
        num_input=args.num_input,
        num_output=args.num_output,
    )
    print(f"Stage3 运行完成：{run_dir.as_posix()}")
    print("结果文件：distribution_metrics.csv / distribution_samples.csv / llm_io.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
