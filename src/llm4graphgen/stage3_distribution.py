"""Stage3：Distribution-based 3 子任务评测。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm4graphgen.parsers import GraphParseResult, parse_graph_output
from llm4graphgen.stage2_rule_based import Graph, adjacency, components_count, is_cycle, is_tree


@dataclass(frozen=True)
class DistributionTaskConfig:
    task_id: str
    task_name: str
    prompt: str
    samples: tuple[str, ...]
    judge_name: str
    judge_args: dict[str, str]


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


def judge_motif(graph: Graph, motif: str) -> bool:
    if graph.n > 40:
        raise RuntimeError("图规模超过 motif 匹配上限（n > 40），本次判别中止。")
    edges = _canonical_edges(graph)
    if motif == "triangle":
        for a in range(graph.n):
            for b in range(a + 1, graph.n):
                if (a, b) not in edges:
                    continue
                for c in range(b + 1, graph.n):
                    if (a, c) in edges and (b, c) in edges:
                        return True
        return False
    raise ValueError(f"未知 motif：{motif}")


def _task_configs() -> list[DistributionTaskConfig]:
    return [
        DistributionTaskConfig(
            task_id="trees_or_cycles",
            task_name="Trees-or-cycles",
            prompt="生成一张树或环图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(5, [(0,1),(1,2),(2,3),(3,4)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])",
                "(5, [(0,1),(1,2),(2,0),(2,3)])",
                "bad-output",
                "(4, [(0,1),(1,2),(2,3)])",
                "(6, [(0,1),(1,2),(3,4),(4,5)])",
            ),
            judge_name="trees_or_cycles",
            judge_args={},
        ),
        DistributionTaskConfig(
            task_id="union_of_components",
            task_name="Union-of-components",
            prompt="生成由多个连通分量组成，且每个分量为树或环的图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(6, [(0,1),(1,2),(3,4),(4,5)])",
                "(6, [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5)])",
                "(6, [(0,1),(1,2),(2,0),(3,4)])",
                "component-bad",
                "(7, [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6)])",
            ),
            judge_name="union_of_components",
            judge_args={},
        ),
        DistributionTaskConfig(
            task_id="motif",
            task_name="Motif",
            prompt="生成包含三角形 motif 的图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(5, [(0,1),(1,2),(2,0),(2,3)])",
                "(5, [(0,1),(1,2),(2,3),(3,4)])",
                "(6, [(0,1),(1,2),(2,0),(3,4),(4,5)])",
                "(60, [(0,1)])",
                "motif-bad",
                "(4, [(0,1),(1,2),(2,0),(0,3)])",
            ),
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


def run_stage3(output_root: Path, run_id: str, model: str = "mock-distribution-stage3", temperature: float = 0.0) -> tuple[int, Path]:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = _task_configs()
    sample_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    llm_rows: list[dict[str, object]] = []

    for task in tasks:
        total = len(task.samples)
        parse_success_count = 0
        parse_fail_count = 0
        judge_success_count = 0
        judge_fail_count = 0
        pred_positive_count = 0

        for idx, raw in enumerate(task.samples, start=1):
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
                except Exception as exc:  # noqa: BLE001
                    judge_fail_count += 1
                    judge_positive = None
                    judge_failure_reason = str(exc)
            else:
                parse_fail_count += 1

            sample_rows.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "sample_id": idx,
                    "raw_output": raw,
                    "parse_success": parse_result.success,
                    "judge_success": judge_positive is not None,
                    "judge_positive": "" if judge_positive is None else bool(judge_positive),
                    "parse_failure_reason": parse_result.parse_failure_reason or "",
                    "judge_failure_reason": judge_failure_reason,
                }
            )

            llm_rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_id": task.task_id,
                    "sample_id": idx,
                    "provider": "mock",
                    "prompt": task.prompt,
                    "model": model,
                    "temperature": temperature,
                    "raw_output": raw,
                    "parsed_result": parse_result.to_dict() if parse_result.success else None,
                    "parse_failure_reason": parse_result.parse_failure_reason,
                    "judge_result": judge_positive,
                    "judge_failure_reason": judge_failure_reason if judge_positive is None else None,
                }
            )

        p_pred = (pred_positive_count / judge_success_count) if judge_success_count else 0.0
        p_gen = (pred_positive_count / total) if total else 0.0
        parse_fail_rate = (parse_fail_count / total) if total else 0.0
        judge_fail_rate = (judge_fail_count / parse_success_count) if parse_success_count else 0.0

        metric_rows.append(
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "total_samples": total,
                "parse_success_count": parse_success_count,
                "parse_fail_count": parse_fail_count,
                "parse_fail_rate": f"{parse_fail_rate:.4f}",
                "judge_success_count": judge_success_count,
                "judge_fail_count": judge_fail_count,
                "judge_fail_rate": f"{judge_fail_rate:.4f}",
                "pred_positive_count": pred_positive_count,
                "p_pred": f"{p_pred:.4f}",
                "p_gen": f"{p_gen:.4f}",
            }
        )

    _write_csv(run_dir / "distribution_metrics.csv", metric_rows)
    _write_csv(run_dir / "distribution_samples.csv", sample_rows)
    _write_csv(
        run_dir / "failure_cases.csv",
        [row for row in sample_rows if (not bool(row["parse_success"])) or (not bool(row["judge_success"]))],
    )
    _write_jsonl(run_dir / "llm_io.jsonl", llm_rows)
    _write_run_log(run_dir / "run.log", metric_rows)
    return 0, run_dir


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


def _write_run_log(path: Path, metrics: list[dict[str, object]]) -> None:
    lines = [
        "Stage3 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 阶段范围：Distribution-based 3 子任务（p_pred / p_gen）",
        "- 指标口径：",
        "  - p_pred = pred_positive_count / judge_success_count",
        "  - p_gen = pred_positive_count / total_samples",
        "  - parse_fail_rate = parse_fail_count / total_samples",
        "  - judge_fail_rate = judge_fail_count / parse_success_count",
        "",
        "任务摘要：",
    ]
    for row in metrics:
        lines.append(
            f"- {row['task_id']}: p_pred={row['p_pred']}, p_gen={row['p_gen']}, "
            f"parse_fail_rate={row['parse_fail_rate']}, judge_fail_rate={row['judge_fail_rate']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage3 Distribution-based 3 子任务评测")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model", default="mock-distribution-stage3")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    _, run_dir = run_stage3(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
    )
    print(f"Stage3 运行完成：{run_dir.as_posix()}")
    print("结果文件：distribution_metrics.csv / distribution_samples.csv / llm_io.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
