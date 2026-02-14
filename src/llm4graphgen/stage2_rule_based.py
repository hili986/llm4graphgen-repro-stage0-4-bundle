"""Stage2：Rule-based 8 任务评测与导出。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

from llm4graphgen.parsers import GraphParseResult, parse_graph_output


@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    task_name: str
    prompt: str
    samples: tuple[str, ...]
    train_seen_signatures: frozenset[str]
    validator_name: str
    validator_args: dict[str, int]


def canonical_signature(n: int, edges: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    ordered = sorted((u, v) if u <= v else (v, u) for u, v in edges)
    edge_text = ",".join(f"{u}-{v}" for u, v in ordered)
    return f"{n}|{edge_text}"


def build_graph(parse_result: GraphParseResult) -> Graph:
    assert parse_result.success and parse_result.n is not None
    return Graph(n=parse_result.n, edges=tuple(parse_result.edges))


def adjacency(graph: Graph) -> list[set[int]]:
    adj = [set() for _ in range(graph.n)]
    for u, v in graph.edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def components_count(graph: Graph) -> int:
    adj = adjacency(graph)
    visited = [False] * graph.n
    count = 0
    for start in range(graph.n):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
    return count


def is_connected(graph: Graph) -> bool:
    return components_count(graph) == 1 if graph.n > 0 else True


def is_tree(graph: Graph) -> bool:
    if graph.n == 0:
        return False
    return is_connected(graph) and len(graph.edges) == graph.n - 1


def is_cycle(graph: Graph) -> bool:
    if graph.n < 3:
        return False
    if not is_connected(graph):
        return False
    if len(graph.edges) != graph.n:
        return False
    degrees = [len(x) for x in adjacency(graph)]
    return all(d == 2 for d in degrees)


def is_bipartite(graph: Graph) -> bool:
    adj = adjacency(graph)
    color = [-1] * graph.n
    for i in range(graph.n):
        if color[i] != -1:
            continue
        queue = [i]
        color[i] = 0
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def contains_k5(graph: Graph) -> bool:
    edge_set = set(graph.edges)
    for nodes in combinations(range(graph.n), 5):
        ok = True
        for u, v in combinations(nodes, 2):
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) not in edge_set:
                ok = False
                break
        if ok:
            return True
    return False


def contains_k33(graph: Graph) -> bool:
    edge_set = set(graph.edges)
    for nodes in combinations(range(graph.n), 6):
        nodes = list(nodes)
        for left in combinations(nodes, 3):
            left_set = set(left)
            right = [x for x in nodes if x not in left_set]
            ok = True
            for u in left:
                for v in right:
                    a, b = (u, v) if u <= v else (v, u)
                    if (a, b) not in edge_set:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for u, v in combinations(left, 2):
                a, b = (u, v) if u <= v else (v, u)
                if (a, b) in edge_set:
                    ok = False
                    break
            if not ok:
                continue
            for u, v in combinations(right, 2):
                a, b = (u, v) if u <= v else (v, u)
                if (a, b) in edge_set:
                    ok = False
                    break
            if ok:
                return True
    return False


def is_planar_approx(graph: Graph) -> bool:
    n = graph.n
    m = len(graph.edges)
    if n <= 4:
        return True
    if m > 3 * n - 6:
        return False
    if is_bipartite(graph) and m > 2 * n - 4:
        return False
    if contains_k5(graph):
        return False
    if contains_k33(graph):
        return False
    return True


def is_k_regular(graph: Graph, k: int) -> bool:
    return all(len(nei) == k for nei in adjacency(graph))


def is_wheel(graph: Graph) -> bool:
    if graph.n < 4:
        return False
    adj = adjacency(graph)
    centers = [i for i in range(graph.n) if len(adj[i]) == graph.n - 1]
    if len(centers) != 1:
        return False
    center = centers[0]
    rim = [i for i in range(graph.n) if i != center]
    for node in rim:
        if len(adj[node]) != 3:
            return False
    rim_edges = []
    for u, v in graph.edges:
        if u != center and v != center:
            rim_edges.append((u, v))
    rim_graph = Graph(n=graph.n - 1, edges=tuple(_reindex_edges(rim, rim_edges)))
    return is_cycle(rim_graph)


def _reindex_edges(nodes: list[int], edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mapping = {old: new for new, old in enumerate(nodes)}
    reindexed: list[tuple[int, int]] = []
    for u, v in edges:
        a = mapping[u]
        b = mapping[v]
        reindexed.append((a, b) if a <= b else (b, a))
    return reindexed


def is_k_colorable(graph: Graph, k: int) -> bool:
    adj = adjacency(graph)
    order = sorted(range(graph.n), key=lambda x: len(adj[x]), reverse=True)
    color = [-1] * graph.n

    def backtrack(pos: int) -> bool:
        if pos == graph.n:
            return True
        u = order[pos]
        used = {color[v] for v in adj[u] if color[v] != -1}
        for c in range(k):
            if c in used:
                continue
            color[u] = c
            if backtrack(pos + 1):
                return True
            color[u] = -1
        return False

    return backtrack(0)


def validate_graph(graph: Graph, task: TaskConfig) -> bool:
    vid = task.validator_name
    args = task.validator_args
    if vid == "tree":
        return is_tree(graph)
    if vid == "cycle":
        return is_cycle(graph)
    if vid == "planar":
        return is_planar_approx(graph)
    if vid == "components":
        return components_count(graph) == int(args["components"])
    if vid == "k_regular":
        return is_k_regular(graph, int(args["k"]))
    if vid == "wheel":
        return is_wheel(graph)
    if vid == "bipartite":
        return is_bipartite(graph)
    if vid == "k_coloring":
        return is_k_colorable(graph, int(args["k"]))
    raise ValueError(f"未知任务判定器：{vid}")


def _task_configs() -> list[TaskConfig]:
    return [
        TaskConfig(
            task_id="tree",
            task_name="Tree",
            prompt="生成一张树图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(5, [(0,1),(1,2),(2,3),(3,4)])",
                "(5, [(0,1),(0,2),(0,3),(0,4)])",
                "(5, [(0,1),(1,2),(2,3),(3,4)])",
                "(5, [(0,1),(1,2),(2,0),(3,4)])",
                "bad-output",
                "(5, [(0,1),(1,2),(2,3),(0,4)])",
            ),
            train_seen_signatures=frozenset(
                {
                    canonical_signature(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
                    canonical_signature(5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
                }
            ),
            validator_name="tree",
            validator_args={},
        ),
        TaskConfig(
            task_id="cycle",
            task_name="Cycle",
            prompt="生成一张单环图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,1)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5)])",
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
                "oops",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,2)])",
            ),
            train_seen_signatures=frozenset(
                {canonical_signature(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])}
            ),
            validator_name="cycle",
            validator_args={},
        ),
        TaskConfig(
            task_id="planar",
            task_name="Planar",
            prompt="生成一张平面图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
                "(5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])",
                "(6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])",
                "(4, [(0,1),(1,2),(2,3)])",
                "planar???",
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
            ),
            train_seen_signatures=frozenset({canonical_signature(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])}),
            validator_name="planar",
            validator_args={},
        ),
        TaskConfig(
            task_id="components",
            task_name="#Components",
            prompt="生成恰好 2 个连通分量的图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(6, [(0,1),(1,2),(3,4),(4,5)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5)])",
                "(6, [(0,1),(2,3)])",
                "(6, [(0,1),(1,2),(3,4),(4,5)])",
                "bad components",
                "(6, [(0,1),(1,2),(2,3),(4,5)])",
            ),
            train_seen_signatures=frozenset({canonical_signature(6, [(0, 1), (1, 2), (3, 4), (4, 5)])}),
            validator_name="components",
            validator_args={"components": 2},
        ),
        TaskConfig(
            task_id="k_regular",
            task_name="k-regular",
            prompt="生成一张 2-regular 图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])",
                "(6, [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5)])",
                "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])",
                "reg",
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
            ),
            train_seen_signatures=frozenset({canonical_signature(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])}),
            validator_name="k_regular",
            validator_args={"k": 2},
        ),
        TaskConfig(
            task_id="wheel",
            task_name="Wheel",
            prompt="生成一张轮图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(6, [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5),(5,1)])",
                "(6, [(0,2),(0,3),(0,4),(0,5),(0,1),(1,2),(2,3),(3,4),(4,5),(5,1)])",
                "(6, [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5)])",
                "(6, [(0,1),(0,2),(0,3),(0,4),(0,5)])",
                "wheel",
                "(5, [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)])",
            ),
            train_seen_signatures=frozenset(
                {
                    canonical_signature(
                        6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
                    )
                }
            ),
            validator_name="wheel",
            validator_args={},
        ),
        TaskConfig(
            task_id="bipartite",
            task_name="Bipartite",
            prompt="生成一张二分图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])",
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
                "(6, [(0,1),(1,2),(2,3)])",
                "(6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])",
                "bipartite",
                "(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])",
            ),
            train_seen_signatures=frozenset({canonical_signature(6, [(0, 1), (1, 2), (2, 3)])}),
            validator_name="bipartite",
            validator_args={},
        ),
        TaskConfig(
            task_id="k_coloring",
            task_name="k-coloring",
            prompt="生成一张 3-可着色图，输出格式为 (n, [(u,v), ...])。",
            samples=(
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
                "(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])",
                "(6, [(0,1),(1,2),(2,3)])",
                "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
                "kcolor",
                "(3, [(0,1),(1,2),(2,0)])",
            ),
            train_seen_signatures=frozenset({canonical_signature(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])}),
            validator_name="k_coloring",
            validator_args={"k": 3},
        ),
    ]


def run_stage2(output_root: Path, run_id: str, model: str = "mock-rule-stage2", temperature: float = 0.0) -> tuple[int, Path]:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = _task_configs()
    llm_records: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for task in tasks:
        valid_signatures: list[str] = []
        parse_success_count = 0
        valid_count = 0
        parse_fail_count = 0

        for idx, raw_output in enumerate(task.samples, start=1):
            parse_result = parse_graph_output(raw_output)
            is_valid = False
            signature = ""
            if parse_result.success:
                parse_success_count += 1
                graph = build_graph(parse_result)
                is_valid = validate_graph(graph, task)
                if is_valid:
                    valid_count += 1
                    signature = canonical_signature(graph.n, graph.edges)
                    valid_signatures.append(signature)
            else:
                parse_fail_count += 1

            parsed_result = parse_result.to_dict() if parse_result.success else None
            failure_reason = parse_result.parse_failure_reason

            sample_rows.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "sample_id": idx,
                    "raw_output": raw_output,
                    "parse_success": parse_result.success,
                    "is_valid": is_valid,
                    "signature": signature,
                    "parse_failure_reason": failure_reason or "",
                }
            )

            llm_records.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_id": task.task_id,
                    "sample_id": idx,
                    "provider": "mock",
                    "prompt": task.prompt,
                    "model": model,
                    "temperature": temperature,
                    "raw_output": raw_output,
                    "parsed_result": parsed_result,
                    "parse_failure_reason": failure_reason,
                    "is_valid": is_valid,
                }
            )

        unique_valid = set(valid_signatures)
        unique_valid_count = len(unique_valid)
        novel_unique_count = sum(1 for sig in unique_valid if sig not in task.train_seen_signatures)

        total = len(task.samples)
        valid_rate = valid_count / total if total else 0.0
        unique_rate = unique_valid_count / valid_count if valid_count else 0.0
        novel_rate = novel_unique_count / unique_valid_count if unique_valid_count else 0.0

        metric_rows.append(
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "total_samples": total,
                "parse_success_count": parse_success_count,
                "parse_fail_count": parse_fail_count,
                "valid_count": valid_count,
                "unique_valid_count": unique_valid_count,
                "novel_unique_count": novel_unique_count,
                "valid_rate": f"{valid_rate:.4f}",
                "unique_rate": f"{unique_rate:.4f}",
                "novel_rate": f"{novel_rate:.4f}",
            }
        )

    _write_jsonl(run_dir / "llm_io.jsonl", llm_records)
    _write_csv(run_dir / "rule_based_samples.csv", sample_rows)
    _write_csv(run_dir / "rule_based_metrics.csv", metric_rows)
    _write_csv(
        run_dir / "failure_cases.csv",
        [row for row in sample_rows if (not bool(row["parse_success"])) or (not bool(row["is_valid"]))],
    )

    _write_run_log(run_dir / "run.log", metric_rows)
    return 0, run_dir


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_log(path: Path, metrics: list[dict[str, object]]) -> None:
    lines = [
        "Stage2 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 阶段范围：Rule-based 8 任务（valid/unique/novel）",
        "- 指标口径：",
        "  - valid_rate = valid_count / total_samples",
        "  - unique_rate = unique_valid_count / valid_count",
        "  - novel_rate = novel_unique_count / unique_valid_count",
        "  - novel 判定：与预设 train_seen_signatures 不同的唯一有效图视为 novel",
        "",
        "任务摘要：",
    ]
    for row in metrics:
        lines.append(
            f"- {row['task_id']}: valid={row['valid_rate']}, unique={row['unique_rate']}, novel={row['novel_rate']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage2 Rule-based 8 任务评测")
    parser.add_argument("--output-root", default="runs", help="输出目录根路径")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model", default="mock-rule-stage2")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    _, run_dir = run_stage2(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
    )
    print(f"Stage2 运行完成：{run_dir.as_posix()}")
    print("结果文件：rule_based_metrics.csv / rule_based_samples.csv / llm_io.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
