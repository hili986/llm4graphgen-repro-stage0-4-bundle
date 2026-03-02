"""Stage2：Rule-based 8 任务评测 — 对齐论文实验设计。

改进点（v3, vs v2）：
- [P2a] 图同构去重（WL hash）替代 canonical signature，对齐论文 Unique Rate
- [P2b] 支持 Small/Medium/Large 三种图规模消融（对齐论文 Table 4/8）
- [P3a] 支持多模型便捷切换
- 图规模对齐论文 Table 7（15/16 节点）
- 精确平面性检测（NetworkX Boyer-Myrvold）
- 支持 4 种 prompting 策略
- 支持真实 LLM 调用（100 samples/task）
- 支持多次重复实验 + 均值±标准差
- 保留 Mock 模式用于快速测试
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import networkx as nx

from llm4graphgen.parsers import GraphParseResult, parse_graph_output
from llm4graphgen.prompts import build_rule_prompt


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    task_name: str
    n: int
    validator_name: str
    validator_args: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 图工具函数
# ---------------------------------------------------------------------------

def canonical_signature(n: int, edges: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    """边排序签名 — 快速但不考虑图同构。"""
    ordered = sorted((u, v) if u <= v else (v, u) for u, v in edges)
    edge_text = ",".join(f"{u}-{v}" for u, v in ordered)
    return f"{n}|{edge_text}"


def isomorphism_hash(n: int, edges: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    """[P2a] 基于图同构的哈希 — 使用 Weisfeiler-Lehman 图哈希。

    同构的图会得到相同的哈希值，能正确处理节点重标号。
    论文中 Unique Rate 应当基于图同构判定去重。
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return nx.weisfeiler_lehman_graph_hash(G)


def build_graph(parse_result: GraphParseResult) -> Graph:
    assert parse_result.success and parse_result.n is not None
    return Graph(n=parse_result.n, edges=tuple(parse_result.edges))


def adjacency(graph: Graph) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(graph.n)]
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


def is_planar(graph: Graph) -> bool:
    """精确平面性检测 — 使用 NetworkX Boyer-Myrvold 算法，O(n) 时间。"""
    G = nx.Graph()
    G.add_nodes_from(range(graph.n))
    G.add_edges_from(graph.edges)
    is_planar_result, _ = nx.check_planarity(G)
    return is_planar_result


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
    rim_edges: list[tuple[int, int]] = []
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
    """k-着色检测 — 使用贪心着色作为上界快速检查。"""
    G = nx.Graph()
    G.add_nodes_from(range(graph.n))
    G.add_edges_from(graph.edges)
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    num_colors = max(coloring.values()) + 1 if coloring else 0
    if num_colors <= k:
        return True
    adj = adjacency(graph)
    order = sorted(range(graph.n), key=lambda x: len(adj[x]), reverse=True)
    color_arr = [-1] * graph.n

    def backtrack(pos: int) -> bool:
        if pos == graph.n:
            return True
        u = order[pos]
        used = {color_arr[v] for v in adj[u] if color_arr[v] != -1}
        for c in range(k):
            if c in used:
                continue
            color_arr[u] = c
            if backtrack(pos + 1):
                return True
            color_arr[u] = -1
        return False

    return backtrack(0)


# ---------------------------------------------------------------------------
# 验证器分派
# ---------------------------------------------------------------------------

def validate_graph(graph: Graph, task: TaskConfig) -> bool:
    vid = task.validator_name
    args = task.validator_args
    if vid == "tree":
        return is_tree(graph)
    if vid == "cycle":
        return is_cycle(graph)
    if vid == "planar":
        expected_edges = args.get("m", 0)
        ok = is_planar(graph)
        if expected_edges > 0 and len(graph.edges) != expected_edges:
            ok = False
        return ok
    if vid == "components":
        return components_count(graph) == int(args["k"])
    if vid == "k_regular":
        return is_k_regular(graph, int(args["k"]))
    if vid == "wheel":
        return is_wheel(graph)
    if vid == "bipartite":
        return is_bipartite(graph)
    if vid == "k_coloring":
        expected_edges = args.get("m", 0)
        ok = is_k_colorable(graph, int(args["k"]))
        if expected_edges > 0 and len(graph.edges) != expected_edges:
            ok = False
        return ok
    raise ValueError(f"未知任务判定器：{vid}")


# ---------------------------------------------------------------------------
# 任务配置 — 对齐论文 Table 7/8（支持 Small/Medium/Large 三种规模）
# ---------------------------------------------------------------------------

# [P2b] 论文 Table 7 定义的三种规模
SIZE_PRESETS: dict[str, dict[str, dict[str, int]]] = {
    "small": {
        "tree":       {"n": 8},
        "cycle":      {"n": 10},                    # Table 8: 10
        "planar":     {"n": 8,  "m": 12},
        "components": {"n": 8,  "k": 3},
        "k_regular":  {"n": 12, "k": 3},            # Table 8: 12
        "wheel":      {"n": 8},
        "bipartite":  {"n": 6,  "k": 3},
        "k_coloring": {"n": 10, "m": 20, "k": 3},   # Table 8: n=10, m=20
    },
    "medium": {
        "tree":       {"n": 15},
        "cycle":      {"n": 15},
        "planar":     {"n": 15, "m": 24},
        "components": {"n": 15, "k": 5},
        "k_regular":  {"n": 16, "k": 3},
        "wheel":      {"n": 15},
        "bipartite":  {"n": 10, "k": 5},
        "k_coloring": {"n": 15, "m": 32, "k": 3},
    },
    "large": {
        "tree":       {"n": 30},
        "cycle":      {"n": 20},                     # Table 8: 20
        "planar":     {"n": 30, "m": 50},
        "components": {"n": 30, "k": 10},
        "k_regular":  {"n": 20, "k": 3},             # Table 8: 20
        "wheel":      {"n": 30},
        "bipartite":  {"n": 20, "k": 10},
        "k_coloring": {"n": 18, "m": 39, "k": 3},    # Table 8: n=18, m=39
    },
}


def _task_configs(size: str = "medium") -> list[TaskConfig]:
    """生成任务配置，支持 small/medium/large 三种规模。"""
    preset = SIZE_PRESETS[size]

    configs = []
    for task_id, params in [
        ("tree", "Tree"),
        ("cycle", "Cycle"),
        ("planar", "Planar"),
        ("components", "#Components"),
        ("k_regular", "k-regular"),
        ("wheel", "Wheel"),
        ("bipartite", "Bipartite"),
        ("k_coloring", "k-coloring"),
    ]:
        p = preset[task_id]
        validator_args = {k: v for k, v in p.items() if k != "n"}
        configs.append(TaskConfig(
            task_id=task_id,
            task_name=params,
            n=p["n"],
            validator_name=task_id,
            validator_args=validator_args,
        ))
    return configs


# ---------------------------------------------------------------------------
# Mock 样本生成（用于无 API 快速测试）— 扩充到论文级别
# ---------------------------------------------------------------------------

def _generate_mock_samples(task: TaskConfig, num_samples: int = 6) -> list[str]:
    """生成与论文参数对齐的 Mock 样本（用于离线测试）。"""
    import random as _rand
    from llm4graphgen.graph_samplers import random_tree, random_cycle, random_wheel, format_graph

    rng = _rand.Random(42 + hash(task.task_id))
    samples: list[str] = []
    n = task.n

    for i in range(num_samples):
        if i == num_samples - 1:
            samples.append("bad-output")
            continue

        try:
            if task.task_id == "tree":
                gn, ge = random_tree(n, rng)
                samples.append(format_graph(gn, ge))
            elif task.task_id == "cycle":
                gn, ge = random_cycle(n, rng)
                samples.append(format_graph(gn, ge))
            elif task.task_id == "wheel":
                gn, ge = random_wheel(n, rng)
                samples.append(format_graph(gn, ge))
            elif task.task_id == "bipartite":
                edges = []
                for u in range(n // 2):
                    for v in range(n // 2, n):
                        if rng.random() < 0.4:
                            edges.append((u, v))
                if not edges:
                    edges.append((0, n // 2))
                samples.append(format_graph(n, sorted(edges)))
            elif task.task_id == "k_regular":
                k = task.validator_args["k"]
                gn, ge = random_cycle(n, rng)
                samples.append(format_graph(gn, ge))
            elif task.task_id == "components":
                k_comp = task.validator_args["k"]
                chunk = n // k_comp
                edges = []
                offset = 0
                for c in range(k_comp):
                    sz = chunk if c < k_comp - 1 else n - offset
                    for j in range(sz - 1):
                        edges.append((offset + j, offset + j + 1))
                    offset += sz
                samples.append(format_graph(n, sorted(edges)))
            elif task.task_id == "planar":
                edges = []
                for j in range(n - 1):
                    edges.append((j, j + 1))
                extra = 0
                while len(edges) < task.validator_args.get("m", 24) and extra < 100:
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u != v:
                        e = (min(u, v), max(u, v))
                        if e not in edges:
                            edges.append(e)
                    extra += 1
                samples.append(format_graph(n, sorted(edges)))
            elif task.task_id == "k_coloring":
                edges = []
                for j in range(n - 1):
                    edges.append((j, j + 1))
                while len(edges) < task.validator_args.get("m", 32):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u != v:
                        e = (min(u, v), max(u, v))
                        if e not in edges:
                            edges.append(e)
                samples.append(format_graph(n, sorted(edges)))
            else:
                samples.append(format_graph(n, [(i, (i + 1) % n) for i in range(n)]))
        except Exception:
            samples.append(format_graph(n, [(0, 1)]))

    return samples


# ---------------------------------------------------------------------------
# LLM 输出解析：从可能包含推理过程的文本中提取图
# ---------------------------------------------------------------------------

def extract_graph_from_response(response: str) -> str:
    """从 LLM 响应中提取最后一个 (n, [...]) 格式的图。

    支持 CoT 模式（响应中包含推理文本 + 最终图）。
    """
    pattern = r'\(\s*\d+\s*,\s*\[.*?\]\s*\)'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1]
    return response.strip()


# ---------------------------------------------------------------------------
# 核心运行引擎
# ---------------------------------------------------------------------------

def run_stage2(
    output_root: Path,
    run_id: str,
    model: str = "mock-rule-stage2",
    temperature: float = 0.8,
    provider=None,
    strategy: str = "zero_shot",
    num_samples: int = 100,
    num_repeats: int = 1,
    unique_method: str = "isomorphism",
    size: str = "medium",
) -> tuple[int, Path]:
    """运行 Stage2 评测。

    Args:
        provider: BaseProvider 实例。为 None 时使用 Mock 数据。
        strategy: "zero_shot" | "few_shot" | "zero_shot_cot" | "few_shot_cot"
        num_samples: 每任务样本数（论文为 100）
        num_repeats: 重复实验次数
        unique_method: "signature" (v2 边排序) 或 "isomorphism" (v3 WL hash，默认)
        size: "small" | "medium" | "large" (对齐论文 Table 4/8)
    """
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = _task_configs(size=size)
    use_mock = provider is None

    # 选择去重函数
    def compute_unique_key(n: int, edges) -> str:
        if unique_method == "isomorphism":
            return isomorphism_hash(n, edges)
        return canonical_signature(n, edges)

    all_repeat_metrics: list[list[dict[str, object]]] = []

    for repeat_idx in range(num_repeats):
        llm_records: list[dict[str, object]] = []
        sample_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []

        for task in tasks:
            prompt = build_rule_prompt(task.task_id, strategy, n=task.n)

            if use_mock:
                raw_outputs = _generate_mock_samples(task, num_samples=min(num_samples, 20))
            else:
                raw_outputs = []
                for _ in range(num_samples):
                    try:
                        resp = provider.generate(prompt, model, temperature)
                        raw_outputs.append(extract_graph_from_response(resp))
                    except Exception as exc:
                        raw_outputs.append(f"LLM_ERROR: {exc}")

            valid_keys: list[str] = []
            parse_success_count = 0
            valid_count = 0
            parse_fail_count = 0

            # 收集 few-shot 示例签名作为 train_seen
            from llm4graphgen.prompts import RULE_TASK_DESCRIPTIONS
            train_seen: set[str] = set()
            for ex in RULE_TASK_DESCRIPTIONS[task.task_id]["few_shot_examples"]:
                pr = parse_graph_output(ex)
                if pr.success and pr.n is not None:
                    train_seen.add(compute_unique_key(pr.n, pr.edges))

            for idx, raw_output in enumerate(raw_outputs, start=1):
                parse_result = parse_graph_output(raw_output)
                is_valid = False
                unique_key = ""
                if parse_result.success:
                    parse_success_count += 1
                    graph = build_graph(parse_result)
                    is_valid = validate_graph(graph, task)
                    if is_valid:
                        valid_count += 1
                        unique_key = compute_unique_key(graph.n, graph.edges)
                        valid_keys.append(unique_key)
                else:
                    parse_fail_count += 1

                parsed_result = parse_result.to_dict() if parse_result.success else None
                failure_reason = parse_result.parse_failure_reason

                sample_rows.append({
                    "repeat": repeat_idx + 1,
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "sample_id": idx,
                    "raw_output": raw_output[:200],
                    "parse_success": parse_result.success,
                    "is_valid": is_valid,
                    "unique_key": unique_key,
                    "unique_method": unique_method,
                    "parse_failure_reason": failure_reason or "",
                })

                llm_records.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "repeat": repeat_idx + 1,
                    "task_id": task.task_id,
                    "sample_id": idx,
                    "provider": "mock" if use_mock else provider.name,
                    "strategy": strategy,
                    "prompt": prompt[:300],
                    "model": model,
                    "temperature": temperature,
                    "raw_output": raw_output[:500],
                    "parsed_result": parsed_result,
                    "parse_failure_reason": failure_reason,
                    "is_valid": is_valid,
                })

            unique_valid = set(valid_keys)
            unique_valid_count = len(unique_valid)
            novel_unique_count = sum(1 for key in unique_valid if key not in train_seen)

            total = len(raw_outputs)
            valid_rate = valid_count / total if total else 0.0
            unique_rate = unique_valid_count / valid_count if valid_count else 0.0
            novel_rate = novel_unique_count / unique_valid_count if unique_valid_count else 0.0

            metric_rows.append({
                "repeat": repeat_idx + 1,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "strategy": strategy,
                "size": size,
                "n": task.n,
                "unique_method": unique_method,
                "total_samples": total,
                "parse_success_count": parse_success_count,
                "parse_fail_count": parse_fail_count,
                "valid_count": valid_count,
                "unique_valid_count": unique_valid_count,
                "novel_unique_count": novel_unique_count,
                "valid_rate": f"{valid_rate:.4f}",
                "unique_rate": f"{unique_rate:.4f}",
                "novel_rate": f"{novel_rate:.4f}",
            })

        all_repeat_metrics.append(metric_rows)

        suffix = f"_r{repeat_idx + 1}" if num_repeats > 1 else ""
        _write_jsonl(run_dir / f"llm_io{suffix}.jsonl", llm_records)
        _write_csv(run_dir / f"rule_based_samples{suffix}.csv", sample_rows)
        _write_csv(run_dir / f"rule_based_metrics{suffix}.csv", metric_rows)
        _write_csv(
            run_dir / f"failure_cases{suffix}.csv",
            [r for r in sample_rows if not bool(r["parse_success"]) or not bool(r["is_valid"])],
        )

    summary_rows = _compute_summary(all_repeat_metrics)
    _write_csv(run_dir / "rule_based_summary.csv", summary_rows)
    _write_run_log(
        run_dir / "run.log", summary_rows, strategy, model, temperature,
        num_samples, num_repeats, unique_method, size,
    )
    return 0, run_dir


def _compute_summary(all_repeats: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """多次重复实验的均值±标准差汇总。"""
    import numpy as _np

    if len(all_repeats) == 1:
        return all_repeats[0]

    task_ids: list[str] = []
    for row in all_repeats[0]:
        task_ids.append(str(row["task_id"]))

    summary: list[dict[str, object]] = []
    for i, tid in enumerate(task_ids):
        vals = {"valid_rate": [], "unique_rate": [], "novel_rate": []}
        for repeat_metrics in all_repeats:
            row = repeat_metrics[i]
            for key in vals:
                vals[key].append(float(str(row[key])))

        row0 = all_repeats[0][i]
        s: dict[str, object] = {
            "task_id": tid,
            "task_name": row0["task_name"],
            "strategy": row0["strategy"],
            "size": row0.get("size", "medium"),
            "n": row0["n"],
            "unique_method": row0.get("unique_method", "signature"),
            "num_repeats": len(all_repeats),
        }
        for key in vals:
            arr = _np.array(vals[key])
            s[f"{key}_mean"] = f"{arr.mean():.1f}" if key == "valid_rate" else f"{arr.mean():.4f}"
            s[f"{key}_std"] = f"{arr.std():.1f}" if key == "valid_rate" else f"{arr.std():.4f}"
            s[f"{key}_display"] = f"{arr.mean() * 100:.1f}±{arr.std() * 100:.1f}"
        summary.append(s)

    return summary


# ---------------------------------------------------------------------------
# 输出写入工具
# ---------------------------------------------------------------------------

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


def _write_run_log(
    path: Path,
    metrics: list[dict[str, object]],
    strategy: str,
    model: str,
    temperature: float,
    num_samples: int,
    num_repeats: int,
    unique_method: str,
    size: str,
) -> None:
    lines = [
        "Stage2 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：{model}",
        f"- 策略：{strategy}",
        f"- 温度：{temperature}",
        f"- 每任务样本数：{num_samples}",
        f"- 重复次数：{num_repeats}",
        f"- 图规模：{size}",
        f"- 去重方式：{unique_method}",
        "- 阶段范围：Rule-based 8 任务（valid/unique/novel）",
        "- 平面性检测：NetworkX Boyer-Myrvold 精确算法",
        "- 指标口径：",
        "  - valid_rate = valid_count / total_samples",
        "  - unique_rate = unique_valid_count / valid_count",
        "  - novel_rate = novel_unique_count / unique_valid_count",
        "",
        "任务摘要：",
    ]
    for row in metrics:
        if "valid_rate_display" in row:
            lines.append(
                f"- {row['task_id']}: valid={row['valid_rate_display']}, "
                f"unique={row['unique_rate_display']}, novel={row['novel_rate_display']}"
            )
        else:
            lines.append(
                f"- {row['task_id']}: valid={row['valid_rate']}, "
                f"unique={row['unique_rate']}, novel={row['novel_rate']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage2 Rule-based 8 任务评测（论文对齐版 v3）")
    parser.add_argument("--output-root", default="runs", help="输出目录根路径")
    parser.add_argument("--run-id", default=None, help="运行 ID，默认自动生成描述性名称")
    parser.add_argument("--model", default="mock-rule-stage2")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"])
    parser.add_argument("--num-samples", type=int, default=100, help="每任务样本数（论文为100）")
    parser.add_argument("--num-repeats", type=int, default=1, help="重复实验次数")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai"])
    parser.add_argument("--unique-method", default="isomorphism",
                        choices=["signature", "isomorphism"],
                        help="去重方式: signature (v2 边排序) 或 isomorphism (v3 WL hash)")
    parser.add_argument("--size", default="medium",
                        choices=["small", "medium", "large"],
                        help="图规模：small/medium/large (对齐论文 Table 4/8)")
    args = parser.parse_args(argv)

    # 自动生成描述性 run_id
    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]  # 去掉可能的路径前缀
        parts = [
            "stage2", model_short, args.strategy, args.size,
            f"{args.num_samples}s",
        ]
        if args.num_repeats > 1:
            parts.append(f"r{args.num_repeats}")
        parts.append(ts)
        args.run_id = "_".join(parts)

    prov = None
    if args.provider == "openai":
        from llm4graphgen.providers import OpenAIProvider
        prov = OpenAIProvider()

    _, run_dir = run_stage2(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
        provider=prov,
        strategy=args.strategy,
        num_samples=args.num_samples,
        num_repeats=args.num_repeats,
        unique_method=args.unique_method,
        size=args.size,
    )
    print(f"Stage2 运行完成：{run_dir.as_posix()}")
    print("结果文件：rule_based_summary.csv / rule_based_metrics.csv / llm_io.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
