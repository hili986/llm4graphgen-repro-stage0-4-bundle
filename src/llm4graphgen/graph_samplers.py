"""图分布采样器：为 Distribution-based 任务生成输入图。

对齐论文的分布实验设计：
- Trees-or-cycles：以概率 p 生成 tree，1-p 生成 cycle
- Union-of-components：2 个连通分量，以概率 p 为同类型
- Motif：base(tree/ladder/wheel) + motif(triangle/house/crane)
"""

from __future__ import annotations

import random
from itertools import combinations


def random_tree(n: int, rng: random.Random | None = None) -> tuple[int, list[tuple[int, int]]]:
    """生成随机树（Prufer sequence 方法）。"""
    if rng is None:
        rng = random.Random()
    if n <= 1:
        return (n, [])
    if n == 2:
        return (2, [(0, 1)])
    # Prufer sequence
    seq = [rng.randint(0, n - 1) for _ in range(n - 2)]
    degree = [1] * n
    for v in seq:
        degree[v] += 1
    edges: list[tuple[int, int]] = []
    for v in seq:
        for u in range(n):
            if degree[u] == 1:
                edges.append((min(u, v), max(u, v)))
                degree[u] -= 1
                degree[v] -= 1
                break
    # last edge
    last = [u for u in range(n) if degree[u] == 1]
    if len(last) == 2:
        edges.append((min(last[0], last[1]), max(last[0], last[1])))
    return (n, edges)


def random_cycle(n: int, rng: random.Random | None = None) -> tuple[int, list[tuple[int, int]]]:
    """生成随机环图。"""
    if rng is None:
        rng = random.Random()
    if n < 3:
        raise ValueError("Cycle requires n >= 3")
    perm = list(range(n))
    rng.shuffle(perm)
    edges = []
    for i in range(n):
        u, v = perm[i], perm[(i + 1) % n]
        edges.append((min(u, v), max(u, v)))
    return (n, sorted(set(edges)))


def random_ladder(n_rungs: int, rng: random.Random | None = None) -> tuple[int, list[tuple[int, int]]]:
    """生成梯子图 (ladder graph)，2*n_rungs 个节点。"""
    if rng is None:
        rng = random.Random()
    n = 2 * n_rungs
    edges = []
    for i in range(n_rungs - 1):
        edges.append((i, i + 1))
        edges.append((n_rungs + i, n_rungs + i + 1))
    for i in range(n_rungs):
        edges.append((i, n_rungs + i))
    edges = sorted(set((min(u, v), max(u, v)) for u, v in edges))
    return (n, edges)


def random_wheel(n: int, rng: random.Random | None = None) -> tuple[int, list[tuple[int, int]]]:
    """生成轮图：中心节点 0 + 外圈 n-1 个节点。"""
    if rng is None:
        rng = random.Random()
    if n < 4:
        raise ValueError("Wheel requires n >= 4")
    edges = []
    for i in range(1, n):
        edges.append((0, i))
    for i in range(1, n - 1):
        edges.append((i, i + 1))
    edges.append((1, n - 1))
    edges = sorted(set((min(u, v), max(u, v)) for u, v in edges))
    return (n, edges)


def attach_motif_triangle(
    base_n: int, base_edges: list[tuple[int, int]], rng: random.Random | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """在 base 图上附加一个三角形 motif。"""
    if rng is None:
        rng = random.Random()
    attach_node = rng.randint(0, base_n - 1)
    a, b = base_n, base_n + 1
    new_edges = base_edges + [(min(attach_node, a), max(attach_node, a)),
                               (min(attach_node, b), max(attach_node, b)),
                               (a, b)]
    return (base_n + 2, sorted(set(new_edges)))


def attach_motif_house(
    base_n: int, base_edges: list[tuple[int, int]], rng: random.Random | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """在 base 图上附加一个房子 motif (5 节点子图)。"""
    if rng is None:
        rng = random.Random()
    attach_node = rng.randint(0, base_n - 1)
    a, b, c, d = base_n, base_n + 1, base_n + 2, base_n + 3
    new_edges = base_edges + [
        (min(attach_node, a), max(attach_node, a)),
        (a, b), (b, c), (c, d), (d, a), (a, c),
    ]
    return (base_n + 4, sorted(set((min(u, v), max(u, v)) for u, v in new_edges)))


def attach_motif_crane(
    base_n: int, base_edges: list[tuple[int, int]], rng: random.Random | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """在 base 图上附加一个鹤 motif。"""
    if rng is None:
        rng = random.Random()
    attach_node = rng.randint(0, base_n - 1)
    a, b, c, d = base_n, base_n + 1, base_n + 2, base_n + 3
    new_edges = base_edges + [
        (min(attach_node, a), max(attach_node, a)),
        (a, b), (a, c), (a, d), (b, c), (c, d), (b, d),
    ]
    return (base_n + 4, sorted(set((min(u, v), max(u, v)) for u, v in new_edges)))


def format_graph(n: int, edges: list[tuple[int, int]]) -> str:
    """将图格式化为 (n, [(u,v), ...]) 字符串。"""
    edge_str = ",".join(f"({u},{v})" for u, v in edges)
    return f"({n}, [{edge_str}])"


def sample_trees_or_cycles(
    p: float, n_nodes: int, num_samples: int, seed: int | None = None,
) -> list[str]:
    """按分布采样：概率 p 生成 tree，1-p 生成 cycle。"""
    rng = random.Random(seed)
    results = []
    for _ in range(num_samples):
        if rng.random() < p:
            gn, ge = random_tree(n_nodes, rng)
        else:
            gn, ge = random_cycle(n_nodes, rng)
        results.append(format_graph(gn, ge))
    return results


def sample_union_of_components(
    p: float, comp_nodes: int, num_samples: int, seed: int | None = None,
) -> list[str]:
    """按分布采样 2 分量图：概率 p 两个都是 tree，1-p 两个都是 cycle。"""
    rng = random.Random(seed)
    results = []
    for _ in range(num_samples):
        all_edges = []
        if rng.random() < p:
            _, e1 = random_tree(comp_nodes, rng)
            _, e2 = random_tree(comp_nodes, rng)
        else:
            _, e1 = random_cycle(comp_nodes, rng)
            _, e2 = random_cycle(comp_nodes, rng)
        e2_shifted = [(u + comp_nodes, v + comp_nodes) for u, v in e2]
        all_edges = sorted(e1 + e2_shifted)
        results.append(format_graph(2 * comp_nodes, all_edges))
    return results


def sample_motif_graphs(
    p: float, base_nodes: int, num_samples: int, seed: int | None = None,
) -> list[str]:
    """按分布采样 motif 图：概率 p base 为 tree，否则为 ladder/wheel。"""
    rng = random.Random(seed)
    motif_funcs = [attach_motif_triangle, attach_motif_house, attach_motif_crane]
    results = []
    for _ in range(num_samples):
        if rng.random() < p:
            gn, ge = random_tree(base_nodes, rng)
        else:
            if rng.random() < 0.5:
                gn, ge = random_ladder(max(2, base_nodes // 2), rng)
            else:
                gn, ge = random_wheel(base_nodes, rng)
        motif_func = rng.choice(motif_funcs)
        gn, ge = motif_func(gn, ge, rng)
        results.append(format_graph(gn, ge))
    return results
