"""Prompt 模板模块：对齐论文 4 种 prompting 策略。

策略：Zero-shot / Few-shot / Zero-shot+CoT / Few-shot+CoT
图输出格式：(n, [(u,v), ...])
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Rule-based task prompt templates (对齐论文 Table 7 参数)
# ---------------------------------------------------------------------------

# 论文使用英文 prompt，以下模板与论文对齐
RULE_TASK_DESCRIPTIONS: dict[str, dict] = {
    "tree": {
        "desc": "Generate a tree (connected acyclic graph) with {n} nodes.",
        "default_n": 15,
        "params": {},
        "few_shot_examples": [
            "(6, [(0,1),(1,2),(2,3),(3,4),(4,5)])",
            "(6, [(0,1),(0,2),(0,3),(1,4),(1,5)])",
        ],
        "cot_hint": (
            "Think step by step: 1) Start with node 0. "
            "2) Connect each new node to exactly one existing node. "
            "3) Ensure exactly n-1 edges and all nodes are connected."
        ),
    },
    "cycle": {
        "desc": "Generate a single cycle graph with {n} nodes.",
        "default_n": 15,
        "params": {},
        "few_shot_examples": [
            "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])",
            "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
        ],
        "cot_hint": (
            "Think step by step: 1) Create a path visiting all {n} nodes: 0-1-2-...-{nm1}. "
            "2) Add a final edge from node {nm1} back to node 0 to complete the cycle. "
            "3) Each node should have exactly degree 2."
        ),
    },
    "planar": {
        "desc": "Generate a planar graph with {n} nodes and {m} edges.",
        "default_n": 15,
        "params": {"m": 24},
        "few_shot_examples": [
            "(6, [(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,3)])",
            "(5, [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)])",
        ],
        "cot_hint": (
            "Think step by step: 1) A planar graph can be drawn without edge crossings. "
            "2) For {n} nodes, a planar graph has at most 3n-6 = {max_edges} edges. "
            "3) Build the graph incrementally, avoiding K5 and K3,3 subgraph structures. "
            "4) Ensure exactly {m} edges."
        ),
    },
    "components": {
        "desc": "Generate a graph with {n} nodes that has exactly {k} connected components.",
        "default_n": 15,
        "params": {"k": 5},
        "few_shot_examples": [
            "(8, [(0,1),(2,3),(4,5),(6,7)])",
            "(6, [(0,1),(0,2),(3,4)])",
        ],
        "cot_hint": (
            "Think step by step: 1) Partition the {n} nodes into exactly {k} groups. "
            "2) Add edges only within each group, never between groups. "
            "3) Ensure each group has at least one node and is internally connected."
        ),
    },
    "k_regular": {
        "desc": "Generate a {k}-regular graph with {n} nodes where every node has degree exactly {k}.",
        "default_n": 16,
        "params": {"k": 3},
        "few_shot_examples": [
            "(6, [(0,1),(0,3),(0,5),(1,2),(1,4),(2,3),(2,5),(3,4),(4,5)])",
            "(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])",
        ],
        "cot_hint": (
            "Think step by step: 1) Every node must have exactly {k} neighbors. "
            "2) The total number of edges must be n*k/2 = {total_edges}. "
            "3) Build the graph ensuring the degree constraint is met for all nodes."
        ),
    },
    "wheel": {
        "desc": "Generate a wheel graph with {n} nodes.",
        "default_n": 15,
        "params": {},
        "few_shot_examples": [
            "(5, [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)])",
            "(6, [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5),(5,1)])",
        ],
        "cot_hint": (
            "Think step by step: 1) Choose node 0 as the center (hub). "
            "2) Connect node 0 to all other nodes 1..{nm1}. "
            "3) Connect the rim nodes in a cycle: 1-2-3-...-{nm1}-1. "
            "4) The center has degree {nm1}, each rim node has degree 3."
        ),
    },
    "bipartite": {
        "desc": "Generate a bipartite graph with two partitions of {k} nodes each.",
        "default_n": 10,
        "params": {"k": 5},
        "few_shot_examples": [
            "(6, [(0,3),(0,4),(0,5),(1,3),(1,4),(2,5)])",
            "(6, [(0,3),(0,4),(1,4),(1,5),(2,3),(2,5)])",
        ],
        "cot_hint": (
            "Think step by step: 1) Split nodes into two groups: {{0,...,{km1}}} and {{{k},...,{nm1}}}. "
            "2) Only add edges between the two groups, never within a group. "
            "3) A graph is bipartite if and only if it has no odd-length cycles."
        ),
    },
    "k_coloring": {
        "desc": "Generate a graph with {n} nodes and {m} edges that is {k}-colorable.",
        "default_n": 15,
        "params": {"k": 3, "m": 32},
        "few_shot_examples": [
            "(5, [(0,1),(1,2),(2,3),(3,4),(4,0)])",
            "(6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3)])",
        ],
        "cot_hint": (
            "Think step by step: 1) A {k}-colorable graph can be vertex-colored with {k} colors "
            "such that no two adjacent vertices share the same color. "
            "2) Partition nodes into {k} color classes. "
            "3) Add edges only between nodes of different colors to guarantee {k}-colorability. "
            "4) Ensure exactly {m} edges total."
        ),
    },
}

OUTPUT_FORMAT_INSTRUCTION = (
    "Output the graph in this exact format: (n, [(u1,v1), (u2,v2), ...])\n"
    "where n is the number of nodes (labeled 0 to n-1) and the list contains all edges.\n"
    "Output ONLY the graph tuple, no explanation."
)

OUTPUT_FORMAT_INSTRUCTION_COT = (
    "Output the graph in this exact format: (n, [(u1,v1), (u2,v2), ...])\n"
    "where n is the number of nodes (labeled 0 to n-1) and the list contains all edges.\n"
    "Show your reasoning first, then output the final graph on the last line."
)


def build_rule_prompt(
    task_id: str,
    strategy: str,
    n: int | None = None,
) -> str:
    """构建 Rule-based 任务的 prompt。

    strategy: "zero_shot" | "few_shot" | "zero_shot_cot" | "few_shot_cot"
    """
    cfg = RULE_TASK_DESCRIPTIONS[task_id]
    if n is None:
        n = cfg["default_n"]
    params = dict(cfg["params"])
    params["n"] = n
    params["nm1"] = n - 1
    params["km1"] = params.get("k", 2) - 1
    if "k" in params:
        params["total_edges"] = n * params["k"] // 2
    if "m" not in params:
        params["m"] = 0
    params["max_edges"] = 3 * n - 6

    desc = cfg["desc"].format(**params)

    parts: list[str] = []

    if strategy in ("few_shot", "few_shot_cot"):
        parts.append("Here are some examples:\n")
        for i, ex in enumerate(cfg["few_shot_examples"], 1):
            parts.append(f"Example {i}: {ex}")
        parts.append("")

    parts.append(desc)

    if strategy in ("zero_shot_cot", "few_shot_cot"):
        cot = cfg["cot_hint"].format(**params)
        parts.append("")
        parts.append(cot)
        parts.append("")
        parts.append(OUTPUT_FORMAT_INSTRUCTION_COT)
    else:
        parts.append("")
        parts.append(OUTPUT_FORMAT_INSTRUCTION)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Distribution-based task prompt templates
# ---------------------------------------------------------------------------

DISTRIBUTION_TASKS: dict[str, dict] = {
    # 支持 task_id 直接查找和通配查找
    "trees_or_cycles": {
        "desc": (
            "Below are {num_input} graphs sampled from a distribution where each graph "
            "is a tree with probability p and a cycle with probability 1-p.\n\n"
            "{input_graphs}\n\n"
            "Based on these examples, infer the value of p. "
            "Then generate {num_output} new graphs from the same distribution."
        ),
    },
    "union_of_components": {
        "desc": (
            "Below are {num_input} graphs. Each graph consists of exactly 2 connected components. "
            "Each component is either a tree or a cycle. With probability p both components "
            "are trees, and with probability 1-p both are cycles.\n\n"
            "{input_graphs}\n\n"
            "Based on these examples, infer the value of p. "
            "Then generate {num_output} new graphs from the same distribution."
        ),
    },
    "motif": {
        "desc": (
            "Below are {num_input} graphs. Each graph is constructed by combining a base graph "
            "with a motif. The base graph is selected from {{tree, ladder, wheel}} and the motif "
            "is selected from {{triangle, house, crane}}. With probability p the base is a tree, "
            "otherwise it is a ladder or wheel.\n\n"
            "{input_graphs}\n\n"
            "Based on these examples, infer the value of p. "
            "Then generate {num_output} new graphs from the same distribution."
        ),
    },
}


def build_distribution_prompt(
    task_id: str,
    input_graphs: list[str],
    num_output: int = 10,
    strategy: str = "zero_shot",
) -> str:
    """构建 Distribution-based 任务的 prompt。"""
    # 支持 motif_triangle -> motif 映射
    lookup_id = task_id
    if lookup_id not in DISTRIBUTION_TASKS:
        for key in DISTRIBUTION_TASKS:
            if task_id.startswith(key) or key.startswith(task_id.split("_")[0]):
                lookup_id = key
                break
    cfg = DISTRIBUTION_TASKS[lookup_id]
    graphs_text = "\n".join(f"Graph {i+1}: {g}" for i, g in enumerate(input_graphs))
    desc = cfg["desc"].format(
        num_input=len(input_graphs),
        num_output=num_output,
        input_graphs=graphs_text,
    )

    parts = [desc, ""]

    if strategy in ("zero_shot_cot", "few_shot_cot"):
        parts.append(
            "Think step by step: First analyze the input graphs to determine "
            "if each is a tree or cycle. Count the proportions to estimate p. "
            "Then generate new graphs according to your estimated distribution."
        )
        parts.append("")
        parts.append(OUTPUT_FORMAT_INSTRUCTION_COT)
    else:
        parts.append(OUTPUT_FORMAT_INSTRUCTION)

    parts.append(
        "\nOutput each graph on a separate line. "
        "Also state your estimated p value on the first line as: p = <value>"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Property-based (MolHIV) prompt templates
# ---------------------------------------------------------------------------

def build_property_prompt(
    positive_examples: list[str],
    num_generate: int = 20,
    strategy: str = "few_shot",
) -> str:
    """构建 Property-based 分子生成的 prompt。"""
    examples_text = "\n".join(f"  {smi}" for smi in positive_examples)

    parts: list[str] = []
    parts.append(
        "The following molecules are known to inhibit HIV replication. "
        "They are represented in SMILES notation:\n"
    )
    parts.append(examples_text)
    parts.append("")
    parts.append(
        f"Generate {num_generate} new molecules (in SMILES notation) that "
        "are likely to also inhibit HIV replication. "
        "Each molecule should be on a separate line. "
        "Output ONLY the SMILES strings, one per line."
    )

    if strategy in ("few_shot_cot", "zero_shot_cot"):
        parts.append("")
        parts.append(
            "Think step by step about what structural features these molecules share "
            "that might contribute to HIV inhibition, then generate molecules with "
            "similar features. Show your reasoning first, then list the SMILES."
        )

    return "\n".join(parts)
