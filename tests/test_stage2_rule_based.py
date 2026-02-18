"""Stage2 Rule-based 测试（v3 对齐论文改进版）。"""

from pathlib import Path
import shutil

from llm4graphgen.stage2_rule_based import (
    Graph,
    canonical_signature,
    isomorphism_hash,
    components_count,
    is_bipartite,
    is_k_colorable,
    is_planar,
    is_tree,
    is_cycle,
    is_wheel,
    is_k_regular,
    run_stage2,
    SIZE_PRESETS,
)


def test_basic_graph_rules():
    tree = Graph(n=5, edges=((0, 1), (1, 2), (2, 3), (3, 4)))
    assert is_tree(tree) is True
    assert is_cycle(tree) is False
    assert components_count(tree) == 1

    odd_cycle = Graph(n=5, edges=((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)))
    assert is_cycle(odd_cycle) is True
    assert is_bipartite(odd_cycle) is False
    assert is_k_colorable(odd_cycle, 3) is True


def test_planar_exact():
    """精确平面性检测（NetworkX Boyer-Myrvold）。"""
    k5 = Graph(
        n=5,
        edges=((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
    )
    assert is_planar(k5) is False

    k4 = Graph(n=4, edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    assert is_planar(k4) is True

    k33 = Graph(n=6, edges=((0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)))
    assert is_planar(k33) is False


def test_wheel_and_regular():
    w6 = Graph(
        n=6,
        edges=((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)),
    )
    assert is_wheel(w6) is True

    k4 = Graph(n=4, edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    assert is_k_regular(k4, 3) is True


def test_stage2_run_generates_outputs():
    root = Path("runs") / "test_stage2_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage2(output_root=root, run_id="case", num_samples=6)
    assert code == 0
    assert (run_dir / "llm_io.jsonl").exists()
    assert (run_dir / "rule_based_metrics.csv").exists()
    assert (run_dir / "rule_based_summary.csv").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "run.log").exists()
    shutil.rmtree(root)


def test_signature_is_stable_for_edge_order():
    s1 = canonical_signature(4, [(0, 1), (1, 2), (2, 3)])
    s2 = canonical_signature(4, [(2, 3), (1, 2), (0, 1)])
    assert s1 == s2


def test_isomorphism_hash_detects_relabeling():
    """[P2a] 同构图应有相同 hash，不同构图应有不同 hash。"""
    # 路径图 0-1-2-3 和重标号后的 3-2-1-0（同构）
    h1 = isomorphism_hash(4, [(0, 1), (1, 2), (2, 3)])
    h2 = isomorphism_hash(4, [(3, 2), (2, 1), (1, 0)])
    assert h1 == h2

    # 星图 vs 路径图（不同构）
    h_star = isomorphism_hash(4, [(0, 1), (0, 2), (0, 3)])
    h_path = isomorphism_hash(4, [(0, 1), (1, 2), (2, 3)])
    assert h_star != h_path


def test_size_presets():
    """[P2b] 验证三种规模配置。"""
    assert "small" in SIZE_PRESETS
    assert "medium" in SIZE_PRESETS
    assert "large" in SIZE_PRESETS

    # medium 应与论文 Table 7 一致
    assert SIZE_PRESETS["medium"]["tree"]["n"] == 15
    assert SIZE_PRESETS["medium"]["k_regular"]["n"] == 16
    assert SIZE_PRESETS["medium"]["k_regular"]["k"] == 3
    assert SIZE_PRESETS["medium"]["bipartite"]["n"] == 10
    assert SIZE_PRESETS["medium"]["k_coloring"]["m"] == 32

    # small < medium < large
    for task_id in SIZE_PRESETS["small"]:
        assert SIZE_PRESETS["small"][task_id]["n"] < SIZE_PRESETS["medium"][task_id]["n"]
        assert SIZE_PRESETS["medium"][task_id]["n"] < SIZE_PRESETS["large"][task_id]["n"]


def test_stage2_with_isomorphism_unique():
    """[P2a] 验证 isomorphism 去重方式能正常运行。"""
    root = Path("runs") / "test_stage2_iso_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage2(
        output_root=root, run_id="case", num_samples=6,
        unique_method="isomorphism",
    )
    assert code == 0
    assert (run_dir / "rule_based_summary.csv").exists()
    shutil.rmtree(root)


def test_stage2_small_size():
    """[P2b] 验证 small 规模能正常运行。"""
    root = Path("runs") / "test_stage2_small_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage2(
        output_root=root, run_id="case", num_samples=6,
        size="small",
    )
    assert code == 0
    shutil.rmtree(root)
