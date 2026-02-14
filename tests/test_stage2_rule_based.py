from pathlib import Path
import shutil

from llm4graphgen.stage2_rule_based import (
    Graph,
    canonical_signature,
    components_count,
    is_bipartite,
    is_k_colorable,
    is_planar_approx,
    is_tree,
    is_wheel,
    run_stage2,
)


def test_basic_graph_rules():
    tree = Graph(n=5, edges=((0, 1), (1, 2), (2, 3), (3, 4)))
    assert is_tree(tree) is True
    assert components_count(tree) == 1

    odd_cycle = Graph(n=5, edges=((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)))
    assert is_bipartite(odd_cycle) is False
    assert is_k_colorable(odd_cycle, 3) is True


def test_planar_and_wheel_rules():
    k5 = Graph(
        n=5,
        edges=((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
    )
    assert is_planar_approx(k5) is False

    w6 = Graph(
        n=6,
        edges=((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)),
    )
    assert is_wheel(w6) is True


def test_stage2_run_generates_outputs():
    root = Path("runs") / "test_stage2_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage2(output_root=root, run_id="case")
    assert code == 0
    assert run_dir == root / "case"
    assert (run_dir / "llm_io.jsonl").exists()
    assert (run_dir / "rule_based_samples.csv").exists()
    assert (run_dir / "rule_based_metrics.csv").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "run.log").exists()
    shutil.rmtree(root)


def test_signature_is_stable_for_edge_order():
    s1 = canonical_signature(4, [(0, 1), (1, 2), (2, 3)])
    s2 = canonical_signature(4, [(2, 3), (1, 2), (0, 1)])
    assert s1 == s2
