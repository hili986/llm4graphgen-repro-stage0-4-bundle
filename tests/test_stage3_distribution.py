"""Stage3 Distribution-based 测试（v3 对齐论文改进版）。"""

from pathlib import Path
import csv
import shutil

from llm4graphgen.stage2_rule_based import Graph
from llm4graphgen.stage3_distribution import (
    judge_motif,
    judge_trees_or_cycles,
    judge_union_of_components,
    run_stage3,
)


def test_distribution_judges_basic():
    tree = Graph(n=4, edges=((0, 1), (1, 2), (2, 3)))
    cycle = Graph(n=4, edges=((0, 1), (1, 2), (2, 3), (0, 3)))
    assert judge_trees_or_cycles(tree) is True
    assert judge_trees_or_cycles(cycle) is True

    union_ok = Graph(n=6, edges=((0, 1), (1, 2), (3, 4), (4, 5)))
    assert judge_union_of_components(union_ok) is True


def test_motif_triangle_match():
    has_triangle = Graph(n=5, edges=((0, 1), (1, 2), (0, 2), (2, 3)))
    no_triangle = Graph(n=5, edges=((0, 1), (1, 2), (2, 3), (3, 4)))
    assert judge_motif(has_triangle, motif="triangle") is True
    assert judge_motif(no_triangle, motif="triangle") is False


def test_stage3_run_outputs_and_metrics():
    root = Path("runs") / "test_stage3_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage3(
        output_root=root, run_id="case",
        p_values=[0.5], num_output=5,
    )
    assert code == 0
    assert (run_dir / "distribution_metrics.csv").exists()
    assert (run_dir / "distribution_samples.csv").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "llm_io.jsonl").exists()
    assert (run_dir / "run.log").exists()

    with (run_dir / "distribution_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        assert "p_true" in row
        assert "p_gen" in row
        assert "p_pred" in row
        assert "strategy" in row
    shutil.rmtree(root)


def test_stage3_all_strategies():
    """[P1b] 验证 strategy='all' 能运行全部 4 种策略。"""
    root = Path("runs") / "test_stage3_all_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage3(
        output_root=root, run_id="case",
        strategy="all",
        p_values=[0.5], num_output=3,
    )
    assert code == 0

    with (run_dir / "distribution_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # 4 strategies x 3 tasks x 1 p-value = 12 rows
    assert len(rows) == 12
    strategies_seen = set(row["strategy"] for row in rows)
    assert strategies_seen == {"zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"}
    shutil.rmtree(root)


def test_stage3_multi_repeat():
    """[P3b] 验证多次重复实验。"""
    root = Path("runs") / "test_stage3_repeat_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage3(
        output_root=root, run_id="case",
        p_values=[0.5], num_output=3,
        num_repeats=2,
    )
    assert code == 0
    # 应有 summary 文件
    assert (run_dir / "distribution_summary.csv").exists()

    with (run_dir / "distribution_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # 2 repeats x 3 tasks x 1 p-value = 6 rows
    assert len(rows) == 6
    shutil.rmtree(root)
