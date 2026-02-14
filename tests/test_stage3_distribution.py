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
    code, run_dir = run_stage3(output_root=root, run_id="case")
    assert code == 0
    assert (run_dir / "distribution_metrics.csv").exists()
    assert (run_dir / "distribution_samples.csv").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "llm_io.jsonl").exists()
    assert (run_dir / "run.log").exists()

    with (run_dir / "distribution_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    motif_row = next(r for r in rows if r["task_id"] == "motif")
    assert float(motif_row["parse_fail_rate"]) > 0.0
    assert float(motif_row["judge_fail_rate"]) > 0.0
    shutil.rmtree(root)
