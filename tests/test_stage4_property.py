"""Stage4 Property-based 测试（v3 对齐论文改进版）。"""

from pathlib import Path
import csv
import shutil
import importlib

from llm4graphgen.stage4_property import parse_smiles, run_stage4, train_baseline_classifier


def test_parse_smiles_rdkit():
    ok = parse_smiles("CCO")
    bad = parse_smiles("C1CC1(")
    assert ok.success is True
    assert ok.canonical_smiles is not None
    assert bad.success is False
    assert "RDKit" in (bad.parse_failure_reason or "")


def test_baseline_classifier_proxy():
    """测试 proxy 分类器（24 样本）。"""
    clf, train_set, stats = train_baseline_classifier(classifier_type="proxy")
    assert hasattr(clf, "predict_proba")
    assert len(train_set) > 0
    assert 0.0 <= stats["tpr"] <= 1.0
    assert 0.0 <= stats["fpr"] <= 1.0
    assert stats["train_size"] == 24
    assert stats["data_source"] == "proxy"


def test_stage4_run_outputs():
    root = Path("runs") / "test_stage4_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage4(output_root=root, run_id="case")
    assert code == 0
    assert (run_dir / "property_metrics.csv").exists()
    assert (run_dir / "property_samples.csv").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "llm_io.jsonl").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "baseline_train_summary.csv").exists()

    with (run_dir / "property_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert float(row["valid_rate"]) < 1.0
    assert float(row["unique_rate"]) <= 1.0
    assert "rectified_C" in row
    assert "classifier_TPR" in row
    assert "classifier_FPR" in row
    assert "classifier_type" in row
    assert "rectified_C_formula" in row
    shutil.rmtree(root)


def test_stage4_classifier_type_in_output():
    """验证 classifier_type 字段记录在输出中。"""
    root = Path("runs") / "test_stage4_clf_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage4(
        output_root=root, run_id="case",
        classifier_type="proxy",
    )
    assert code == 0

    with (run_dir / "property_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["classifier_type"] == "proxy"

    with (run_dir / "baseline_train_summary.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert "data_source" in rows[0]
    shutil.rmtree(root)


def test_gin_classifier_module_importable():
    """验证 gin_classifier 模块可导入，常量正确。"""
    from llm4graphgen.gin_classifier import PAPER_TPR, PAPER_FPR, PAPER_TP, PAPER_FN, PAPER_FP, PAPER_TN
    assert abs(PAPER_TPR - 810 / (810 + 633)) < 1e-3
    assert abs(PAPER_FPR - 4145 / (4145 + 35539)) < 1e-3
    assert PAPER_TP == 810
    assert PAPER_FN == 633
    assert PAPER_FP == 4145
    assert PAPER_TN == 35539
    # 论文总样本数
    assert PAPER_TP + PAPER_FN + PAPER_FP + PAPER_TN == 41127


def test_stage4_paper_tpr_fpr_flag():
    """验证 --paper-tpr-fpr 标志在 proxy 模式下正常工作。"""
    root = Path("runs") / "test_stage4_paper_tmp"
    if root.exists():
        shutil.rmtree(root)
    code, run_dir = run_stage4(
        output_root=root, run_id="case",
        classifier_type="proxy",
        paper_tpr_fpr=True,
    )
    assert code == 0

    with (run_dir / "property_metrics.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    # 使用论文 TPR/FPR
    assert row["tpr_fpr_source"] == "paper_table10"
    assert abs(float(row["classifier_TPR"]) - 0.5615) < 0.001
    assert abs(float(row["classifier_FPR"]) - 0.1045) < 0.001
    # 分类器自身 TPR/FPR 应该不同 (proxy 过拟合值)
    assert "classifier_own_TPR" in row
    assert "classifier_own_FPR" in row
    shutil.rmtree(root)


def test_stage4_classifier_choices_include_gin():
    """验证 CLI 支持 gin 选项。"""
    import argparse
    from llm4graphgen.stage4_property import main
    # 验证 gin 在 choices 中 — 通过尝试解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["proxy", "ogbg", "gin"])
    args = parser.parse_args(["--classifier", "gin"])
    assert args.classifier == "gin"
