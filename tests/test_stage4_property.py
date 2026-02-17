"""Stage4 Property-based 测试（对齐论文改进版）。"""

from pathlib import Path
import csv
import shutil

from llm4graphgen.stage4_property import parse_smiles, run_stage4, train_baseline_classifier


def test_parse_smiles_rdkit():
    ok = parse_smiles("CCO")
    bad = parse_smiles("C1CC1(")
    assert ok.success is True
    assert ok.canonical_smiles is not None
    assert bad.success is False
    assert "RDKit" in (bad.parse_failure_reason or "")


def test_baseline_classifier_trainable():
    clf, train_set, stats = train_baseline_classifier()
    assert hasattr(clf, "predict_proba")
    assert len(train_set) > 0
    # 验证 FPR/TPR 在合理范围
    assert 0.0 <= stats["tpr"] <= 1.0
    assert 0.0 <= stats["fpr"] <= 1.0
    # 训练集应有 24 个样本（12 正 + 12 负）
    assert stats["train_size"] == 24


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
    # 验证 rectified_C 使用论文公式
    assert "rectified_C" in row
    assert "classifier_TPR" in row
    assert "classifier_FPR" in row
    assert "rectified_C_formula" in row
    shutil.rmtree(root)
