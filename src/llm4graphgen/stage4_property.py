"""Stage4：Property-based（MolHIV）评测。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class SmilesParseResult:
    success: bool
    canonical_smiles: str | None
    parse_failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "canonical_smiles": self.canonical_smiles,
            "parse_failure_reason": self.parse_failure_reason,
        }


def parse_smiles(smiles: str) -> SmilesParseResult:
    text = smiles.strip()
    if not text:
        return SmilesParseResult(False, None, "SMILES 为空字符串。")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return SmilesParseResult(False, None, "RDKit 无法解析该 SMILES。")
    cano = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return SmilesParseResult(True, cano, None)


def _fingerprint_array(smiles: str, fp_size: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效 SMILES，无法生成指纹：{smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp_size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _few_shot_dataset() -> tuple[list[str], list[int]]:
    # 说明：此处为 Stage4 最小可复现的 few-shot 代理样本，不代表 OGBG-MolHIV 官方划分。
    positives = [
        "CCN(CC)CC",
        "CCOC(=O)N1CCN(CC1)C",
        "c1ccncc1",
        "c1ccccc1N",
        "CCN1CCN(CC1)C(=O)C",
    ]
    negatives = [
        "CCO",
        "CCCC",
        "CC(=O)O",
        "c1ccccc1",
        "C1CCCCC1",
        "CC(C)O",
    ]
    xs = positives + negatives
    ys = [1] * len(positives) + [0] * len(negatives)
    return xs, ys


def train_baseline_classifier() -> tuple[LogisticRegression, set[str]]:
    xs, ys = _few_shot_dataset()
    xmat = np.vstack([_fingerprint_array(x) for x in xs])
    yarr = np.array(ys, dtype=np.int64)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    clf.fit(xmat, yarr)
    train_cano = {parse_smiles(x).canonical_smiles for x in xs if parse_smiles(x).success}
    return clf, {x for x in train_cano if x is not None}


def _generated_mock_outputs() -> list[str]:
    return [
        "CCN(CC)CC",
        "CCOC(=O)N1CCN(CC1)C",
        "invalid_smiles",
        "c1ccccc1N",
        "CCO",
        "C1CC1",
        "C1=CC=CC=C1O",
        "C1CC1(",
        "N[N+](=O)[O-]",
        "CCN1CCN(CC1)C(=O)C",
        "CCN(CC)CC",
        "",
    ]


def run_stage4(output_root: Path, run_id: str, model: str = "mock-mol-stage4", temperature: float = 0.0) -> tuple[int, Path]:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clf, train_canonical = train_baseline_classifier()
    raw_outputs = _generated_mock_outputs()

    sample_rows: list[dict[str, object]] = []
    llm_rows: list[dict[str, object]] = []
    valid_scores: list[float] = []
    valid_labels: list[int] = []
    valid_canonical: list[str] = []

    for idx, raw in enumerate(raw_outputs, start=1):
        parse_result = parse_smiles(raw)
        score = ""
        pred_label = ""
        is_novel = ""

        if parse_result.success and parse_result.canonical_smiles is not None:
            cano = parse_result.canonical_smiles
            valid_canonical.append(cano)
            fp = _fingerprint_array(cano).reshape(1, -1)
            prob = float(clf.predict_proba(fp)[0, 1])
            label = int(prob >= 0.5)
            valid_scores.append(prob)
            valid_labels.append(label)
            score = f"{prob:.6f}"
            pred_label = label
            is_novel = cano not in train_canonical

        sample_rows.append(
            {
                "sample_id": idx,
                "raw_output": raw,
                "is_valid": parse_result.success,
                "canonical_smiles": parse_result.canonical_smiles or "",
                "parse_failure_reason": parse_result.parse_failure_reason or "",
                "classifier_score": score,
                "classifier_positive": pred_label,
                "is_novel_vs_fewshot": is_novel,
            }
        )

        llm_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task_id": "molhiv_property",
                "sample_id": idx,
                "provider": "mock",
                "prompt": "请生成可能具有 HIV 抑制性质的分子，输出 SMILES 字符串。",
                "model": model,
                "temperature": temperature,
                "raw_output": raw,
                "parsed_result": parse_result.to_dict() if parse_result.success else None,
                "parse_failure_reason": parse_result.parse_failure_reason,
            }
        )

    total = len(raw_outputs)
    valid_count = len(valid_canonical)
    valid_rate = valid_count / total if total else 0.0
    unique_valid = set(valid_canonical)
    unique_valid_count = len(unique_valid)
    unique_rate = unique_valid_count / valid_count if valid_count else 0.0
    novel_unique_count = sum(1 for smi in unique_valid if smi not in train_canonical)
    novel_rate = novel_unique_count / unique_valid_count if unique_valid_count else 0.0

    c_value = float(np.mean(valid_labels)) if valid_labels else 0.0
    cm_value = float(np.mean(valid_scores)) if valid_scores else 0.0
    rectified_c = c_value * valid_rate

    metrics_row = {
        "total_samples": total,
        "valid_count": valid_count,
        "valid_rate": f"{valid_rate:.4f}",
        "unique_valid_count": unique_valid_count,
        "unique_rate": f"{unique_rate:.4f}",
        "novel_unique_count": novel_unique_count,
        "novel_rate": f"{novel_rate:.4f}",
        "CM": f"{cm_value:.4f}",
        "C": f"{c_value:.4f}",
        "rectified_C": f"{rectified_c:.4f}",
    }

    _write_csv(run_dir / "property_samples.csv", sample_rows)
    _write_csv(run_dir / "property_metrics.csv", [metrics_row])
    _write_csv(
        run_dir / "failure_cases.csv",
        [row for row in sample_rows if not bool(row["is_valid"])],
    )
    _write_jsonl(run_dir / "llm_io.jsonl", llm_rows)
    _write_csv(
        run_dir / "baseline_train_summary.csv",
        [{"train_total": len(_few_shot_dataset()[0]), "train_unique_canonical": len(train_canonical)}],
    )
    _write_run_log(run_dir / "run.log", metrics_row)
    return 0, run_dir


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_run_log(path: Path, metrics_row: dict[str, object]) -> None:
    lines = [
        "Stage4 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 阶段范围：Property-based（MolHIV）",
        "- 指标口径：",
        "  - Unique = unique_valid_count / valid_count",
        "  - Novel = novel_unique_count / unique_valid_count（相对 few-shot 训练集）",
        "  - C = 有效样本中 baseline 判为正类的比例",
        "  - CM = 有效样本中 baseline 正类概率均值",
        "  - rectified_C = C * valid_rate",
        "",
        "结果摘要：",
        f"- total={metrics_row['total_samples']}, valid={metrics_row['valid_count']}, valid_rate={metrics_row['valid_rate']}",
        f"- unique_rate={metrics_row['unique_rate']}, novel_rate={metrics_row['novel_rate']}",
        f"- CM={metrics_row['CM']}, C={metrics_row['C']}, rectified_C={metrics_row['rectified_C']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage4 Property-based MolHIV 评测")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model", default="mock-mol-stage4")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    _, run_dir = run_stage4(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
    )
    print(f"Stage4 运行完成：{run_dir.as_posix()}")
    print("结果文件：property_metrics.csv / property_samples.csv / llm_io.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
