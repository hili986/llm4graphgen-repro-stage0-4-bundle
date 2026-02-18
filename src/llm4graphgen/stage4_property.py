"""Stage4：Property-based（MolHIV）评测 — 对齐论文实验设计。

改进点（v3, vs v2）：
- [P0] 支持 OGBG-MolHIV 全量数据集训练分类器，消除过拟合
- [P0+] 支持 GIN (Graph Isomorphism Network) 分类器，与论文架构一致
- [P0+] 支持 --paper-tpr-fpr 使用论文 Table 10 的 TPR/FPR 值
- [P1a] 默认生成 100 个分子，多次 LLM 调用自动拼接
- 修正 rectified_C 公式为论文定义：C(G) = (C_M(G) - FPR) / (TPR - FPR)
- 支持 Few-shot 和 Few-shot+CoT 两种策略
- 支持真实 LLM 分子生成
- 增加分子描述符分析
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


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


def _mol_descriptors(smiles: str) -> dict[str, float]:
    """计算分子描述符。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
    }


# ---------------------------------------------------------------------------
# 代理训练数据 — 用于 proxy 分类器和 few-shot prompt
# ---------------------------------------------------------------------------

def _few_shot_dataset() -> tuple[list[str], list[int]]:
    """扩展的 MolHIV 代理 few-shot 数据集。

    正类：已知 HIV 抑制活性分子的 SMILES（来源：文献报道的 HIV 抑制剂骨架）
    负类：常见无活性小分子
    """
    positives = [
        "CCN(CC)CC",
        "CCOC(=O)N1CCN(CC1)C",
        "c1ccncc1",
        "c1ccccc1N",
        "CCN1CCN(CC1)C(=O)C",
        "c1ccc2[nH]ccc2c1",          # indole
        "c1cnc2ccccc2n1",             # quinazoline
        "c1ccc(-c2ccccn2)cc1",        # 2-phenylpyridine
        "CC(=O)Nc1ccc(O)cc1",         # paracetamol-like
        "c1ccc2c(c1)c1ccccc1[nH]2",  # carbazole
        "NS(=O)(=O)c1ccc(N)cc1",     # sulfanilamide
        "Cc1ccc(NC(=O)c2ccccc2)cc1",  # benzanilide
    ]
    negatives = [
        "CCO",
        "CCCC",
        "CC(=O)O",
        "c1ccccc1",
        "C1CCCCC1",
        "CC(C)O",
        "CCCCCC",                     # hexane
        "CC(=O)CC",                   # butanone
        "OC(=O)CC(O)=O",             # malonic acid
        "CCOC(=O)CC",                 # ethyl propanoate
        "C1CCCC1",                    # cyclopentane
        "CCOCC",                      # diethyl ether
    ]
    xs = positives + negatives
    ys = [1] * len(positives) + [0] * len(negatives)
    return xs, ys


def train_baseline_classifier(
    classifier_type: str = "proxy",
    data_root: str = "data",
    verbose: bool = True,
) -> tuple:
    """训练 baseline 分类器。

    Args:
        classifier_type: "proxy" | "ogbg" | "gin"
            - proxy: 24 个代理样本 LR
            - ogbg: OGBG-MolHIV 全量 LR
            - gin: OGBG-MolHIV GIN (Graph Isomorphism Network)
        data_root: OGBG-MolHIV 数据目录 (ogbg/gin 模式)

    Returns:
        (clf, train_canonical, classifier_stats)
        对于 gin 模式，clf 是 GIN 模型对象。
    """
    if classifier_type == "gin":
        from llm4graphgen.gin_classifier import train_gin_classifier
        return train_gin_classifier(data_root=data_root, verbose=verbose)

    if classifier_type == "ogbg":
        from llm4graphgen.molhiv_classifier import train_ogbg_classifier
        return train_ogbg_classifier(data_root=data_root, verbose=verbose)

    # proxy 模式 — 向后兼容 v2
    xs, ys = _few_shot_dataset()
    xmat = np.vstack([_fingerprint_array(x) for x in xs])
    yarr = np.array(ys, dtype=np.int64)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    clf.fit(xmat, yarr)

    y_pred = clf.predict(xmat)
    cm = confusion_matrix(yarr, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    train_cano = {parse_smiles(x).canonical_smiles for x in xs if parse_smiles(x).success}

    stats: dict[str, object] = {
        "tpr": tpr,
        "fpr": fpr,
        "train_size": len(xs),
        "train_positive": sum(ys),
        "train_negative": len(ys) - sum(ys),
        "confusion_matrix": cm.tolist(),
        "classifier_type": "LogisticRegression(proxy-24)",
        "data_source": "proxy",
        "eval_set": "train",
    }

    return clf, {x for x in train_cano if x is not None}, stats


def _generated_mock_outputs() -> list[str]:
    """扩展的 Mock LLM 输出（模拟 Few-shot+CoT 生成）。"""
    return [
        "CCN(CC)CC",
        "CCOC(=O)N1CCN(CC1)C",
        "c1ccccc1N",
        "c1ccncc1",
        "CCN1CCN(CC1)C(=O)C",
        "c1ccc(-c2ccccn2)cc1",
        "CC(=O)Nc1ccccc1",
        "c1ccc2[nH]ccc2c1",
        "CCO",
        "C1CC1",
        "C1=CC=CC=C1O",
        "N[N+](=O)[O-]",
        "CCCCCCCC",
        "CC(C)(C)O",
        "c1ccnc(-c2ccccc2)c1",
        "CC(=O)N1CCCC1",
        "c1cc(N)ccc1O",
        "Nc1ccc(Cl)cc1",
        "invalid_smiles",
        "C1CC1(",
        "",
    ]


# ---------------------------------------------------------------------------
# LLM 输出解析
# ---------------------------------------------------------------------------

def extract_smiles_from_response(response: str) -> list[str]:
    """从 LLM 响应中提取 SMILES 字符串列表。"""
    lines = response.strip().split("\n")
    smiles_list = []
    for line in lines:
        line = line.strip()
        if not line or len(line) > 200:
            continue
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
        cleaned = re.sub(r'^[-*]\s*', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned and not cleaned.startswith(('#', '//', 'Note', 'The', 'These', 'I ')):
            smiles_list.append(cleaned)
    return smiles_list


# ---------------------------------------------------------------------------
# 核心运行引擎
# ---------------------------------------------------------------------------

def run_stage4(
    output_root: Path,
    run_id: str,
    model: str = "mock-mol-stage4",
    temperature: float = 0.5,
    provider=None,
    strategy: str = "few_shot",
    num_generate: int = 100,
    classifier_type: str = "proxy",
    data_root: str = "data",
    paper_tpr_fpr: bool = False,
) -> tuple[int, Path]:
    """运行 Stage4 评测。

    Args:
        num_generate: 生成分子数（论文为 100+，v3 默认 100）
        classifier_type: "proxy" | "ogbg" | "gin"
        data_root: OGBG-MolHIV 数据目录
        paper_tpr_fpr: 是否使用论文 Table 10 的 TPR/FPR 计算 rectified_C
    """
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [Stage4] 分类器模式: {classifier_type}")
    if paper_tpr_fpr:
        print("  [Stage4] 使用论文 Table 10 TPR/FPR 计算 rectified_C")
    clf, train_canonical, clf_stats = train_baseline_classifier(
        classifier_type=classifier_type, data_root=data_root
    )
    is_gin = classifier_type == "gin"
    use_mock = provider is None

    # 获取 LLM 输出
    if use_mock:
        raw_outputs = _generated_mock_outputs()
    else:
        from llm4graphgen.prompts import build_property_prompt
        xs_pos = [x for x, y in zip(*_few_shot_dataset()) if y == 1]

        raw_outputs = []
        # [P1a] 多次调用 LLM 以收集足够多的分子
        # 每次请求生成 batch_size 个，直到达到 num_generate
        batch_size = min(50, num_generate)
        collected = 0
        max_calls = (num_generate // batch_size) + 3  # 允许多调几次以补偿解析失败

        for call_idx in range(max_calls):
            remaining = num_generate - collected
            if remaining <= 0:
                break
            request_size = min(batch_size, remaining + 10)  # 多请求一些以补偿

            prompt = build_property_prompt(xs_pos[:5], request_size, strategy)
            try:
                resp = provider.generate(prompt, model, temperature)
                batch = extract_smiles_from_response(resp)
                raw_outputs.extend(batch)
                collected += len(batch)
                print(f"  [Stage4] LLM 调用 {call_idx+1}: 获取 {len(batch)} 个分子 "
                      f"(累计 {len(raw_outputs)})")
            except Exception as exc:
                raw_outputs.append(f"LLM_ERROR: {exc}")
                print(f"  [Stage4] LLM 调用 {call_idx+1}: 错误 - {exc}")

    sample_rows: list[dict[str, object]] = []
    llm_rows: list[dict[str, object]] = []
    valid_scores: list[float] = []
    valid_labels: list[int] = []
    valid_canonical: list[str] = []

    # GIN 模式：先收集所有有效 SMILES，批量推理
    if is_gin:
        # 先解析所有 SMILES
        parsed_results = [(i, raw, parse_smiles(raw)) for i, raw in enumerate(raw_outputs, start=1)]
        valid_smiles_for_gin = [
            pr.canonical_smiles
            for _, _, pr in parsed_results
            if pr.success and pr.canonical_smiles is not None
        ]
        # 批量 GIN 推理
        if valid_smiles_for_gin:
            from llm4graphgen.gin_classifier import gin_predict_proba
            gin_probs = gin_predict_proba(clf, valid_smiles_for_gin)
        else:
            gin_probs = []
        gin_prob_idx = 0

    for idx, raw in enumerate(raw_outputs, start=1):
        parse_result = parse_smiles(raw)
        score = ""
        pred_label = ""
        is_novel = ""
        descriptors: dict[str, float] = {}

        if parse_result.success and parse_result.canonical_smiles is not None:
            cano = parse_result.canonical_smiles
            valid_canonical.append(cano)

            if is_gin:
                # GIN 打分
                prob_val = gin_probs[gin_prob_idx] if gin_prob_idx < len(gin_probs) else None
                gin_prob_idx += 1
                if prob_val is not None:
                    prob = prob_val
                else:
                    prob = 0.0  # GIN 无法处理的分子
            else:
                # LR 打分
                fp = _fingerprint_array(cano).reshape(1, -1)
                prob = float(clf.predict_proba(fp)[0, 1])

            label = int(prob >= 0.5)
            valid_scores.append(prob)
            valid_labels.append(label)
            score = f"{prob:.6f}"
            pred_label = label
            is_novel = cano not in train_canonical
            descriptors = _mol_descriptors(cano)

        sample_rows.append({
            "sample_id": idx,
            "raw_output": raw,
            "is_valid": parse_result.success,
            "canonical_smiles": parse_result.canonical_smiles or "",
            "parse_failure_reason": parse_result.parse_failure_reason or "",
            "classifier_score": score,
            "classifier_positive": pred_label,
            "is_novel_vs_train": is_novel,
            "mol_weight": descriptors.get("mol_weight", ""),
            "logp": descriptors.get("logp", ""),
            "num_h_donors": descriptors.get("num_h_donors", ""),
            "num_h_acceptors": descriptors.get("num_h_acceptors", ""),
        })

        llm_rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_id": "molhiv_property",
            "sample_id": idx,
            "provider": "mock" if use_mock else provider.name,
            "strategy": strategy,
            "model": model,
            "temperature": temperature,
            "raw_output": raw,
            "parsed_result": parse_result.to_dict() if parse_result.success else None,
            "parse_failure_reason": parse_result.parse_failure_reason,
        })

    # 计算指标
    total = len(raw_outputs)
    valid_count = len(valid_canonical)
    valid_rate = valid_count / total if total else 0.0
    unique_valid = set(valid_canonical)
    unique_valid_count = len(unique_valid)
    unique_rate = unique_valid_count / valid_count if valid_count else 0.0
    novel_unique_count = sum(1 for smi in unique_valid if smi not in train_canonical)
    novel_rate = novel_unique_count / unique_valid_count if unique_valid_count else 0.0

    # C_M(G) — 分类器原始预测正类概率均值
    cm_value = float(np.mean(valid_scores)) if valid_scores else 0.0
    c_raw = float(np.mean(valid_labels)) if valid_labels else 0.0

    # rectified_C — 论文公式：C(G) = (C_M(G) - FPR) / (TPR - FPR)
    clf_tpr = float(clf_stats["tpr"])
    clf_fpr = float(clf_stats["fpr"])

    # 使用论文 Table 10 TPR/FPR 或分类器自身的 TPR/FPR
    if paper_tpr_fpr:
        from llm4graphgen.gin_classifier import PAPER_TPR, PAPER_FPR
        tpr = PAPER_TPR
        fpr = PAPER_FPR
        tpr_fpr_source = "paper_table10"
    else:
        tpr = clf_tpr
        fpr = clf_fpr
        tpr_fpr_source = "classifier"

    if abs(tpr - fpr) > 1e-6:
        rectified_c = (cm_value - fpr) / (tpr - fpr)
        rectified_c = max(0.0, min(1.0, rectified_c))  # clip to [0, 1]
    else:
        rectified_c = 0.0

    metrics_row = {
        "total_samples": total,
        "valid_count": valid_count,
        "valid_rate": f"{valid_rate:.4f}",
        "unique_valid_count": unique_valid_count,
        "unique_rate": f"{unique_rate:.4f}",
        "novel_unique_count": novel_unique_count,
        "novel_rate": f"{novel_rate:.4f}",
        "CM": f"{cm_value:.4f}",
        "C_raw": f"{c_raw:.4f}",
        "rectified_C": f"{rectified_c:.4f}",
        "classifier_TPR": f"{tpr:.4f}",
        "classifier_FPR": f"{fpr:.4f}",
        "classifier_own_TPR": f"{clf_tpr:.4f}",
        "classifier_own_FPR": f"{clf_fpr:.4f}",
        "classifier_type": classifier_type,
        "tpr_fpr_source": tpr_fpr_source,
        "rectified_C_formula": "C(G) = (CM - FPR) / (TPR - FPR)",
    }

    _write_csv(run_dir / "property_samples.csv", sample_rows)
    _write_csv(run_dir / "property_metrics.csv", [metrics_row])
    _write_csv(
        run_dir / "failure_cases.csv",
        [row for row in sample_rows if not bool(row["is_valid"])],
    )
    _write_jsonl(run_dir / "llm_io.jsonl", llm_rows)

    # 训练集摘要
    clf_summary: dict[str, object] = {
        "classifier_type": clf_stats.get("classifier_type", classifier_type),
        "data_source": clf_stats.get("data_source", classifier_type),
        "train_total": clf_stats["train_size"],
        "train_positive": clf_stats["train_positive"],
        "train_negative": clf_stats["train_negative"],
        "train_unique_canonical": len(train_canonical),
        "classifier_own_TPR": f"{clf_tpr:.4f}",
        "classifier_own_FPR": f"{clf_fpr:.4f}",
        "rectified_C_TPR": f"{tpr:.4f}",
        "rectified_C_FPR": f"{fpr:.4f}",
        "tpr_fpr_source": tpr_fpr_source,
        "eval_set": clf_stats.get("eval_set", "unknown"),
    }
    if "accuracy" in clf_stats:
        clf_summary["accuracy"] = f"{clf_stats['accuracy']:.4f}"
    if "test_rocauc" in clf_stats:
        clf_summary["test_rocauc"] = f"{clf_stats['test_rocauc']:.4f}"
    if "confusion_matrix" in clf_stats:
        clf_summary["confusion_matrix"] = str(clf_stats["confusion_matrix"])
    if "confusion_matrix_full" in clf_stats:
        clf_summary["confusion_matrix_full"] = str(clf_stats["confusion_matrix_full"])

    _write_csv(run_dir / "baseline_train_summary.csv", [clf_summary])
    _write_run_log(run_dir / "run.log", metrics_row, clf_summary, strategy, model, temperature)
    return 0, run_dir


# ---------------------------------------------------------------------------
# 输出写入工具
# ---------------------------------------------------------------------------

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


def _write_run_log(
    path: Path,
    metrics_row: dict[str, object],
    clf_summary: dict[str, object],
    strategy: str,
    model: str,
    temperature: float,
) -> None:
    lines = [
        "Stage4 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：{model}",
        f"- 策略：{strategy}",
        f"- 温度：{temperature}",
        "- 阶段范围：Property-based（MolHIV）",
        f"- 分类器类型：{clf_summary.get('classifier_type', 'unknown')}",
        f"- 训练数据来源：{clf_summary.get('data_source', 'unknown')}",
        f"- 训练集大小：{clf_summary.get('train_total', 'N/A')} "
        f"(正: {clf_summary.get('train_positive', 'N/A')}, "
        f"负: {clf_summary.get('train_negative', 'N/A')})",
        f"- TPR/FPR 评估集：{clf_summary.get('eval_set', 'unknown')}",
        f"- TPR/FPR 来源：{clf_summary.get('tpr_fpr_source', 'classifier')}",
        f"- 分类器自身 TPR/FPR：{clf_summary.get('classifier_own_TPR', 'N/A')} / {clf_summary.get('classifier_own_FPR', 'N/A')}",
        f"- rectified_C 使用 TPR/FPR：{clf_summary.get('rectified_C_TPR', 'N/A')} / {clf_summary.get('rectified_C_FPR', 'N/A')}",
        "- 指标口径（对齐论文）：",
        "  - CM = 有效样本中分类器正类概率均值",
        "  - C_raw = 有效样本中分类器判为正类的比例",
        "  - rectified_C = (CM - FPR) / (TPR - FPR)  [论文公式]",
        "  - Unique = unique_valid_count / valid_count",
        "  - Novel = novel_unique_count / unique_valid_count（相对训练集）",
        f"  - rectified_C 用 TPR = {metrics_row['classifier_TPR']}",
        f"  - rectified_C 用 FPR = {metrics_row['classifier_FPR']}",
        "",
        "结果摘要：",
        f"- total={metrics_row['total_samples']}, valid={metrics_row['valid_count']}, valid_rate={metrics_row['valid_rate']}",
        f"- unique_rate={metrics_row['unique_rate']}, novel_rate={metrics_row['novel_rate']}",
        f"- CM={metrics_row['CM']}, C_raw={metrics_row['C_raw']}, rectified_C={metrics_row['rectified_C']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage4 Property-based MolHIV 评测（论文对齐版 v3）")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=None, help="运行 ID，默认自动生成描述性名称")
    parser.add_argument("--model", default="mock-mol-stage4")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--strategy", default="few_shot",
                        choices=["few_shot", "few_shot_cot"])
    parser.add_argument("--num-generate", type=int, default=100,
                        help="生成分子数（论文为 100+，v3 默认 100）")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai"])
    parser.add_argument("--classifier", default="proxy", choices=["proxy", "ogbg", "gin"],
                        help="分类器类型: proxy (24样本LR) / ogbg (MolHIV全量LR) / gin (GIN网络)")
    parser.add_argument("--paper-tpr-fpr", action="store_true", default=False,
                        help="使用论文 Table 10 的 TPR=0.5615/FPR=0.1045 计算 rectified_C")
    parser.add_argument("--data-root", default="data",
                        help="OGBG-MolHIV 数据目录 (仅 --classifier ogbg/gin)")
    args = parser.parse_args(argv)

    # 自动生成描述性 run_id
    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        parts = [
            "stage4", model_short, args.strategy, args.classifier,
        ]
        if args.paper_tpr_fpr:
            parts.append("paper-tpr")
        parts.append(f"{args.num_generate}g")
        parts.append(ts)
        args.run_id = "_".join(parts)

    prov = None
    if args.provider == "openai":
        from llm4graphgen.providers import OpenAIProvider
        prov = OpenAIProvider()

    _, run_dir = run_stage4(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
        provider=prov,
        strategy=args.strategy,
        num_generate=args.num_generate,
        classifier_type=args.classifier,
        data_root=args.data_root,
        paper_tpr_fpr=args.paper_tpr_fpr,
    )
    print(f"Stage4 运行完成：{run_dir.as_posix()}")
    print("结果文件：property_metrics.csv / property_samples.csv / llm_io.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
