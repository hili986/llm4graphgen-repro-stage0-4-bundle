"""OGBG-MolHIV 分类器模块 — 对齐论文 Stage4 Property-based 评测。

论文使用 GNN (GIN/GCN) 在完整 OGBG-MolHIV 数据集 (~41,127 分子) 上训练分类器，
用于计算 C_M 和 rectified_C 指标。

本模块提供两种分类器：
1. ogbg: 在完整 OGBG-MolHIV 数据集上训练 LogisticRegression (Morgan fingerprint)
2. proxy: 在 24 个代理样本上训练（向后兼容 v2）

使用 ogbg 分类器可获得与论文可比的 TPR/FPR，消除过拟合问题。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Morgan fingerprint 工具
# ---------------------------------------------------------------------------

def fingerprint_array(smiles: str, fp_size: int = 1024) -> np.ndarray | None:
    """计算 Morgan 指纹向量。返回 None 表示 SMILES 无效。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp_size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ---------------------------------------------------------------------------
# OGBG-MolHIV 数据加载
# ---------------------------------------------------------------------------

def _load_ogbg_molhiv(data_root: str = "data") -> tuple[list[str], list[int], dict[str, list[int]]]:
    """加载 OGBG-MolHIV 数据集。

    Returns:
        (smiles_list, labels, split_dict)
        split_dict 包含 'train', 'valid', 'test' 三组索引。
    """
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError:
        raise ImportError(
            "需要安装 ogb 包才能使用 OGBG-MolHIV 分类器。\n"
            "请运行: pip install ogb\n"
            "首次运行会自动下载数据集 (~3MB)。"
        )

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=data_root)
    split_idx = dataset.get_idx_split()

    # 获取 SMILES — 从 OGB 的 mapping 文件中读取
    import pandas as pd
    mapping_dir = Path(data_root) / "ogbg_molhiv" / "mapping"
    mapping_file = mapping_dir / "mol.csv.gz"

    if not mapping_file.exists():
        # 尝试备选路径
        alt_paths = [
            Path(data_root) / "ogbg-molhiv" / "mapping" / "mol.csv.gz",
            Path(data_root) / "ogbg_molhiv" / "raw" / "mol.csv.gz",
        ]
        for alt in alt_paths:
            if alt.exists():
                mapping_file = alt
                break

    if mapping_file.exists():
        df = pd.read_csv(mapping_file)
        smiles_list = df["smiles"].tolist()
    else:
        # 如果找不到 mapping 文件，尝试从 graph 对象重建 SMILES
        raise FileNotFoundError(
            f"找不到 OGBG-MolHIV SMILES mapping 文件。\n"
            f"尝试过的路径: {mapping_file}\n"
            f"请确保 ogb 数据集正确下载。"
        )

    labels = dataset.labels.squeeze().tolist()

    split_dict = {
        "train": split_idx["train"].tolist(),
        "valid": split_idx["valid"].tolist(),
        "test": split_idx["test"].tolist(),
    }

    return smiles_list, labels, split_dict


def _load_ogbg_smiles_only(data_root: str = "data") -> tuple[list[str], list[int], dict[str, list[int]]]:
    """直接从 OGB 下载目录读取 SMILES + labels，不依赖完整 ogb 包。

    这是一个轻量回退方案：如果已经下载过数据集，可以不装 ogb 包直接读取。
    """
    import gzip

    base = Path(data_root)
    candidates = [
        base / "ogbg_molhiv",
        base / "ogbg-molhiv",
    ]

    csv_gz = None
    for cand in candidates:
        f = cand / "mapping" / "mol.csv.gz"
        if f.exists():
            csv_gz = f
            break

    if csv_gz is None:
        raise FileNotFoundError(
            "找不到已下载的 OGBG-MolHIV 数据。\n"
            "请先安装 ogb 包并运行一次以下载数据：\n"
            "  pip install ogb pandas\n"
            "  python -c \"from ogb.graphproppred import PygGraphPropPredDataset; "
            "PygGraphPropPredDataset(name='ogbg-molhiv', root='data/')\""
        )

    # 读取 SMILES + label
    import csv as csv_mod
    with gzip.open(csv_gz, "rt", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        smiles_list = []
        labels = []
        for row in reader:
            smiles_list.append(row["smiles"])
            labels.append(int(row.get("HIV_active", row.get("label", 0))))

    # 读取 split
    split_dir = csv_gz.parent.parent / "split" / "scaffold"
    split_dict: dict[str, list[int]] = {}
    for split_name in ["train", "valid", "test"]:
        split_file = split_dir / f"{split_name}.csv.gz"
        if split_file.exists():
            with gzip.open(split_file, "rt") as f:
                reader = csv_mod.reader(f)
                next(reader, None)  # skip header
                split_dict[split_name] = [int(row[0]) for row in reader if row]
        else:
            split_dict[split_name] = []

    # 如果 split 文件不存在，用 scaffold split 比例近似
    if not split_dict["train"]:
        n = len(smiles_list)
        indices = list(range(n))
        np.random.RandomState(42).shuffle(indices)
        n_train = int(0.8 * n)
        n_valid = int(0.1 * n)
        split_dict["train"] = indices[:n_train]
        split_dict["valid"] = indices[n_train:n_train + n_valid]
        split_dict["test"] = indices[n_train + n_valid:]

    return smiles_list, labels, split_dict


# ---------------------------------------------------------------------------
# 分类器训练
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache")


def _cache_path(data_root: str) -> Path:
    """计算缓存文件路径。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(data_root.encode()).hexdigest()[:8]
    return _CACHE_DIR / f"molhiv_clf_{key}.pkl"


def train_ogbg_classifier(
    data_root: str = "data",
    fp_size: int = 1024,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[LogisticRegression, set[str], dict[str, Any]]:
    """在完整 OGBG-MolHIV 上训练分类器。

    Returns:
        (clf, train_smiles_canonical, stats)
        stats 包含 tpr, fpr, train_size 等信息。
    """
    cache_file = _cache_path(data_root)
    if use_cache and cache_file.exists():
        if verbose:
            print(f"  [MolHIV] 从缓存加载分类器: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["clf"], cached["train_canonical"], cached["stats"]

    if verbose:
        print("  [MolHIV] 加载 OGBG-MolHIV 数据集...")

    # 尝试加载数据
    try:
        smiles_list, labels, split_dict = _load_ogbg_molhiv(data_root)
    except (ImportError, FileNotFoundError):
        try:
            smiles_list, labels, split_dict = _load_ogbg_smiles_only(data_root)
        except FileNotFoundError as e:
            raise RuntimeError(str(e))

    train_idx = split_dict["train"]
    test_idx = split_dict["test"]
    if not test_idx:
        test_idx = split_dict.get("valid", [])

    if verbose:
        print(f"  [MolHIV] 数据集大小: {len(smiles_list)} 分子, "
              f"训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

    # 构建训练集指纹
    if verbose:
        print("  [MolHIV] 计算 Morgan 指纹...")

    train_fps = []
    train_labels = []
    train_smiles_canonical: set[str] = set()
    failed_count = 0

    for idx in train_idx:
        smi = smiles_list[idx]
        fp = fingerprint_array(smi, fp_size)
        if fp is not None:
            train_fps.append(fp)
            train_labels.append(labels[idx])
            mol = Chem.MolFromSmiles(smi)
            if mol:
                train_smiles_canonical.add(
                    Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                )
        else:
            failed_count += 1

    X_train = np.vstack(train_fps)
    y_train = np.array(train_labels, dtype=np.int64)

    if verbose:
        print(f"  [MolHIV] 训练集: {len(train_fps)} 有效分子 ({failed_count} 解析失败)")
        print(f"  [MolHIV] 正类比例: {y_train.mean():.4f}")

    # 构建测试集指纹
    test_fps = []
    test_labels = []
    for idx in test_idx:
        smi = smiles_list[idx]
        fp = fingerprint_array(smi, fp_size)
        if fp is not None:
            test_fps.append(fp)
            test_labels.append(labels[idx])

    X_test = np.vstack(test_fps) if test_fps else np.zeros((0, fp_size))
    y_test = np.array(test_labels, dtype=np.int64) if test_labels else np.array([], dtype=np.int64)

    # 训练分类器
    if verbose:
        print("  [MolHIV] 训练 LogisticRegression 分类器...")

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=0,
        class_weight="balanced",  # 处理类别不平衡
        C=1.0,
    )
    clf.fit(X_train, y_train)

    # 在测试集上评估 — 获取真实 TPR/FPR
    if len(y_test) > 0:
        y_pred_test = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # 处理退化情况
            tn, fp, fn, tp = 0, 0, 0, 0
            if cm.shape[0] >= 1:
                tn = int(cm[0, 0])
            if cm.shape[0] >= 2 and cm.shape[1] >= 2:
                fp, fn, tp = int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        if verbose:
            print(f"  [MolHIV] 测试集性能: Acc={accuracy:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
            print(f"  [MolHIV] 混淆矩阵: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    else:
        # 回退到训练集评估
        y_pred_train = clf.predict(X_train)
        cm = confusion_matrix(y_train, y_pred_train)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if verbose:
            print(f"  [MolHIV] 训练集性能 (无测试集): Acc={accuracy:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")

    stats: dict[str, Any] = {
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy,
        "train_size": len(train_fps),
        "train_positive": int(y_train.sum()),
        "train_negative": int(len(y_train) - y_train.sum()),
        "test_size": len(test_fps),
        "test_positive": int(y_test.sum()) if len(y_test) > 0 else 0,
        "test_negative": int(len(y_test) - y_test.sum()) if len(y_test) > 0 else 0,
        "confusion_matrix": cm.tolist() if len(y_test) > 0 else [],
        "classifier_type": "LogisticRegression(balanced)",
        "fp_type": f"Morgan(radius=2, size={fp_size})",
        "data_source": "ogbg-molhiv",
        "eval_set": "test" if len(y_test) > 0 else "train",
    }

    # 缓存
    if use_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "clf": clf,
                    "train_canonical": train_smiles_canonical,
                    "stats": stats,
                }, f)
            if verbose:
                print(f"  [MolHIV] 分类器已缓存至: {cache_file}")
        except Exception as e:
            if verbose:
                print(f"  [MolHIV] 缓存失败 (不影响运行): {e}")

    return clf, train_smiles_canonical, stats
