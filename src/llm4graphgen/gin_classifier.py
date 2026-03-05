"""GIN (Graph Isomorphism Network) 分类器 — 对齐论文 Stage4 Property-based 评测。

论文 (arXiv:2403.14358) 使用 "a GNN classifier trained on the OGBG-MolHIV dataset"。
本模块实现 OGB 官方 GIN baseline，在 OGBG-MolHIV 上训练，用于计算 C_M 和 rectified_C。

OGB 官方 GIN 默认超参：
- num_layer: 5
- emb_dim: 300
- dropout: 0.5
- batch_size: 32
- epochs: 100
- lr: 0.001 (Adam)
- pooling: mean
- JK (Jump Knowledge): last
- loss: BCEWithLogitsLoss

依赖：torch, torch_geometric, ogb
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 论文 Table 10 TPR/FPR 常量
# ---------------------------------------------------------------------------

PAPER_TPR = 0.5615  # 810 / (810 + 633)
PAPER_FPR = 0.1045  # 4145 / (4145 + 35539)
PAPER_TP = 810
PAPER_FN = 633
PAPER_FP = 4145
PAPER_TN = 35539


# ---------------------------------------------------------------------------
# GIN 模型定义
# ---------------------------------------------------------------------------

def _build_gin_model(num_tasks: int = 1, num_layer: int = 5, emb_dim: int = 300,
                     drop_ratio: float = 0.5):
    """构建 OGB 标准 GIN 模型。

    遵循 OGB 官方实现:
    https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINEConv, global_mean_pool
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

    class GINConvLayer(nn.Module):
        """单层 GIN 卷积（带 bond 特征）。"""

        def __init__(self, emb_dim: int):
            super().__init__()
            mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim),
            )
            self.conv = GINEConv(mlp)
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
            self.bn = nn.BatchNorm1d(emb_dim)

        def forward(self, x, edge_index, edge_attr):
            edge_emb = self.bond_encoder(edge_attr)
            return self.bn(F.relu(self.conv(x, edge_index, edge_emb)))

    class GINModel(nn.Module):
        """OGB 标准 GIN 分类模型。"""

        def __init__(self, num_tasks, num_layer, emb_dim, drop_ratio):
            super().__init__()
            self.num_layer = num_layer
            self.drop_ratio = drop_ratio
            self.atom_encoder = AtomEncoder(emb_dim=emb_dim)
            self.convs = nn.ModuleList(
                [GINConvLayer(emb_dim) for _ in range(num_layer)]
            )
            self.graph_pred_linear = nn.Linear(emb_dim, num_tasks)

        def forward(self, batched_data):
            x = self.atom_encoder(batched_data.x)
            edge_index = batched_data.edge_index
            edge_attr = batched_data.edge_attr

            for layer in self.convs:
                x = layer(x, edge_index, edge_attr)
                x = F.dropout(x, p=self.drop_ratio, training=self.training)

            # JK = last (使用最后一层的输出)
            x = global_mean_pool(x, batched_data.batch)
            return self.graph_pred_linear(x)

    return GINModel(num_tasks, num_layer, emb_dim, drop_ratio)


# ---------------------------------------------------------------------------
# SMILES → PyG Data 转换
# ---------------------------------------------------------------------------

def smiles_to_pyg_data(smiles: str):
    """将 SMILES 转换为 PyG Data 对象。

    使用 ogb.utils.mol.smiles2graph 获取标准化的原子/键特征，
    然后转为 torch_geometric.data.Data。
    """
    import torch
    from torch_geometric.data import Data
    from ogb.utils.mol import smiles2graph

    graph = smiles2graph(smiles)
    if graph is None:
        return None

    data = Data(
        x=torch.tensor(graph["node_feat"], dtype=torch.long),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.long),
        num_nodes=graph["num_nodes"],
    )
    return data


# ---------------------------------------------------------------------------
# 训练函数
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache")


def _gin_cache_path(data_root: str, num_layer: int, emb_dim: int, epochs: int) -> Path:
    """计算 GIN 模型检查点路径。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(
        f"{data_root}_{num_layer}_{emb_dim}_{epochs}".encode()
    ).hexdigest()[:8]
    return _CACHE_DIR / f"gin_molhiv_{key}.pt"


def train_gin_classifier(
    data_root: str = "data",
    num_layer: int = 5,
    emb_dim: int = 300,
    drop_ratio: float = 0.5,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 0.001,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[Any, set[str], dict[str, Any]]:
    """在 OGBG-MolHIV 上训练 GIN 分类器。

    Returns:
        (model, train_smiles_canonical, stats)
        model: 训练好的 GIN 模型（eval 模式）
        train_smiles_canonical: 训练集 canonical SMILES 集合
        stats: 包含 tpr, fpr, train_size 等信息
    """
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    from rdkit import Chem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_file = _gin_cache_path(data_root, num_layer, emb_dim, epochs)

    # ---- 加载数据集 ----
    # PyTorch 2.6+ 默认 weights_only=True，但 ogb 内部 torch.load 未适配
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(
        *a, **{**kw, "weights_only": kw.get("weights_only", False)}
    )
    if verbose:
        print("  [GIN] 加载 OGBG-MolHIV 数据集...")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=data_root)
    torch.load = _orig_torch_load  # 恢复原始行为
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=batch_size, shuffle=False
    )

    # ---- 收集训练集 SMILES ----
    import pandas as pd
    mapping_dir = Path(data_root) / "ogbg_molhiv" / "mapping"
    mapping_file = mapping_dir / "mol.csv.gz"
    if not mapping_file.exists():
        alt = Path(data_root) / "ogbg-molhiv" / "mapping" / "mol.csv.gz"
        if alt.exists():
            mapping_file = alt

    train_smiles_canonical: set[str] = set()
    if mapping_file.exists():
        df = pd.read_csv(mapping_file)
        all_smiles = df["smiles"].tolist()
        for idx in split_idx["train"].tolist():
            mol = Chem.MolFromSmiles(all_smiles[idx])
            if mol:
                train_smiles_canonical.add(
                    Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                )
        if verbose:
            print(f"  [GIN] 训练集 canonical SMILES: {len(train_smiles_canonical)} 个")

    # ---- 构建模型 ----
    model = _build_gin_model(
        num_tasks=1, num_layer=num_layer, emb_dim=emb_dim, drop_ratio=drop_ratio
    )
    model = model.to(device)

    # ---- 尝试加载缓存 ----
    if use_cache and cache_file.exists():
        if verbose:
            print(f"  [GIN] 从缓存加载模型: {cache_file}")
        checkpoint = torch.load(cache_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        stats = checkpoint["stats"]
        if "train_canonical" in checkpoint:
            train_smiles_canonical = checkpoint["train_canonical"]
        return model, train_smiles_canonical, stats

    # ---- 训练 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    evaluator = Evaluator(name="ogbg-molhiv")

    best_valid_rocauc = 0.0
    best_model_state = None

    if verbose:
        print(f"  [GIN] 开始训练: {epochs} epochs, device={device}")
        print(f"  [GIN] 超参: layers={num_layer}, dim={emb_dim}, "
              f"dropout={drop_ratio}, lr={lr}, batch={batch_size}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            is_labeled = batch.y == batch.y  # 过滤 NaN
            loss = criterion(
                pred[is_labeled].float(),
                batch.y[is_labeled].float(),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)

        # Validate
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            y_true_list = []
            y_pred_list = []
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    y_true_list.append(batch.y.cpu())
                    y_pred_list.append(pred.cpu())

            y_true = torch.cat(y_true_list, dim=0).numpy()
            y_pred = torch.cat(y_pred_list, dim=0).numpy()
            result = evaluator.eval({
                "y_true": y_true,
                "y_pred": y_pred,
            })
            valid_rocauc = result["rocauc"]

            if verbose:
                print(f"  [GIN] Epoch {epoch:3d}/{epochs}: "
                      f"loss={avg_loss:.4f}, valid_rocauc={valid_rocauc:.4f}")

            if valid_rocauc > best_valid_rocauc:
                best_valid_rocauc = valid_rocauc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- 加载最佳模型 ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)
    model.eval()

    if verbose:
        print(f"  [GIN] 训练完成, best valid ROC-AUC = {best_valid_rocauc:.4f}")

    # ---- 在全数据集上评估 (对齐论文 Table 10) ----
    # 论文 Table 10 的混淆矩阵覆盖全部 41,127 个分子
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_y_true = []
    all_y_score = []
    all_y_pred = []

    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            all_y_true.append(batch.y.cpu())
            all_y_score.append(probs.cpu())
            all_y_pred.append(preds.cpu())

    all_y_true = torch.cat(all_y_true, dim=0).squeeze().numpy()
    all_y_score = torch.cat(all_y_score, dim=0).squeeze().numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).squeeze().numpy()

    # 混淆矩阵 (全数据集)
    tp = int(((all_y_true == 1) & (all_y_pred == 1)).sum())
    fn = int(((all_y_true == 1) & (all_y_pred == 0)).sum())
    fp = int(((all_y_true == 0) & (all_y_pred == 1)).sum())
    tn = int(((all_y_true == 0) & (all_y_pred == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    if verbose:
        print(f"  [GIN] 全数据集评估 (对齐论文 Table 10):")
        print(f"  [GIN]   TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"  [GIN]   TPR={tpr:.4f}, FPR={fpr:.4f}, Acc={accuracy:.4f}")
        print(f"  [GIN]   总样本={tp+fn+fp+tn} (论文: 41127)")
        print(f"  [GIN] 论文 Table 10 参考:")
        print(f"  [GIN]   TPR={PAPER_TPR:.4f}, FPR={PAPER_FPR:.4f}")

    # ---- 在测试集上也评估 ROC-AUC ----
    test_y_true = []
    test_y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_y_true.append(batch.y.cpu())
            test_y_pred.append(pred.cpu())

    test_y_true_arr = torch.cat(test_y_true, dim=0).numpy()
    test_y_pred_arr = torch.cat(test_y_pred, dim=0).numpy()
    test_result = evaluator.eval({
        "y_true": test_y_true_arr,
        "y_pred": test_y_pred_arr,
    })
    test_rocauc = test_result["rocauc"]

    if verbose:
        print(f"  [GIN] 测试集 ROC-AUC = {test_rocauc:.4f} "
              f"(OGB baseline: 0.7558 ± 0.0140)")

    stats: dict[str, Any] = {
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy,
        "train_size": len(split_idx["train"]),
        "train_positive": int((all_y_true[split_idx["train"].numpy()] == 1).sum()),
        "train_negative": int((all_y_true[split_idx["train"].numpy()] == 0).sum()),
        "test_rocauc": test_rocauc,
        "best_valid_rocauc": best_valid_rocauc,
        "confusion_matrix_full": [[tn, fp], [fn, tp]],
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "total_samples": tp + fn + fp + tn,
        "classifier_type": f"GIN(layers={num_layer}, dim={emb_dim})",
        "data_source": "ogbg-molhiv",
        "eval_set": "full_dataset",
        "device": str(device),
        "epochs": epochs,
    }

    # ---- 缓存模型 ----
    if use_cache:
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "stats": stats,
                "train_canonical": train_smiles_canonical,
                "config": {
                    "num_layer": num_layer,
                    "emb_dim": emb_dim,
                    "drop_ratio": drop_ratio,
                    "epochs": epochs,
                },
            }, cache_file)
            if verbose:
                print(f"  [GIN] 模型已缓存至: {cache_file}")
        except Exception as e:
            if verbose:
                print(f"  [GIN] 缓存失败 (不影响运行): {e}")

    return model, train_smiles_canonical, stats


# ---------------------------------------------------------------------------
# 推理函数 — 对生成的 SMILES 进行正类概率打分
# ---------------------------------------------------------------------------

def gin_predict_proba(model, smiles_list: list[str], batch_size: int = 64) -> list[float | None]:
    """用训练好的 GIN 对 SMILES 列表进行正类概率预测。

    Args:
        model: 训练好的 GIN 模型 (eval mode)
        smiles_list: SMILES 字符串列表
        batch_size: 推理 batch 大小

    Returns:
        概率列表，无法解析的 SMILES 返回 None
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    device = next(model.parameters()).device

    # 转换 SMILES 为 PyG Data
    data_list = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        try:
            data = smiles_to_pyg_data(smi)
            if data is not None and data.num_nodes > 0:
                data_list.append(data)
                valid_indices.append(i)
        except Exception:
            continue

    # 初始化结果
    results: list[float | None] = [None] * len(smiles_list)

    if not data_list:
        return results

    # 批量推理
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([probs.item()])
            all_probs.extend(probs.tolist())

    # 填入结果
    for idx, prob in zip(valid_indices, all_probs):
        results[idx] = float(prob)

    return results
