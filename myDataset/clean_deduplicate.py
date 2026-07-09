"""
语义去重模块

核心思路 (两阶段级联):
  1. instruction 级: 对所有 instruction 做 embedding → 近似度矩阵
     每条搜索 TOP-K, 保留 instruction_sim > threshold_inst 的候选对
  2. output 级: 对候选对计算 output embedding 相似度
     若 output_sim > threshold_out → 并查集归入同一组
  3. 每组内质量评分 → 保留最优一条, 删除其余

使用方式:
    from clean_deduplicate import semantic_dedup
    deduped, stats = semantic_dedup(dataset)

依赖:
    pip install sentence-transformers

注意:
    - 原本依赖 faiss 进行 TOP-K 搜索, 现改用纯 numpy 批量计算余弦相似度,
      避免 faiss 安装困难的问题。
    - 对于超大数据集 (>5 万条), 改用分块 (block-wise) 相似度计算以控制内存。
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具: 并查集 (Disjoint Set Union)
# ---------------------------------------------------------------------------

class UnionFind:
    """带路径压缩和按秩合并的并查集"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1


# ---------------------------------------------------------------------------
# 质量评分
# ---------------------------------------------------------------------------

def score_item(item: dict) -> float:
    """
    对单条样本进行多维质量评分。

    用于同组内「谁留下」的决策 —— 分数越高代表样本质量越好。

    评分维度:
      - 代码块覆盖率 (``` 代码段)
      - 输出长度合理性
      - 代码质量信号 (import, def, class, print, return)
      - 指令复杂度 (长度)
    """
    instruction = item["messages"][0]["content"]
    output = item["messages"][1]["content"]
    score = 0.0

    # 1. 代码块覆盖率 — 优质代码数据的核心信号
    code_block_count = output.count("```") // 2
    if code_block_count >= 1:
        score += 20.0
    if code_block_count >= 2:
        score += 15.0          # 多段代码通常含更完整的解释

    # 2. 输出长度评分
    output_words = len(output.split())
    if 100 <= output_words <= 2000:
        score += 20.0
    elif 50 <= output_words < 100:
        score += 10.0
    elif 2000 < output_words <= 3000:
        score += 10.0

    # 3. 指令复杂度 — 长 instruction 通常对应复杂任务
    inst_words = len(instruction.split())
    score += min(inst_words / 20.0, 5.0) * 2.0   # max 10

    # 4. 代码质量信号
    if "import " in output:
        score += 10.0
    if "def " in output:
        score += 8.0
    if "class " in output:
        score += 10.0
    if "print(" in output or "return " in output:
        score += 5.0
    if "go" in output:
        score -= 5.0   # 针对一些不规范语言，减少评分。

    # 5. 无代码块的纯文本应答 → 对代码 SFT 价值低
    if code_block_count == 0:
        score -= 15.0

    return score


# ---------------------------------------------------------------------------
# Embedding 工具
# ---------------------------------------------------------------------------

def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cpu",
    show_progress: bool = True,
) -> np.ndarray:
    """
    计算文本列表的 embedding (L2 归一化)。
    返回 shape=(n, dim) 的 numpy 数组。
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2 归一化 → 余弦相似度 = 点积
        convert_to_numpy=True,
    )
    return embeddings



def _find_top_k_fast(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算余弦相似度矩阵并获取 TOP-K 索引和分数。

    使用整块矩阵乘法 (n × n), 适用于 N ≤ 50000 的中小数据集。
    返回:
        top_scores:  shape (n, k)
        top_indices: shape (n, k)    (均不含自身)
    """
    n = embeddings.shape[0]
    # 余弦相似度 = 点积 (已 L2 归一化)
    sim = np.dot(embeddings, embeddings.T)          # (n, n)

    # 将对角线 (自身相似度=1) 置为 -inf, 避免排第一
    np.fill_diagonal(sim, -np.inf)

    # 取每行最大的 k 个
    # 如果 n-1 < k, 修正 k 为 n-1
    actual_k = min(k, n - 1)
    top_indices = np.argpartition(-sim, actual_k - 1, axis=1)[:, :actual_k]

    # argpartition 不保证排序, 再按分数排序这 actual_k 个
    row_idx = np.arange(n)[:, None]
    top_scores_before = sim[row_idx, top_indices]
    # 按分数降序排列
    sort_order = np.argsort(-top_scores_before, axis=1)
    top_indices = top_indices[row_idx, sort_order]
    top_scores = top_scores_before[row_idx, sort_order]

    # 如果要求的 k > actual_k, 用 -1 填充 (但不影响后续过滤,
    # 因为 score=-inf 不会超过阈值)
    if actual_k < k:
        pad = np.full((n, k - actual_k), -1, dtype=top_indices.dtype)
        top_indices = np.concatenate([top_indices, pad], axis=1)
        pad_scores = np.full((n, k - actual_k), -np.inf, dtype=top_scores.dtype)
        top_scores = np.concatenate([top_scores, pad_scores], axis=1)

    return top_scores, top_indices


def _find_top_k_blockwise(
    embeddings: np.ndarray,
    k: int,
    block_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分块计算 TOP-K, 避免一次性构建 n×n 矩阵 (内存友好)。

    对于超大 N (>50000) 逐块计算, 每块只需 block_size × n 的临时矩阵。
    """
    n = embeddings.shape[0]
    actual_k = min(k, n - 1)

    # L2 归一化 (虽然外部已归一化, 再确保一次也无妨)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_normalized = embeddings / norms

    top_scores = np.full((n, actual_k), -np.inf, dtype=np.float32)
    top_indices = np.full((n, actual_k), -1, dtype=np.int64)

    # 分块处理: 每次处理 block_size 个 query
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = emb_normalized[start:end]                     # (B, D)
        sim_block = np.dot(block, emb_normalized.T)           # (B, n)

        # 将对角线上自身相似度置为 -inf
        for local_i in range(end - start):
            global_i = start + local_i
            sim_block[local_i, global_i] = -np.inf

        # 取 TOP-K
        local_top_indices = np.argpartition(-sim_block, actual_k - 1, axis=1)[:, :actual_k]
        row_local = np.arange(end - start)[:, None]
        top_scores_local = sim_block[row_local, local_top_indices]

        # 排序
        sort_order = np.argsort(-top_scores_local, axis=1)
        local_top_indices = local_top_indices[row_local, sort_order]
        top_scores_local = top_scores_local[row_local, sort_order]

        top_indices[start:end] = local_top_indices
        top_scores[start:end] = top_scores_local

    # 如果要求的 k > actual_k, 填充
    if actual_k < k:
        pad_idx = np.full((n, k - actual_k), -1, dtype=np.int64)
        top_indices = np.concatenate([top_indices, pad_idx], axis=1)
        pad_scores = np.full((n, k - actual_k), -np.inf, dtype=np.float32)
        top_scores = np.concatenate([top_scores, pad_scores], axis=1)

    return top_scores, top_indices


def find_top_k(
    embeddings: np.ndarray,
    k: int,
    block_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一入口: 根据数据规模自动选择整块 / 分块策略。

    参数:
        embeddings: (n, dim) L2 归一化后的 embedding
        k: 每个 query 返回的最近邻数
        block_size: 分块大小, 仅对超大 N 生效

    返回:
        top_scores:  shape (n, k)
        top_indices: shape (n, k)   (均不含自身)
                      未达到 k 个时, index = -1, score = -inf
    """
    n = embeddings.shape[0]
    if n <= 50000:
        return _find_top_k_fast(embeddings, k)
    else:
        logger.info(f"数据集较大 (n={n}), 采用分块 TOP-K 搜索, block_size={block_size}")
        return _find_top_k_blockwise(embeddings, k, block_size=block_size)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def semantic_dedup(
    dataset: List[dict],
    inst_sim_threshold: float = 0.75,
    output_sim_threshold: float = 0.80,
    top_k: int = 15,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[List[dict], dict]:

    n = len(dataset)

    if n == 0:
        return [], {"total": 0, "removed": 0, "kept": 0, "dup_groups": 0}

    # ---- Step 1: 提取文本 ----
    instructions = [item["messages"][0]["content"] for item in dataset]
    outputs = [item["messages"][1]["content"] for item in dataset]

    # ---- Step 2: instruction embedding → TOP-K 搜索 ----
    inst_emb = compute_embeddings(instructions, model_name, batch_size, device)

    k = min(top_k + 1, n)             # +1 因为首位是自身; find_top_k 会跳过自身
    top_scores, top_indices = find_top_k(inst_emb, k)

    # ---- Step 3: 筛选 instruction 相似候选对 ----
    candidate_pairs = []              # [(i, j, inst_sim), ...]
    checked = set()
    for i in range(n):
        for pos in range(k):
            j = int(top_indices[i][pos])
            sim = float(top_scores[i][pos])

            if j < 0:                 # 填充的无效位
                continue
            if j <= i:                # 避免重复对
                continue
            if sim < inst_sim_threshold:
                continue

            pair = (i, j)
            if pair in checked:
                continue
            checked.add(pair)
            candidate_pairs.append((i, j, sim))

    # ---- Step 4: output embedding 相似度 → 并查集归组 ----
    uf = UnionFind(n)

    if candidate_pairs:
        out_emb = compute_embeddings(outputs, model_name, batch_size, device)

        matched = 0
        for i, j, inst_sim in candidate_pairs:
            out_sim = float(np.dot(out_emb[i], out_emb[j]))   # 已归一化 → cosine
            if out_sim >= output_sim_threshold:
                uf.union(i, j)
                matched += 1

    # ---- Step 5: 归组 → 组内评分 → 保留最优 ----
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    dup_groups = 0
    removed_indices: set = set()

    for root, members in groups.items():
        if len(members) == 1:
            continue          # 独苗, 无需处理

        dup_groups += 1
        # 组内按质量评分降序排列
        scored = [(score_item(dataset[idx]), idx) for idx in members]
        scored.sort(key=lambda x: x[0], reverse=True)

        # 保留最高分, 其余标记删除
        for _, idx in scored[1:]:
            removed_indices.add(idx)

    deduped = [dataset[i] for i in range(n) if i not in removed_indices]

    stats = {
        "total": n,
        "total_groups": len(groups),
        "dup_groups": dup_groups,
        "removed": len(removed_indices),
        "kept": len(deduped),
        "inst_sim_threshold": inst_sim_threshold,
        "output_sim_threshold": output_sim_threshold,
        "top_k": top_k,
        "embedding_model": model_name,
    }

    return deduped, stats
