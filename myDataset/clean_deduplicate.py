"""
语义去重模块 (Semantic Deduplication)

核心思路 (两阶段级联):
  1. instruction 级: 对所有 instruction 做 embedding → FAISS 索引
     每条搜索 TOP-K, 保留 instruction_sim > threshold_inst 的候选对
  2. output 级: 对候选对计算 output embedding 相似度
     若 output_sim > threshold_out → 并查集归入同一组
  3. 每组内质量评分 → 保留最优一条, 删除其余

使用方式:
    from clean_deduplicate import semantic_dedup
    deduped, stats = semantic_dedup(dataset)

依赖:
    pip install sentence-transformers faiss-cpu
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
        score += 8.0
    if "print(" in output or "return " in output:
        score += 5.0

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

    logger.info(f"加载 embedding 模型: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"计算 {len(texts)} 条文本的 embedding ...")
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2 归一化 → FAISS IP = cosine
        convert_to_numpy=True,
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray):
    """构建 FAISS 内积索引 (归一化后等价于 cosine 相似度)"""
    import faiss

    dim = embeddings.shape[1]
    logger.info(f"构建 FAISS 索引 (dim={dim}, n={len(embeddings)})")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


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
    """
    语义去重主函数。

    Parameters
    ----------
    dataset : List[dict]
        每条有 ``messages`` 字段: [{"role":"user","content":...}, {"role":"assistant","content":...}]
    inst_sim_threshold : float
        instruction 相似度阈值, 高于此值才进入 output 级比对
    output_sim_threshold : float
        output 相似度阈值, 高于此值才归入同一组
    top_k : int
        FAISS 搜索的候选数 (实际返回 top_k+1, 包含自身)
    model_name : str
        Sentence-Transformer 模型名
    batch_size : int
        Embedding 计算的批次大小
    device : str
        "cpu" 或 "cuda"

    Returns
    -------
    (deduped_dataset, stats)
        deduped_dataset : 去重后的数据
        stats           : 去重统计信息
    """
    n = len(dataset)
    logger.info("=" * 56)
    logger.info(f"语义去重开始 | 输入: {n} 条")
    logger.info(f"  阈值: inst≥{inst_sim_threshold}, output≥{output_sim_threshold}")
    logger.info(f"  TOP-K: {top_k}  |  模型: {model_name}")
    logger.info("=" * 56)

    if n == 0:
        return [], {"total": 0, "removed": 0, "kept": 0, "dup_groups": 0}

    # ---- Step 1: 提取文本 ----
    instructions = [item["messages"][0]["content"] for item in dataset]
    outputs = [item["messages"][1]["content"] for item in dataset]

    # ---- Step 2: instruction embedding → FAISS 索引 → TOP-K 搜索 ----
    inst_emb = compute_embeddings(instructions, model_name, batch_size, device)
    index = build_faiss_index(inst_emb)

    k = min(top_k + 1, n)               # +1 因为首位是自身
    top_scores, top_indices = index.search(inst_emb, k)

    # ---- Step 3: 筛选 instruction 相似候选对 ----
    candidate_pairs = []                # [(i, j, inst_sim), ...]
    checked = set()
    for i in range(n):
        for pos in range(1, k):         # 跳过自身 (pos=0)
            j = int(top_indices[i][pos])
            sim = float(top_scores[i][pos])

            if j <= i:                  # 避免重复对; 同时排除自身
                continue
            if sim < inst_sim_threshold:
                continue

            pair = (i, j)
            if pair in checked:
                continue
            checked.add(pair)
            candidate_pairs.append((i, j, sim))

    logger.info(f"  → instruction 候选对: {len(candidate_pairs)}")

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

        logger.info(f"  → instruction + output 均匹配: {matched} 对")
    else:
        logger.info(f"  → 无候选对, 跳过 output 对比")

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

    logger.info("=" * 56)
    logger.info(f"语义去重完成: {n} → {len(deduped)}  (去除 {len(removed_indices)} 条)")
    if dup_groups:
        logger.info(f"  共 {dup_groups} 个重复组被合并")
    logger.info("=" * 56)

    return deduped, stats
