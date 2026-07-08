"""
流程:
  原始数据集
    │
    ├─ 1. basic_clean()      — 空值/空字符/换行符清理
    ├─ 2. length_filter()    — 基于单词数的长度过滤
    ├─ 3. exact_dedup()      — 精确去重 (prompt + output 完全一致)
    ├─ 4. semantic_dedup()   — 语义去重 (instruction + output 双阈值)
    └─ 5. 保存为 JSONL 文件

"""

import os
import json
import time
import logging
from typing import List, Optional

from dataset import MyDataset
from clean_deduplicate import semantic_dedup

# 长度过滤阈值 
MIN_PROMPT_WORDS = 5
MIN_OUTPUT_WORDS = 30
MAX_OUTPUT_WORDS = 4096

# 语义去重阈值
INST_SIM_THRESHOLD = 0.80       # instruction 相似度门槛
OUTPUT_SIM_THRESHOLD = 0.80     # output 相似度门槛
TOP_K = 15                      # 每个样本搜索 TOP-K 候选

# 输出路径
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def basic_clean(example: dict) -> Optional[dict]:
    """基础清洗: 空值 / 空字符串 / 非法字符 / 换行符统一"""
    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]

    if user is None or assistant is None:
        return None

    user = user.strip()
    assistant = assistant.strip()

    if len(user) == 0 or len(assistant) == 0:
        return None

    # 删除空字符
    user = user.replace("\x00", "")
    assistant = assistant.replace("\x00", "")

    # 统一换行
    user = user.replace("\r\n", "\n")
    assistant = assistant.replace("\r\n", "\n")

    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def length_filter(example: dict) -> Optional[dict]:
    """长度过滤: 过短 / 过长的样本丢弃"""
    prompt = example["messages"][0]["content"]
    answer = example["messages"][1]["content"]

    prompt_words = len(prompt.split())
    answer_words = len(answer.split())

    if prompt_words < MIN_PROMPT_WORDS:
        return None
    if answer_words < MIN_OUTPUT_WORDS:
        return None
    if answer_words > MAX_OUTPUT_WORDS:
        return None

    return example



def run_pipeline(use_semantic_dedup: bool = True) -> List[dict]:
    """执行完整清洗管线, 返回清洗后的数据"""

    # ---- Step 0: 加载 ----

    t0 = time.time()

    dataset = MyDataset()
    total_raw = len(dataset)

    # ---- 逐条清洗 (链表式处理, 节省内存) ----
    passed_clean = 0
    passed_length = 0
    filtered_data = []
    seen_exact = set()          # 精确去重 的 set

    for idx in range(total_raw):
        raw = dataset[idx]

        # 1. 基础清洗
        cleaned = basic_clean(raw)
        if cleaned is None:
            continue
        passed_clean += 1

        # 2. 长度过滤
        length_ok = length_filter(cleaned)
        if length_ok is None:
            continue
        passed_length += 1

        # 3. 精确去重
        key = (
            length_ok["messages"][0]["content"],
            length_ok["messages"][1]["content"],
        )
        if key in seen_exact:
            continue
        seen_exact.add(key)

        filtered_data.append(length_ok)


    # ---- 4. 语义去重 ----
    if use_semantic_dedup and len(filtered_data) > 0:
        deduped, stats = semantic_dedup(
            filtered_data,
            inst_sim_threshold=INST_SIM_THRESHOLD,
            output_sim_threshold=OUTPUT_SIM_THRESHOLD,
            top_k=TOP_K,
        )
        final_data = deduped
    else:
        final_data = filtered_data
        stats = {"kept": len(final_data), "removed": 0, "dup_groups": 0}


    return final_data


def save_jsonl(data: List[dict], filepath: str):
    """将数据保存为 JSONL 格式 (每行一个 JSON 对象)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[dict]:
    """从 JSONL 文件加载数据"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 执行清洗管线
    clean_data = run_pipeline(use_semantic_dedup=True)

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "cleaned_data.jsonl")
    save_jsonl(clean_data, output_path)

