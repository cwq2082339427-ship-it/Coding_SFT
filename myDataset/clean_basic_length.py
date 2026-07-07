import os
import json
from dataset import MyDataset
#去重，防止Distribution Bias，让模型输出产生偏向性。
#DEITA论文说明，数据的多样性和质量比数据量更重要
import re

def basic_clean(example):

    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]

    if user is None or assistant is None:
        return None

    user = user.strip()
    assistant = assistant.strip()

    if len(user)==0 or len(assistant)==0:
        return None

    # 删除非法字符
    user = user.replace("\x00","")
    assistant = assistant.replace("\x00","")

    # 统一换行
    user = user.replace("\r\n","\n")
    assistant = assistant.replace("\r\n","\n")

    return {
        "messages":[
            {
                "role":"user",
                "content":user
            },
            {
                "role":"assistant",
                "content":assistant
            }
        ]
    }

#长度控制，清除过长或者过短的数据(影响训练结果的数据分布)
MIN_PROMPT = 5
MIN_OUTPUT = 30
MAX_OUTPUT = 4096

def length_filter(example):

    prompt = example["messages"][0]["content"]
    answer = example["messages"][1]["content"]

    if len(prompt.split()) < MIN_PROMPT:
        return None

    if len(answer.split()) < MIN_OUTPUT:
        return None

    if len(answer.split()) > MAX_OUTPUT:
        return None

    return example


def filter_dataset(dataset):
    seen = set()
    filtered_data = []

    for idx, data in enumerate(dataset):
        # 1. 基本清洗
        cleaned = basic_clean(data)
        if cleaned is None:
            continue

        # 2. 长度过滤
        filtered = length_filter(cleaned)
        if filtered is None:
            continue

        # 3. 去重
        key = (filtered["messages"][0]["content"], filtered["messages"][1]["content"])
        if key in seen:
            continue  
        seen.add(key)
        filtered_data.append(filtered)  # 保留

    return filtered_data

mydataset = MyDataset()
new_data = filter_dataset(mydataset)