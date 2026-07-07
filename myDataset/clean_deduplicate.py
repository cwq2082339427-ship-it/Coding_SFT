

seen = set()
clean_data = []
new_data = []
def remove(exapmle,seen):

    for item in clean_data:

        key = (
            item["messages"][0]["content"],
            item["messages"][1]["content"]
        )

        if key in seen:
            continue

        seen.add(key)
        new_data.append(item)

    clean_data = new_data


###语义去重，对相似工作，例如都采用python写一段DFS，只需要保留一个即可，过多的相似内容会导致数据bias，影响训练结果