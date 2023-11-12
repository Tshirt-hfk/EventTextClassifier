# coding=utf-8
from collections import defaultdict
import json
import random

random.seed(555)

input_files = ["./data/data_v2.json"]
train_file = "./data/train.json"
dev_file = "./data/dev.json"

dev_num_per_label = 20
max_text_len_limit = 256


total_data = defaultdict(list)
train_data = []
dev_data = []
max_len = 0
for input_file in input_files:
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            example = json.loads(line.strip())
            if len(example['text']) > max_text_len_limit:
                continue
            max_len = max(max_len, len(example['text']))
            total_data[tuple(example['label'])].append(line)

print("max length:", max_len)
print(total_data.keys())

total_num = 0
for k, v in total_data.items():
    total_num += len(v)
print("total data: ", total_num)

for k, v in total_data.items():
    random.shuffle(v)
    if len(v) > 100:
        train_data += v[:-dev_num_per_label]
        dev_data += v[-dev_num_per_label:]
        print(k, len(v[:-dev_num_per_label]), len(v[-dev_num_per_label:]))
    else:
        train_data += v
        print(k, len(v), 0)
    

with open(train_file, "w", encoding="utf-8") as f_train, open(dev_file, "w", encoding="utf-8") as f_dev:
    for line in train_data:
        f_train.write(line)
    for line in dev_data:
        f_dev.write(line)


