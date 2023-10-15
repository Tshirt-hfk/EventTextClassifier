# coding:utf-8
from collections import defaultdict
import os, json

def parse_data(file_path, out_data):
    with open(file_path, "r", encoding='utf8') as f:
        data_list = json.load(f)
        for data in data_list:
            if len(data['event_list']) == 1:
                label = data['event_list'][0]['class']
                text = data['text']
                id = data['id']
                out_data.append({
                    "id": id,
                    "text": text,
                    "label": label,
                })


with open("./data/data.json", "w", encoding='utf8') as f:
    out_data = []
    for filepath, dirnames, filenames in os.walk('.\\raw_data'):
        for filename in filenames:
            print(os.path.join(filepath, filename))
            parse_data(os.path.join(filepath, filename), out_data)
    label_dict = defaultdict(int)
    for data in out_data:
        label_dict[data['label']] += 1
        f.write(json.dumps(data, ensure_ascii=False)+"\n")
    for k, v in label_dict.items():
        print(k + ": " + str(v))