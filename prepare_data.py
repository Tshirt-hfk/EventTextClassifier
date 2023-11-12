# coding:utf-8
from collections import defaultdict
import os, json

# def parse_data(file_path, out_data):
#     with open(file_path, "r", encoding='utf8') as f:
#         data_list = json.load(f)
#         for data in data_list:
#             if len(data['event_list']) == 1:
#                 label = data['event_list'][0]['class']
#                 text = data['text']
#                 id = data['id']
#                 out_data.append({
#                     "id": id,
#                     "text": text,
#                     "label": label,
#                 })


# with open("./data/data.json", "w", encoding='utf8') as f:
#     out_data = []
#     for filepath, dirnames, filenames in os.walk('.\\raw_data'):
#         for filename in filenames:
#             print(os.path.join(filepath, filename))
#             parse_data(os.path.join(filepath, filename), out_data)
#     label_dict = defaultdict(int)
#     for data in out_data:
#         label_dict[data['label']] += 1
#         f.write(json.dumps(data, ensure_ascii=False)+"\n")
#     for k, v in label_dict.items():
#         print(k + ": " + str(v))

LABEL = ['一带一路', '东突', '产业经贸', '其他非传统安全', '内政', '军事其他', '军事合作', '军事战略', '军事行动', '军工科技', 
        '前沿技术', '半岛局势', '印太战略', '国际恐怖', '基础设施', '外交', '太空安全', '宗教', '对华关系', '政治其他', 
        '政治安全', '政治局势', '政策制度', '敌对反华', '数字经济', '数据安全', '核安全', '武器装备', '民族问题其他', 
        '民用科技', '民运', '海外利益', '海外安全其他', '海洋安全', '涉恐其他', '涉疆', '涉疆反恐', '涉蒙', '涉藏', 
        '演习演训', '生物安全', '科技其他', '科技战略', '粮食安全', '经济其他', '经济制裁', '经济政策', '经济治理', 
        '网络安全其他', '网络战略', '网络攻防', '能源安全', '财政', '迈境问题', '邪教', '非政府组织', '高访安保']


def parse_data_v2(file_path, out_data):
    with open(file_path, "r", encoding='utf8') as f:
        json_data = json.load(f)
        for data_list in json_data.values():
            for data in data_list:
                label = sorted(list(set([x for x in data['classification'].split(",") if x in LABEL])))
                text = data['text']
                id = data['id']
                if label:
                    out_data.append({
                        "id": id,
                        "text": text,
                        "label": label,
                    })


with open("./data/data_v2.json", "w", encoding='utf8') as f:
    out_data = []
    for filepath, dirnames, filenames in os.walk('./raw_data/new_data'):
        for filename in filenames:
            print(os.path.join(filepath, filename))
            parse_data_v2(os.path.join(filepath, filename), out_data)
    label_dict = defaultdict(int)
    len_dict = defaultdict(int)
    for data in out_data:
        for label in data['label']:
            if len(label)==1:
                print(data, label)
            label_dict[label] += 1
        len_dict[len(data['text'])] += 1
        f.write(json.dumps(data, ensure_ascii=False)+"\n")
    label_num = sorted([(k, v) for k, v in label_dict.items()], key=lambda x:x[1], reverse=True)
    for k, v in label_num:
        print(k + ": " + str(v))
    print(sorted([label for label, _ in label_num]))
    print("label num:", len(label_dict))
    print("text length:", sorted([(k, v) for k,v in len_dict.items()], key=lambda x:x[0], reverse=True))