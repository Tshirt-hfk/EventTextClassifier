# coding:utf-8
from collections import defaultdict
import json
import random
import torch

LABEL2TYPE = ['一带一路', '东突', '产业经贸', '其他非传统安全', '内政', '军事其他', '军事合作', '军事战略', '军事行动', '军工科技', 
            '前沿技术', '半岛局势', '印太战略', '国际恐怖', '基础设施', '外交', '太空安全', '宗教', '对华关系', '政治其他', 
            '政治安全', '政治局势', '政策制度', '敌对反华', '数字经济', '数据安全', '核安全', '武器装备', '民族问题其他', 
            '民用科技', '民运', '海外利益', '海外安全其他', '海洋安全', '涉恐其他', '涉疆', '涉疆反恐', '涉蒙', '涉藏', 
            '演习演训', '生物安全', '科技其他', '科技战略', '粮食安全', '经济其他', '经济制裁', '经济政策', '经济治理', 
            '网络安全其他', '网络战略', '网络攻防', '能源安全', '财政', '迈境问题', '邪教', '非政府组织', '高访安保']

TYPE2LABEL = {
    label: idx for idx,label in enumerate(LABEL2TYPE)
}

def convert_one_hot(type_list):
    one_hot = [0 for _ in range(len(LABEL2TYPE))]
    for t in type_list:
        one_hot[TYPE2LABEL[t]] = 1
    return one_hot

class TextDataset:
     def __init__(self, file_path, tokenizer):
        super().__init__()
        self.examples = []
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                example = json.loads(line.strip())
                tokenized_input = tokenizer(example["text"])
                example = {
                    "input_ids": tokenized_input.input_ids,
                    "attention_mask": tokenized_input.attention_mask,
                    "token_type_ids": tokenized_input.token_type_ids,
                    "label": convert_one_hot(example['label'])
                }
                self.examples.append(example)

     def __len__(self):
         return len(self.examples)
        
     def __getitem__(self, index):
        assert index < len(self.examples)
        return self.examples[index]


class BatchTextDataset(TextDataset):

    def __init__(self, file_path, tokenizer, batch_size=32, padding=0, max_num_per_label=None, shuffle=True, use_gpu=False, drop_last=False):
        super().__init__(file_path, tokenizer)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_gpu = use_gpu
        self.drop_last = drop_last
        self.padding = padding
        self.max_num_per_label = max_num_per_label
        self.batch_examples = []
        
    def __len__(self):
        return len(self.batch_examples)
    
    def __getitem__(self, idx):
        return self.batch_examples[idx]
  
    def __iter__(self):
        self.batch_padding()
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= len(self.batch_examples):
            raise StopIteration
        data = self.batch_examples[self.idx]
        self.idx += 1
        return data
    
    def batch_padding(self):
        self.batch_examples.clear()
        if self.shuffle:
            random.shuffle(self.examples)
            examples_per_label = defaultdict(list)
            for example in self.examples:
                examples_per_label[tuple(example["label"])].append(example)
            max_num = max([len(v) for v in examples_per_label.values()])
            if self.max_num_per_label is not None:
                max_num = min(max_num, self.max_num_per_label)
            batch = []
            for i in range(max_num):
                for key in examples_per_label.keys():
                    if i < len(examples_per_label[key]):
                        batch.append(examples_per_label[key][i])
                        if len(batch) >= self.batch_size:
                            self.batch_examples.append(self.pad_input(batch))
                            batch.clear()
            if batch and not self.drop_last:
                self.batch_examples.append(self.pad_input(batch))
            random.shuffle(self.batch_examples)
        else:
            max_num = min(len(self.examples), self.max_num_per_label) if self.max_num_per_label is not None else len(self.examples)
            batch = []
            for i in range(max_num):
                batch.append(self.examples[i])
                if len(batch) >= self.batch_size:
                    self.batch_examples.append(self.pad_input(batch))
                    batch.clear()
            if batch and not self.drop_last:
                self.batch_examples.append(self.pad_input(batch))

    def pad_input(self, batch_data):
        batch_input_ids = [data['input_ids'] for data in batch_data]
        batch_attention_mask = [data['attention_mask'] for data in batch_data]
        batch_token_type_ids = [data['token_type_ids'] for data in batch_data]
        batch_labels = [data['label'] for data in batch_data]
        max_len = max([len(input_ids) for input_ids in batch_input_ids])
        batch_input_ids = torch.LongTensor([input_ids + [self.padding] * (max_len - len(input_ids)) 
                                                for input_ids in batch_input_ids])
        batch_attention_mask = torch.LongTensor([attention_mask + [self.padding] * (max_len - len(attention_mask))
                                                    for attention_mask in batch_attention_mask])
        batch_token_type_ids = torch.LongTensor([token_type_ids + [self.padding] * (max_len - len(token_type_ids))
                                                    for token_type_ids in batch_token_type_ids])
        batch_labels = torch.LongTensor(batch_labels)
        if self.use_gpu:
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            batch_labels = batch_labels.cuda()
        return (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels)

