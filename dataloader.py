# coding:utf-8
from collections import defaultdict
import json
import random
import torch
from torch.utils.data import Dataset


TYPE2LABEL = {
    "伤人": 0,
    "咨询": 1,
    "威胁": 2,
    "开展合作": 3,
    "战斗": 4,
    "抗议": 5,
    "拒绝": 6,
    "涉及文件行为": 7,
    "胁迫": 8,
    "非常规大规模暴力": 9,
    "会议活动": 10,
    "关系建立": 11,
    "关系降级": 12,
    "涉国内法行为": 13,
    "涉国际法行为": 14,
    "立法行为": 15,
    "组织机构变更": 16
}
LABEL2TYPE = {v:k for k,v in TYPE2LABEL.items()}

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
                    "label": TYPE2LABEL[example['label']]
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
                examples_per_label[example["label"]].append(example)
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

