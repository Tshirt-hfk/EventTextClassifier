# coding:utf-8
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class TextClassifierModel(nn.Module):

    def __init__(self,
                 bert_dir, num_label,
                 dropout_prob=0.1):
        super(TextClassifierModel, self).__init__()

        assert os.path.exists(bert_dir), 'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)
        self.hidden_size = self.bert_module.config.hidden_size
        self.num_attention_heads = self.bert_module.config.num_attention_heads

        self.num_label = num_label

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_label)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        bert_out = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_out.last_hidden_state, bert_out.pooler_output
        text_logit = self.classifier(bert_out.pooler_output)
        
        return text_logit

    def calc_loss(self, text_logit, text_label):
        
        text_logit = torch.reshape(text_logit, [-1, self.num_label])
        text_label = torch.reshape(text_label, [-1])
        text_loss = F.cross_entropy(text_logit, text_label, reduction='mean')

        return text_loss
