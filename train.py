# coding=utf-8
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from dataloader import TYPE2LABEL, BatchTextDataset
from model import TextClassifierModel

accumulation = 4
epoch = 50
tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
data_train = BatchTextDataset("./data/train.json", tokenizer, batch_size=128, max_num_per_label=100, shuffle=True, use_gpu=True, drop_last=True)
data_dev = BatchTextDataset("./data/dev.json", tokenizer, max_num_per_label=None, shuffle=False, use_gpu=True, drop_last=False)

model = TextClassifierModel("../pretrain/chinese-roberta-wwm-ext", num_label=len(TYPE2LABEL)).cuda()
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if "LayerNorm" not in n], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if "LayerNorm" in n], 'weight_decay': 0.0}
], lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=False)

step = 0
accumulation_loss = 0
for idx in range(1, epoch+1):
    print("epoch {} start to train ====>".format(idx))
    print("epoch: {}, lr: {}".format(idx, scheduler.get_last_lr()))
    model.train()
    
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_train):
        
        # print(input_ids[0])
        # print(attention_mask[0])
        # print(token_type_ids[0])
        # print(labels[0])

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = model.calc_loss(logits, labels)/accumulation
        loss.backward()
        accumulation_loss += loss
        step += 1
        if step % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("epoch: {}-{}, loss: {},".format(idx, step//accumulation, accumulation_loss))
            accumulation_loss = 0

    print("epoch {} start to dev ====>".format(idx))
    model.eval()

    labels_list = []
    pred_labels_list = []
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_dev):
        logits = model(input_ids, attention_mask, token_type_ids)
        pred_labels = logits > 0.5
        labels_list += labels.tolist()
        pred_labels_list += pred_labels.tolist()
    assert len(labels_list) == len(pred_labels_list)
 
    acc_num = label_num = pred_num = 0
    for pred_labels, labels in zip(pred_labels_list, labels_list):
        for pred_label, label in zip(pred_labels, labels):
            if pred_label==1 and label == 1:
                acc_num += 1
            if label == 1:
                label_num += 1
            if pred_label == 1:
                pred_num += 1
    precision = round(acc_num/(label_num+1e-12), 3)
    recall = round(acc_num/(pred_num+1e-12), 3)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    print("precision:", precision, "recall:", recall, "f1:", f1)
    
    scheduler.step()
    torch.save(model.state_dict(),'./output/model_{}.pt'.format(idx))
    