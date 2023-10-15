# coding=utf-8
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from dataloader import TYPE2LABEL, BatchTextDataset
from model import TextClassifierModel

epoch = 10
tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
data_train = BatchTextDataset("./data/train.json", tokenizer, batch_size=64, max_num_per_label=100, shuffle=True, use_gpu=True, drop_last=True)
data_dev = BatchTextDataset("./data/dev.json", tokenizer, max_num_per_label=None, shuffle=False, use_gpu=True, drop_last=False)

model = TextClassifierModel("../pretrain/chinese-roberta-wwm-ext", num_label=len(TYPE2LABEL)).cuda()
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if "LayerNorm" not in n], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if "LayerNorm" in n], 'weight_decay': 0.0}
], lr=4e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)


for idx in range(epoch):
    print("epoch {} start to train ====>".format(idx))
    print("epoch: {}, lr: {}".format(idx, scheduler.get_last_lr()))
    model.train()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_train):
        
        # print(input_ids[0])
        # print(attention_mask[0])
        # print(token_type_ids[0])
        # print(labels[0])

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = model.calc_loss(logits, labels)
        
        print("epoch: {}-{}, loss: {},".format(idx, i, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch {} start to dev ====>".format(idx))
    model.eval()

    labels_list = []
    pred_labels_list = []
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_dev):
        logits = model(input_ids, attention_mask, token_type_ids)
        pred_labels = torch.max(logits, dim=-1)[1]
        labels_list += labels.tolist()
        pred_labels_list += pred_labels.tolist()
    assert len(labels_list) == len(pred_labels_list)
    cmp_list = [x==y for x, y in zip(pred_labels_list, labels_list)]
    print("acc:", round(sum(cmp_list)/len(cmp_list), 3))

    scheduler.step()

    torch.save(model.state_dict(),'./output/model_{}.pt'.format(idx))
    

    # for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_dev_loader):
    #     logits = model(input_ids, attention_mask, token_type_ids)