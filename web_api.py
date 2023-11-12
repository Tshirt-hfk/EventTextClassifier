# coding=utf-8
import torch
from flask import Flask, jsonify, abort, request
from dataloader import LABEL2TYPE, TYPE2LABEL
from model import TextClassifierModel
from transformers import AutoTokenizer
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
model = TextClassifierModel("../pretrain/chinese-roberta-wwm-ext", num_label=len(TYPE2LABEL))
model.load_state_dict(torch.load("./output/model_45.pt"))
mdoel = model.cuda()
model.eval()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/event_classify", methods=["POST"])
def process():
    if request.content_type.startswith('application/json'):
        text = request.json.get('full_text')
        threshold = request.json.get('threshold', 0.5)
    elif request.content_type.startswith('multipart/form-data'):
        text = request.form.get('full_text')
        threshold = request.form.get('threshold', 0.5)
    else:
        text = request.values.get("full_text")
        threshold = request.values.get('threshold', 0.5)

    try:
        tokenized_input = tokenizer([text])
        input_ids = torch.LongTensor(tokenized_input.input_ids).cuda()
        attention_mask = torch.LongTensor(tokenized_input.attention_mask).cuda()
        token_type_ids = torch.LongTensor(tokenized_input.token_type_ids).cuda()
        logits = model(input_ids, attention_mask, token_type_ids)
        pred_logits = logits.tolist()[0]
        last_labels = [LABEL2TYPE[idx] for idx, logit in enumerate(pred_logits) if logit>threshold]
        rsp_json = {
            "code": 200,
            "message": "success",
            "result": {
                "label": last_labels
            }
        }
    except Exception as e:
        rsp_json = {
                "code": 500,
                "message": "%s" % e
            }
    
    return jsonify(rsp_json)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
