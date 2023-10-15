# coding=utf-8
import torch
from flask import Flask, jsonify, abort, request
from dataloader import LABEL2TYPE
from model import TextClassifierModel
from transformers import AutoTokenizer
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
model = TextClassifierModel("../pretrain/chinese-roberta-wwm-ext")
model.load_state_dict(torch.load("./output/model_9.pt"))
mdoel = model.cuda()
model.eval()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/event_classify", methods=["POST"])
def process():
    if request.content_type.startswith('application/json'):
        text = request.json.get('text')
    elif request.content_type.startswith('multipart/form-data'):
        text = request.form.get('text')
    else:
        text = request.values.get("text")

    try:
        tokenized_input = tokenizer([text])
        logits = model(tokenized_input.input_ids, tokenized_input.attention_mask, tokenized_input.token_type_ids)
        pred_label = torch.max(logits, dim=-1)[1][0]
        rsp_json = {
            "code": 200,
            "message": "success",
            "result": {
                "label": LABEL2TYPE[pred_label]
            }
        }
    except Exception as e:
        rps_json = {
                "code": 500,
                "message": "%s" % e
            }
    
    return jsonify(rsp_json)

if __name__ == '__main__':
    app.run(debug=True)
