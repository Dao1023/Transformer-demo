from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 中文模型示例
zh_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
zh_model = BertModel.from_pretrained('bert-base-chinese')

text = "今天天气真好！"
inputs = zh_tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = zh_model(**inputs)
print(f"中文句子 '{text}' 的 [CLS] 向量维度: {outputs.last_hidden_state[:,0,:].shape}")