from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentences = [
    "I love natural language processing.",
    "BERT is a powerful language model.",
    "Today is a sunny day."
]

for sent in sentences:
    inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] token 的输出作为句向量（batch_size=1, hidden_size=768）
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    print(f"Sentence: {sent}")
    # print(f"[CLS] embedding shape: {cls_embedding.shape}")  # torch.Size([768])
    print(f"[CLS] embedding: {cls_embedding}")
    print("---")