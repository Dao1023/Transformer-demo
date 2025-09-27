from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

sent1 = "I enjoy hiking in the mountains."
sent2 = "I like walking in nature."
sent3 = "I hate doing homework."
sent4 = "My name is Dao."
sent5 = "So Nvidia fuck you"

emb1 = get_cls_embedding(sent1)
emb2 = get_cls_embedding(sent2)
emb3 = get_cls_embedding(sent3)
emb4 = get_cls_embedding(sent4)
emb5 = get_cls_embedding(sent5)

sim12 = cosine_similarity(emb1, emb2)[0][0]
sim13 = cosine_similarity(emb1, emb3)[0][0]
sim14 = cosine_similarity(emb1, emb4)[0][0]
sim15 = cosine_similarity(emb1, emb5)[0][0]

print(f"Similarity between sent1 and sent2: {sim12:.4f}")
print(f"Similarity between sent1 and sent3: {sim13:.4f}")
print(f"Similarity between sent1 and sent4: {sim14:.4f}")
print(f"Similarity between sent1 and sent5: {sim15:.4f}")
