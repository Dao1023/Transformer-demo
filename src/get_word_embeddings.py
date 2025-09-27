from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_word_embeddings(sentence, tokenizer, model):
    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0]  # (seq_len, 768)

    # 映射原始词到子词
    words = sentence.split()
    word_embeddings = []
    token_idx = 1  # 跳过 [CLS]

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        word_vecs = []
        for _ in word_tokens:
            if token_idx < len(embeddings):
                word_vecs.append(embeddings[token_idx])
                token_idx += 1
        if word_vecs:
            word_emb = torch.mean(torch.stack(word_vecs), dim=0)
            word_embeddings.append((word, word_emb))
    return word_embeddings

# 测试
sentence = "Playing football is fun."
word_embs = get_word_embeddings(sentence, tokenizer, model)
for word, emb in word_embs:
    print(f"Word: {word}, Embedding shape: {emb.shape}")