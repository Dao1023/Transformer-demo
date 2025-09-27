from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 加载带 MLM 头的 BERT
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.eval()
text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

with torch.no_grad():
    logits = mlm_model(**inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_5 = torch.topk(mask_token_logits, 5, dim=1)
    top_5_tokens = top_5.indices[0].tolist()
    top_5_probs = torch.softmax(top_5.values[0], dim=0).tolist()

print("Top 5 predictions for [MASK]:")
for token, prob in zip(top_5_tokens, top_5_probs):
    print(f"{tokenizer.decode([token])}: {prob:.4f}")
