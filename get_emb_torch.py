import json
import torch

ent2id = json.load(open('./Wiki/ent2ids'))
text_emb =  json.load(open('./Wiki/qwen_text_2.json'))
ent2real = json.load(open('./Wiki/qid2name_final.json'))
ent2texts = torch.zeros((len(ent2id),self.text_dim))
for ent in ent2id:
    if text_emb[ent2real[ent]] != None:
        ent2texts[ent2id[ent]] = torch.tensor(text_emb[ent2real[ent]],dtype=torch.float16)
    else:
        ent2texts[ent2id[ent]] = torch.randn(1, self.text_dim, dtype=torch.float16)
ent2texts = F.normalize(ent2texts, p=2, dim=-1)  

save_path = './Wiki/ent2texts.pt'
torch.save(ent2texts, save_path)