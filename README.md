**Requirement**:

pytorch==1.11

tqdm==4.64

normflows==1.4

dgl==0.9.0

tensorboardx==2.5.1

**Data**:
Get data from:https://drive.google.com/uc?id=1ElKgnVcdoq7vdcC2N_N7lEeFP_Ughf_R&export=download

Get pretrained structural representations from: https://github.com/xwhan/One-shot-Relational-Learning

Get pretrained textual description representations from:

Wiki-One通过网盘分享的文件：qwen_emb.zip
链接: https://pan.baidu.com/s/15JkcYFpGSSOaIygBqFnKNg?pwd=j81t 提取码: j81t 
--来自百度网盘超级会员v1的分享

NELL-One通过网盘分享的文件：qwen_emb.pt
链接: https://pan.baidu.com/s/1BFmJpzcBzCDxxxqNpDINpw?pwd=vqen 提取码: vqen 
--来自百度网盘超级会员v1的分享

Get checkpoints from:

Wiki-One通过网盘分享的文件：dgl_wiki_last.zip
链接: https://pan.baidu.com/s/1oEv57UfgF8WU1Wv76vEMVw?pwd=4xe1 提取码: 4xe1 
--来自百度网盘超级会员v1的分享

NELL-One 通过网盘分享的文件：state_nell_512.zip
链接: https://pan.baidu.com/s/19Z_8IOheLCx3G2BLPKyMsg?pwd=8m7c 提取码: 8m7c 
--来自百度网盘超级会员v1的分享

**Eval**

Download the checkpoint and extract to the state/ folder.

NELL:python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --prefix dgl_nell_512 --device 0 --batch_size 32 --g_batch 512 --learning_rate 0.0001 --step test

Wiki:python main.py --dataset Wiki-One --data_path ./Wiki --few 5 --data_form Pre-Train --prefix dgl_wiki --device 0 --batch_size 64 -dim 50 --g_batch 1024 --eval_epoch 2000 --step test
