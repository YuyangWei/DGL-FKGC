import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
import requests
import time
import logging
import re
#from dgl.ops import segment_topk
# ------------------ RPGNN ------------------

class RelationalPathGNN(nn.Module):
    def __init__(self, g, ent2id, num_rel, parameter):
        super(RelationalPathGNN, self).__init__()
        self.ent2id_dict = ent2id
        self.device = parameter['device']
        self.hop = parameter['hop']
        self.es = parameter['embed_dim']
        self.g_batch = parameter['g_batch']
        self.text_dim = parameter['text_dim']
        self.g = g
        self.num_ent = len(ent2id)
        
        # Neighbor sampling（只负责采样）
        self.sampler = dgl.dataloading.NeighborSampler(
            [100]*parameter['hop'],
            prefetch_node_feats=['feat'],
            prefetch_edge_feats=['feat', 'eid']
        )

        # GNN 本体（文本逻辑在 RPLayer 里）
        self.gcn = RPGNN(
            in_features=self.es,
            hidden_features=self.es * 2,
            out_features=self.es,
            hop=self.hop,
            n_rel=num_rel,
            device = self.device
        )

    def ent2id(self, triples):
        """
        triples: [batch, few, (h, r, t)]
        """
        idx = [
            [[self.ent2id_dict[t[0]], self.ent2id_dict[t[2]]] for t in batch]
            for batch in triples
        ]
        return torch.LongTensor(idx).to(self.device)

    def forward(self, triples):
        """
        返回：
        - out_emb: [batch, few, 2, emb_dim]
        - text_emb: [batch, few, 2, text_dim]
        """
        idx = self.ent2id(triples)
        #print("idx",idx.device)
        #print("self.ent2texts",self.ent2texts.device)
        text_emb = self.g.ndata['text'][idx.cpu()].to(self.device)
        #print("text_emb",text_emb.device)
        batch_size, few_shot = idx.shape[:2]

        # 展平作为 seed nodes
        idx = idx.view(-1)

        dataloader = dgl.dataloading.DataLoader(
            self.g,
            idx,
            self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True
        )

        out_emb = []

        for input_nodes, output_nodes, blocks in dataloader:
            # 节点结构特征
            x = blocks[0].srcdata['feat']

            # 文本特征已在 blocks 中（RPLayer 内部使用）
            out = self.gcn(blocks, x)
            out_emb.append(out)

        out_emb = torch.cat(out_emb, dim=0)
        out_emb = out_emb.view(batch_size, few_shot, 2, -1)

        return [out_emb, text_emb]



# ------------------ RPGNN 模块 ------------------

class RPGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hop, n_rel,device):
        super().__init__()
        emb_dim = in_features
        self.conv_in = RPLayer(emb_dim, in_features, hidden_features, n_rel, device)
        self.conv_out = RPLayer(emb_dim, hidden_features, out_features, n_rel, device)
        self.hop = hop
        if hop > 2:
            self.conv_hidden = nn.ModuleList(
            [RPLayer(emb_dim, hidden_features, hidden_features, n_rel) for _ in range(hop-2)]
            )

    def forward(self, blocks, x):
    
        # 对每一层 block 依次更新
        for i, block in enumerate(blocks):
            if i == 0:
                x = F.relu(self.conv_in(block, x))
            elif i == len(blocks)-1:
                x = F.relu(self.conv_out(block, x,))
            else:
                x = F.relu(self.conv_hidden[i-1](block, x))
        return x


# ------------------ RPLayer 模块 ------------------

class RPLayer(nn.Module):
    def __init__(self, emb_dim, in_feat, out_feat, num_rels, device):
        super().__init__()
        self.num_rels = num_rels
        self.device = device
        # 关系感知线性变换
        self.linear_r = dgl.nn.pytorch.TypedLinear(
            in_feat + emb_dim * 2,
            out_feat,
            num_rels
        )
        # 结构注意力（原 RPGNN 风格）
        self.attn_fc = nn.Linear(out_feat + emb_dim, 1, bias=False)

        # self-loop
        self.loop_weight = nn.Parameter(torch.Tensor(emb_dim, out_feat))
        self.h_bias = nn.Parameter(torch.Tensor(out_feat))

        nn.init.xavier_uniform_(self.loop_weight)
        nn.init.zeros_(self.h_bias)

    def edge_agg(self, edges, topk=15):
        """
        edges: DGL edges
        topk: int or None, 保留每个 dst 节点 sim_text 最大的 topk 条，其余置 0
        """
        # -------- 1. 关系感知消息 --------
        x = torch.cat([edges.src['h'], edges.data['feat'], edges.dst['feat']], dim=-1)
        m = self.linear_r(x, edges.data['eid'])  # [E, out_feat]
    
        # -------- 2. 结构注意力 --------
        attn_struct = self.attn_fc(torch.cat([edges.dst['feat'], m], dim=-1))  # [E,1]
        attn_struct = F.leaky_relu(attn_struct).view(-1)
    
        # -------- 3. 文本语义相似度 --------
        src_text = edges.src['text']  # [E, dim]
        dst_text = edges.dst['text']  # [E, dim]
    
        sim_text = F.cosine_similarity(src_text, dst_text, dim=-1, eps=1e-8)
        sim_attn = F.leaky_relu(sim_text)

        if topk is not None and topk > 0:

            dst_ids = edges.dst[dgl.NID]   # [E]

            # 先按 dst 分组
            sorted_dst, order = torch.sort(dst_ids)
        
            sim_sorted = sim_attn[order]
        
            mask = torch.zeros_like(sim_attn)
        
            unique_dst, counts = torch.unique_consecutive(
                sorted_dst,
                return_counts=True
            )
        
            start = 0
        
            for count in counts:
        
                count = count.item()
        
                end = start + count
        
                group_scores = sim_sorted[start:end]
        
                keep_num = min(topk, count)
        
                topk_idx = torch.topk(
                    group_scores,
                    k=keep_num,
                    largest=True
                ).indices
        
                original_idx = order[start:end][topk_idx]
        
                mask[original_idx] = 1.0
        
                start = end
        
            sim_attn = sim_attn * mask
            attn_struct = attn_struct * mask
        attn = attn_struct + sim_attn
        return {'h': m, 'z': attn.view(-1,1)}


    def forward(self, g, feat):
        with g.local_scope():
            # -------- 节点归一化 --------
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            g.srcdata['h'] = feat * norm

            # -------- Edge message --------
            g.apply_edges(self.edge_agg)
            e = g.edata.pop('z')
            # -------- Edge softmax（dst 归一化）--------
            alpha = edge_softmax(g, e)
            g.edata['h'] = alpha * g.edata['h']

            # -------- 聚合 --------
            g.update_all(
                dgl.function.copy_e('h', 'm'),
                dgl.function.sum('m', 'h')
            )
            h = g.dstdata['h']
            h = h + g.dstdata['feat'] @ self.loop_weight

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            return h * norm + self.h_bias

