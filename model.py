from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import kl_divergence
from relational_path_gnn import RelationalPathGNN
import math

class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed_size * 2, self.n_hidden, self.layers, bidirectional=True, dropout=self.dropout)
        # self.gru = nn.GRU(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden * 2 * self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2).cuda()
        # batchnorm = nn.BatchNorm1d(5, affine=False).cuda()
        # attn_weight = batchnorm(attn_weight)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weight)
        context = context.view(-1, self.n_hidden * 2 * self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers * 2, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers * 2, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)  # change log

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(4))
        #self.weight = [1,1,1,1]
        #nn.init.constant_(self.weight, 1)
        #nn.init.xavier_normal_(self.weight)

    def forward(self, h, t, h_multi, t_multi, r, pos_num):
        score_ss= -torch.norm(h + r - t, 2, -1).squeeze(2)
        score_is = -torch.norm(h_multi + r - t, 2, -1).squeeze(2)
        score_si = -torch.norm(h + r - t_multi, 2, -1).squeeze(2)
        score_ii = -torch.norm(h_multi + r - t_multi, 2, -1).squeeze(2)
        
        #weights = torch.softmax(self.weight, dim=0)  # [4]
        #score = self.weight[0] * score_ss + self.weight[1] * score_is + self.weight[2] * score_si + self.weight[3] * score_ii
        score = score_ss + score_is + score_si + score_ii
        #score = score_ii
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class DGLFKGC(nn.Module):
    def __init__(self, g, dataset, parameter):
        super(DGLFKGC, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.text_dim = parameter['text_dim']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.ent2id = dataset['ent2id']
        self.num_rel = len(self.rel2id)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.num_hidden1 = 500
        self.num_hidden2 = 200
        self.lstm_dim = parameter['lstm_hiddendim']
        self.lstm_layer = parameter['lstm_layers']
        self.np_flow = parameter['flow']
        self.gate_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.gate_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.transfer = nn.Linear(self.text_dim, self.embed_dim)
        self.transfer_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.s_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.s_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.s_attn_w = nn.Linear(self.embed_dim, 1)
        nn.init.xavier_normal_(self.s_w.weight)
        nn.init.constant_(self.s_b, 0)
        nn.init.xavier_normal_(self.s_attn_w.weight)
        nn.init.constant_(self.transfer_b, 0)
 
        self.r_path_gnn = RelationalPathGNN(g, dataset['ent2id'], len(dataset['rel2emb']), parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.r_dim = self.z_dim = 50
            self.relation_learner = LSTM_attn(embed_size=50, n_hidden=400, out_size=50, layers=2, dropout=0.5)
            self.embedding_learner = EmbeddingLearner()

        elif parameter['dataset'] == 'NELL-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=self.lstm_dim, out_size=100,
                                              layers=self.lstm_layer, dropout=self.dropout_p)
            self.embedding_learner = EmbeddingLearner()

        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

        nn.init.xavier_normal_(self.gate_w.weight)
        nn.init.constant_(self.gate_b, 0) 
        nn.init.xavier_normal_(self.transfer.weight) 
        
    def eval_reset(self):
        self.eval_query = None
        self.eval_z = None
        self.eval_rel = None
        self.is_reset = True

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def fuse_cons(self,structured_embedding, text_embedding):

        #text_embedding = self.multi_map(text_embedding)
        
        gate = torch.sigmoid(self.gate_w(text_embedding) + self.gate_b)
        entity_embed = gate * text_embedding + (1.0 - gate) * structured_embedding
        return entity_embed

    def multi_map(self, multi_emb):
        mapped_multi = self.transfer(multi_emb) + self.transfer_b
        return mapped_multi.tanh()

    def get_s_multi(self, struc, multi):    

        multi_emb = self.multi_map(multi)   
        inter = struc * multi_emb 
        out = self.s_w(inter) + self.s_b
        attn_score = self.s_attn_w(out)  # [B, N, 1]    
        attn_weight = F.softmax(attn_score, dim=1)    
        fusion_out = inter * attn_weight
    
        return fusion_out#.tanh()
    
    def forward(self, task, iseval=False, curr_rel='', istest=False):
        # transfer task string into embedding
        support_all, support_negative_all, query_all, negative_all = [self.r_path_gnn(t) for t in task]
        #norm_vector = self.h_embedding(task[0])
        support, support_text = support_all
        support_negative, support_negative_text = support_negative_all
        query, query_text = query_all
        negative, negative_text = negative_all
        
        few = support.shape[1]              
        num_sn = support_negative.shape[1]  
        num_q = query.shape[1]              
        num_n = negative.shape[1] 

        support_refine = self.get_s_multi(support, support_text)
        support_negative_refine = self.get_s_multi(support_negative, support_negative_text)
        query_refine = self.get_s_multi(query, query_text)
        negative_refine = self.get_s_multi(negative, negative_text)
        
        support_fusion = self.fuse_cons(support, support_refine)
        support_few = support_fusion.view(support.shape[0], self.few, 2, self.embed_dim)
         

        rel = self.relation_learner(support_few)  # [bs, 1, d]
        rel.retain_grad()      
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
                sup_neg_e1_multi, sup_neg_e2_multi = self.split_concat(support_refine, support_negative_refine)
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, sup_neg_e1_multi, sup_neg_e2_multi, rel_s, few)
                y = torch.ones_like(p_score).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta	
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q
            
        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)  # [bs, nq+nn, 2, d]
        
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        que_neg_e1_multi, que_neg_e2_multi = self.split_concat(query_refine, negative_refine)
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, que_neg_e1_multi, que_neg_e2_multi, rel_q, num_q)
        

        return p_score, n_score

