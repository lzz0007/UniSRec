import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
import math
from recbole.model.layers import VanillaAttention, MLPLayers
import os
from utils import build_sim, build_knn_neighbourhood, compute_normalized_laplacian, get_norm_adj_mat
import numpy as np


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.relu = nn.ReLU()

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.relu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class UniSRecWhiten(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']

        self.item_embedding = None

        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        # self.gamma = nn.Parameter(torch.zeros(768, 300), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros(768), requires_grad=True)
        # self.g = 8
        self.white_linear = nn.Linear(768, self.hidden_size)
        # self.plm_embedding_whiten = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # plm_emb_whiten = self.channel_whitening(dataset.plm_embedding.weight, self.g)
        # self.plm_embedding_whiten.weight.requires_grad = False
        # self.plm_embedding_whiten.weight.data.copy_(plm_emb_whiten)

        # self.moe_adaptor = MoEAdaptorLayer(
        #     config['n_exps'],
        #     config['adaptor_layers'],
        #     config['adaptor_dropout_prob']
        # )

        # self.gru1 = nn.GRU(
        #     input_size=300,
        #     hidden_size=self.hidden_size,
        #     num_layers=1,
        #     bias=False,
        #     batch_first=True,
        # )
        #
        # self.gru2 = nn.GRU(
        #     input_size=300,
        #     hidden_size=self.hidden_size,
        #     num_layers=1,
        #     bias=False,
        #     batch_first=True,
        # )

        # self.attn_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.gcn_layers = 2

        self.topk = 10
        # if os.path.exists('%s/text_adj_%d.pt'%(config['data_path'], self.topk)):
        #     text_adj = torch.load('%s/text_adj_%d.pt'%(config['data_path'], self.topk))
        # else:
        #     text_adj = build_sim(self.plm_embedding.weight.detach()[1:]) # pos 0 is pad emb
        #     text_adj = build_knn_neighbourhood(text_adj, topk=self.topk)
        #     text_adj = compute_normalized_laplacian(text_adj)
        #     torch.save(text_adj, '%s/text_adj_%d.pt'%(config['data_path'], self.topk))
        # self.text_adj = text_adj.to(config['device'])

        interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        self.norm_adj_matrix = get_norm_adj_mat(interaction_matrix).to(self.device)
        # self.norm_adj_matrix = build_knn_neighbourhood(self.norm_adj_matrix.to_dense(), self.topk).to_sparse()

        # self.gat1 = GAT(nfeat=768, nhid=self.hidden_size, nclass=self.hidden_size, dropout=0.2, nheads=2)
        # self.gat2 = GAT(nfeat=768, nhid=self.hidden_size, nclass=self.hidden_size, dropout=0.2, nheads=2)

        # self.mlp = MLPLayers([self.hidden_size*2, self.hidden_size, self.hidden_size], 0.2, 'relu')

        # self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        # self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, item_seq, item_emb, item_seq_len, ego_seq_emb, text_emb_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = input_emb + ego_seq_emb + text_emb_seq
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def gen_text_adj(self):
        text_adj = build_sim(self.test_item_emb[1:])  # pos 0 is pad emb
        text_adj = build_knn_neighbourhood(text_adj, topk=self.topk)
        text_adj = compute_normalized_laplacian(text_adj)
        return text_adj.to(self.test_item_emb.device)

    def GCN(self):
        # text_graph_emb = self.plm_embedding.weight[1:].clone().detach()
        # text_graph_emb = self.gat1(text_graph_emb, self.text_adj)
        text_graph_emb = self.test_item_emb[1:].clone().detach()
        text_adj = self.gen_text_adj()
        for i in range(self.gcn_layers):
            text_graph_emb = torch.sparse.mm(text_adj, text_graph_emb)
        # text_graph_emb = torch.cat((self.plm_embedding.weight[0].unsqueeze(0), text_graph_emb))  # concat with pad
        text_graph_emb = torch.cat((self.test_item_emb[0].unsqueeze(0), text_graph_emb))  # concat with pad
        # text_graph_emb = F.normalize(text_graph_emb, dim=1)

        # adj_graph_emb = self.plm_embedding.weight[1:].clone().detach()
        # adj_graph_emb = self.gat2(adj_graph_emb, self.norm_adj_matrix)
        adj_graph_emb = self.test_item_emb[1:].clone().detach()
        res = []
        for i in range(self.gcn_layers):
            adj_graph_emb = torch.sparse.mm(self.norm_adj_matrix, adj_graph_emb)
            res.append(adj_graph_emb)
        # adj_graph_emb = torch.cat((self.plm_embedding.weight[0].unsqueeze(0), adj_graph_emb))  # concat with pad
        adj_graph_emb = torch.cat((self.test_item_emb[0].unsqueeze(0), adj_graph_emb))  # concat with pad
        # adj_graph_emb = F.normalize(adj_graph_emb, dim=1)
        return adj_graph_emb, text_graph_emb

    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        # item_seq = interaction['item_seq_sorted']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # # get the items of a batch
        # items_in_batch = torch.unique(item_seq[item_seq != 0])
        # items_in_batch_whitened = self.channel_whitening(self.plm_embedding(items_in_batch), self.g) # here calculate grad
        # self.plm_embedding_whiten.weight[items_in_batch] = items_in_batch_whitened
        # item_emb_list = self.plm_embedding_whiten(item_seq)
        # test_item_emb = self.channel_whitening(self.plm_embedding.weight, self.g)
        # item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))

        self.test_item_emb = self.batch_whitening(self.plm_embedding.weight)
        item_emb_list = self.test_item_emb[item_seq]

        # GCN
        a_g_emb, t_g_emb = self.GCN()
        text_emb_seq = t_g_emb[item_seq]
        # text_emb, _ = self.gru1(text_emb_seq)
        # text_emb = self.gather_indexes(text_emb, item_seq_len - 1)

        ego_emb_seq = a_g_emb[item_seq]
        # adj_emb, _ = self.gru2(ego_emb_seq)
        # adj_emb = self.gather_indexes(adj_emb, item_seq_len - 1)
        # seq_mask = item_seq != 0
        # adj_emb = torch.sum(ego_emb_seq * seq_mask.unsqueeze(-1), 1) / torch.sum(seq_mask.float(), 1).unsqueeze(-1)

        # trm
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len, ego_emb_seq, text_emb_seq)
        # test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        # test_item_emb = self.plm_embedding_whiten.weight

        # combine
        # combined = torch.cat((seq_output, adj_emb), dim=1)
        # combined = self.mlp(combined)
        # combined = self.concat_layer(combined)
        # combined = self.LayerNorm(combined)
        # combined = self.dropout(combined)

        # combined = F.normalize(combined, dim=1)
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(self.test_item_emb + a_g_emb + t_g_emb, dim=1)

        # # combine lstm with trm
        # combined = torch.cat((seq_output.unsqueeze(1), adj_emb.unsqueeze(1), text_emb.unsqueeze(1)), dim=1)
        # combined, _ = self.attn_layer(combined)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # item_emb_list = self.plm_embedding_whiten(item_seq)
        # item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))

        # GCN
        # t_g_emb, a_g_emb = self.GCN()
        # text_emb_seq = t_g_emb[item_seq]
        # text_emb, _ = self.gru1(text_emb_seq)
        # text_emb = self.gather_indexes(text_emb, item_seq_len - 1)
        #
        # ego_emb_seq = a_g_emb[item_seq]
        # adj_emb, _ = self.gru2(ego_emb_seq)
        # adj_emb = self.gather_indexes(adj_emb, item_seq_len - 1)
        #
        # seq_output = F.normalize(seq_output, dim=-1)
        # test_items_emb = F.normalize(self.test_item_emb + t_g_emb + a_g_emb, dim=-1)

        # # combine lstm with trm
        # combined = torch.cat((seq_output.unsqueeze(1), adj_emb.unsqueeze(1), text_emb.unsqueeze(1)), dim=1)
        # combined, _ = self.attn_layer(combined)

        a_g_emb, t_g_emb = self.GCN()
        ego_emb_seq = a_g_emb[item_seq]
        text_emb_seq = t_g_emb[item_seq]
        # seq_mask = item_seq != 0
        # adj_emb = torch.sum(ego_emb_seq * seq_mask.unsqueeze(-1), 1) / torch.sum(seq_mask.float(), 1).unsqueeze(-1)

        item_emb_list = self.test_item_emb[item_seq]
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len, ego_emb_seq, text_emb_seq)
        # test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        # test_items_emb = self.plm_embedding_whiten.weight

        # combine
        # combined = torch.cat((seq_output, adj_emb), dim=1)
        # combined = self.mlp(combined)
        # combined = self.concat_layer(combined)
        # combined = self.LayerNorm(combined)
        # combined = self.dropout(combined)

        # combined = F.normalize(combined, dim=1)
        seq_output = F.normalize(seq_output, dim=1)
        test_items_emb = F.normalize(self.test_item_emb + a_g_emb + t_g_emb, dim=1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def channel_whitening(self, x, g, eps=1e-5):
        # x: input feature with size [m, d] or [m, d, H, W]
        # gamma, beta: the trainable affine
        # g: the group number of group whitening
        x_flatten = x.view(x.size()[0], -1)
        f_dim = x_flatten.size()[-1]
        shuffle = torch.randperm(f_dim).tolist()
        # centering
        mean = x_flatten.mean(-1, keepdim=True)
        x_centered = x_flatten - mean
        x_group = x_centered[:, shuffle].reshape(x.size()[0], g, -1).permute(1, 2, 0)
        f_cov = torch.bmm(x_group.permute(0, 2, 1), x_group) / (x_group.shape[1] - 1)
        eye = torch.eye(x.size(0)).type(x.type()).reshape(1, x.size(0), x.size(0)).repeat(g, 1, 1)
        # compute whitening matrix
        sigma = (1 - eps) * f_cov + eps * eye
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = torch.bmm(u, torch.diag_embed(scale))
        wm = torch.bmm(wm, u.permute(0, 2, 1))
        # whiten
        decorrelated = torch.bmm(x_group, wm)
        shuffle_recover = [shuffle.index(i) for i in range(f_dim)]
        decorrelated = decorrelated.permute(2, 0, 1).reshape(-1, f_dim)[:, shuffle_recover]
        output = decorrelated.view_as(x)
        output = self.white_linear(output)
        return output
        # return output * gamma + beta

    def batch_whitening(self, x):
        # code from "On feature decorrelation in self-supervised learning"
        N, D = x.shape
        # G = math.ceil(2 * D / N)
        G = 2
        new_idx = torch.randperm(D)
        x = x.t()[new_idx].t()
        x = x.view(N, G, D // G)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0, 1)  # G, N, D//G
        covs = x.transpose(1, 2).bmm(x) / N
        W = transformation(covs, x.device, engine='svd')
        x = x.bmm(W)
        output = x.transpose(1, 2).flatten(0, 1)[torch.argsort(new_idx)].t()
        output = self.white_linear(output)
        return output


def transformation(covs, device, engine='symeig'):
    covs = covs.to(device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1, 2).to(device)
    else:
        if engine == 'symeig':
            S, U = torch.symeig(covs.to(device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1, 2))
    return W

