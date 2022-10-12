import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
import os
import numpy as np
import scipy.sparse as sp
from recbole.model.layers import MLPLayers


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


class tSRecV0(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']

        self.item_embedding = None
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

        self.topk = 10
        if os.path.exists('%s/text_adj_%d.pt'%(config['data_path'], self.topk)):
            text_adj = torch.load('%s/text_adj_%d.pt'%(config['data_path'], self.topk))
        else:
            text_adj = build_sim(self.plm_embedding.weight.detach()[1:]) # pos 0 is pad emb
            text_adj = build_knn_neighbourhood(text_adj, topk=self.topk)
            text_adj = compute_normalized_laplacian(text_adj)
            torch.save(text_adj, '%s/text_adj_%d.pt'%(config['data_path'], self.topk))
        self.text_adj = text_adj.to(config['device'])

        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj_matrix = get_norm_adj_mat(interaction_matrix).to(self.device)
        # self.norm_adj_matrix = build_knn_neighbourhood(self.norm_adj_matrix.to_dense(), topk).to_sparse()

        self.gcn_layers = 2

        self.mlp1 = MLPLayers([768, self.hidden_size], dropout=0.2, activation='relu', init_method=True)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp2 = MLPLayers([768, self.hidden_size], dropout=0.2, activation='relu', init_method=True)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))

        # GCN
        self.h = self.plm_embedding.weight[1:].clone().detach()
        for i in range(self.gcn_layers):
            self.h = torch.sparse.mm(self.text_adj, self.h)
        self.h = torch.cat((self.plm_embedding.weight[0].unsqueeze(0), self.h))  # concat with pad
        self.h = self.linear1(self.mlp1(self.h))
        self.h = F.normalize(self.h, dim=1)
        text_adj_seq_emb = self.h[item_seq]

        self.t = self.plm_embedding.weight.clone().detach()
        for i in range(self.gcn_layers):
            self.t = torch.sparse.mm(self.norm_adj_matrix, self.t)
        self.t = self.linear2(self.mlp2(self.t))
        self.t = F.normalize(self.t, dim=1)
        norm_adj_seq_emb = self.t[item_seq]

        seq_output = self.forward(item_seq, item_emb_list+norm_adj_seq_emb+text_adj_seq_emb, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))

        text_adj_seq_emb = self.h[item_seq]
        norm_adj_seq_emb = self.t[item_seq]

        seq_output = self.forward(item_seq, item_emb_list+norm_adj_seq_emb+text_adj_seq_emb, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_norm_adj_mat(interaction_matrix):
    r"""Get the normalized interaction matrix of users and items.

    Construct the square matrix from the training data and normalize it
    using the laplace matrix.

    .. math::
        A_{hat} = D^{-0.5} \times A \times D^{-0.5}

    Returns:
        Sparse tensor of the normalized interaction matrix.
    """
    # build adj matrix
    # A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
    inter_M_t = interaction_matrix.transpose()
    A = inter_M_t.dot(interaction_matrix)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    return SparseL