import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
import os
import scipy.sparse as sp
# from recbole.model.general_recommender import LightGCN
import numpy as np
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


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class tSRec(SASRec):
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

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        # self.norm_adj_matrix = build_knn_neighbourhood(self.norm_adj_matrix.to_dense(), topk).to_sparse()

        self.gcn_layers = 2

        # self.label = [0.8, 0.84, 0.86, 0.88, 0.9, 0.95]
        # self.sim_position_embedding = nn.Embedding(len(self.label)+3, self.hidden_size)
        # self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru1 = nn.GRU(
            input_size=300,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=False,
            batch_first=True,
        )

        self.gru2 = nn.GRU(
            input_size=300,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=False,
            batch_first=True,
        )

        self.attn_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.attn_layer_2 = VanillaAttention(300, self.hidden_size)
        self.mlp = MLPLayers([300, self.hidden_size], dropout=0.2, activation='relu', init_method=True)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def update_text_adj(self):
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        text_adj = build_sim(test_item_emb.detach()[1:])  # pos 0 is pad emb
        text_adj = build_knn_neighbourhood(text_adj, topk=self.topk)
        text_adj = compute_normalized_laplacian(text_adj)
        self.text_adj = text_adj.to(self.device)

    def get_norm_adj_mat(self):
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
        inter_M_t = self.interaction_matrix.transpose()
        A = inter_M_t.dot(self.interaction_matrix)
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

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # sim_position_embedding = self.sim_position_embedding(item_seq_sim_pos)

        input_emb = item_emb + position_embedding
        # if self.train_stage == 'transductive_ft':
        #     input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1] # last trm layer output
        # output = self.gather_indexes(output, item_seq_len - 1) # retrieve emb at the last pos
        return output  # [B H]

    def sim_position(self, res):
        for i, l in enumerate(self.label):
            if i == 0:
                res[(res <= l) & (res > 0)] = i + 2
            else:
                res[(res > self.label[i - 1]) & (res <= l)] = i + 2
        res[(res > self.label[-1]) & (res < 1)] = len(self.label) + 2
        return res.int()

    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        # if self.train_stage == 'transductive_ft':
        #     test_item_emb = test_item_emb + self.item_embedding.weight

        # item_seq_sim = interaction['item_seq_sim']
        # item_seq_sim_pos = self.sim_position(item_seq_sim)

        # GCN
        self.h = test_item_emb[1:].clone().detach()
        # self.h = self.plm_embedding.weight[1:].clone().detach()
        for i in range(self.gcn_layers):
            self.h = torch.sparse.mm(self.text_adj, self.h)
        self.h = torch.cat((test_item_emb[0].unsqueeze(0), self.h)) # concat with pad
        self.h = F.normalize(self.h, dim=1)
        text_emb_seq = self.h[item_seq]
        # text_emb_seq = torch.sum(text_emb_seq, 1) / torch.sum(item_seq != 0, 1).unsqueeze(1)
        # ego_emb_seq = self.linear1(ego_emb_seq)
        # text_emb = self.gru1(text_emb_seq)[1].squeeze()
        text_emb, _ = self.gru1(text_emb_seq)
        text_emb = self.gather_indexes(text_emb, item_seq_len - 1)

        self.t = test_item_emb.clone().detach()
        # self.t = self.plm_embedding.weight.clone().detach()
        for i in range(self.gcn_layers):
            self.t = torch.sparse.mm(self.norm_adj_matrix, self.t)
        self.t = F.normalize(self.t, dim=1)
        ego_emb_seq = self.t[item_seq]
        # ego_emb_seq = torch.sum(ego_emb_seq, 1) / torch.sum(item_seq != 0, 1).unsqueeze(1)
        # text_emb_seq = self.linear2(text_emb_seq)
        # adj_emb = self.gru2(ego_emb_seq)[1].squeeze()
        adj_emb, _ = self.gru2(ego_emb_seq)
        adj_emb = self.gather_indexes(adj_emb, item_seq_len - 1)

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=-1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        # combine GCN with trm emb
        # combined_emb = ego_emb_seq + seq_output
        output = self.gather_indexes(seq_output, item_seq_len - 1)  # retrieve emb at the last pos

        # combine lstm with trm
        # combined = torch.cat((output.unsqueeze(1), adj_emb.unsqueeze(1), text_emb.unsqueeze(1)), dim=1)
        # combined, _ = self.attn_layer(combined)
        combined = output + adj_emb + text_emb

        # enhance item rep
        # combined_item_rep = torch.cat((self.t.unsqueeze(1), self.h.unsqueeze(1)), dim=1)
        # combined_item_rep, _ = self.attn_layer_2(combined_item_rep)
        # combined_item_rep = self.linear(self.mlp(combined_item_rep))
        # combined_item_rep += test_item_emb
        combined_item_rep = test_item_emb + self.t + self.h

        # CE loss
        logits = torch.matmul(combined, combined_item_rep.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)

        # Remove sequences with the same next item
        # pos_id = interaction['item_id']
        # same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        # same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        # loss = self.seq_item_contrastive_task(output, same_pos_id, interaction)

        # # uniformity
        # loss_u1, loss_u2 = self.uniformity(output1), self.uniformity(output2)
        # loss_a = self.alignment(output1, output2)

        # l1 = self.contrastive_task(output, ego_emb_seq, same_pos_id)
        # l2 = self.contrastive_task(output, text_emb_seq, same_pos_id)
        return loss

    def contrastive_task(self, seq_output, pos_items_emb, same_pos_id):
        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        # if self.train_stage == 'transductive_ft':
        #     test_items_emb = test_items_emb + self.item_embedding.weight

        # item_seq_sim = interaction['item_seq_sim']
        # item_seq_sim_pos = self.sim_position(item_seq_sim)

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        text_emb_seq = self.h[item_seq]
        # text_emb_seq = torch.sum(text_emb_seq, 1) / torch.sum(item_seq != 0, 1).unsqueeze(1)
        # ego_emb_seq = self.linear1(ego_emb_seq)
        text_emb, _ = self.gru1(text_emb_seq)
        text_emb = self.gather_indexes(text_emb, item_seq_len - 1)

        ego_emb_seq = self.t[item_seq]
        # ego_emb_seq = torch.sum(ego_emb_seq, 1) / torch.sum(item_seq != 0, 1).unsqueeze(1)
        # text_emb_seq = self.linear2(text_emb_seq)
        adj_emb, _ = self.gru2(ego_emb_seq)
        adj_emb = self.gather_indexes(adj_emb, item_seq_len - 1)

        # combined_emb = self.h[item_seq] + seq_output
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # retrieve emb at the last pos

        # combined = torch.cat((seq_output.unsqueeze(1), adj_emb.unsqueeze(1), text_emb.unsqueeze(1)), dim=1)
        # combined, _ = self.attn_layer(combined)
        combined = seq_output + adj_emb + text_emb

        # # enhance item rep
        # combined_item_rep = torch.cat((self.t.unsqueeze(1), self.h.unsqueeze(1)), dim=1)
        # combined_item_rep, _ = self.attn_layer_2(combined_item_rep)
        # combined_item_rep = self.linear(self.mlp(combined_item_rep))
        # combined_item_rep += test_items_emb
        combined_item_rep = test_items_emb + self.t + self.h

        # compute score
        scores = torch.matmul(combined, combined_item_rep.transpose(0, 1))  # [B n_items]
        return scores
