import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
from typing import Tuple, Optional, Union


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


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class UniSRecPrompt(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.n_items = dataset.item_num

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

        # prompt
        # prefix_size = config['prefix_size']
        # prefix_length = config['prefix_length']
        # self.clip_project = MLP((prefix_size, (self.hidden_size * prefix_length) // 2,
        #                          self.hidden_size * prefix_length))
        self.pre_seq_len = config['n_prefix']
        # self.prefix_embeds = nn.Parameter(torch.empty(self.n_prefix, self.hidden_size)).to(self.device)
        # nn.init.normal_(self.prefix_embeds)
        # self.prefix_embeds = nn.Embedding(dataset.user_num, self.hidden_size, padding_idx=0).to(self.device)
        self.prefix_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )
        # self.clip_project = MLP((self.hidden_size, (self.hidden_size * self.n_prefix) // 2,
        #                          self.hidden_size * self.n_prefix))

        self.position_embedding = nn.Embedding(self.max_seq_length + self.pre_seq_len, self.hidden_size)

        # self.prefix_embeds = nn.Parameter(torch.empty(13102+1, self.hidden_size)).to(self.device)
        # nn.init.normal_(self.prefix_embeds)
        # self.experts = nn.ModuleList([PWLayer(self.hidden_size, self.hidden_size, config['adaptor_dropout_prob'])
        #                               for i in range(self.n_prefix)])

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        prompts = self.prefix_encoder(prefix_tokens)
        prompts = self.prefix_proj(prompts)
        return prompts

    def forward(self, item_seq, item_emb, item_seq_len):
        # expert_outputs = [self.experts[i](self.prefix_embeds) for i in range(self.n_prefix)]
        # expert_outputs = [expert_outputs[i][batch_users, :].unsqueeze(-2) for i in range(self.n_prefix)]
        # prefix = torch.cat(expert_outputs, dim=1)
        # prefix = self.prefix_embeds(batch_users).unsqueeze(1) # B * H
        # prompt_embeds = torch.cat((prefix, item_emb), dim=1)

        prompts = self.get_prompt(batch_size=item_seq.shape[0])
        inputs_embeds = torch.cat((prompts, item_emb), dim=1)

        position_ids = torch.arange(item_seq.size(1)+self.pre_seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.expand(item_seq.shape[0], -1)
        # position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = inputs_embeds + position_embedding
        # input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # attention mask
        prompt_attention_mask = prompts.new_ones((item_seq.shape[0], self.pre_seq_len))
        extended_item_seq = torch.cat([prompt_attention_mask, item_seq], dim=1)
        extended_attention_mask = self.get_attention_mask(extended_item_seq)
        # extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len+self.pre_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # # random sample a few items for each seq
        # rand_items = torch.randint(1, self.n_items, (item_seq.shape[0], self.n_prefix)).to(self.device)
        # item_seq = torch.cat((rand_items, item_seq), dim=1)
        # item_seq_len += self.n_prefix

        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        # batch_users = interaction['user_id']

        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

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
        # batch_users = interaction['user_id']
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
