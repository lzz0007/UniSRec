import numpy as np
import pandas as pd
from data.dataset import UniSRecDataset
from recbole.config import Config
from model.unisrec import UniSRec
from recbole.data import data_preparation
import torch.nn as nn
import torch
from tqdm import tqdm

dataset = 'Pantry'
props = ['props/UniSRec.yaml', 'props/finetune.yaml']
config = Config(model=UniSRec, dataset=dataset, config_file_list=props)
dataset = UniSRecDataset(config)

# raw item text emb
text_emb = dataset.plm_embedding
mapping = dataset.field2id_token['item_id']

train_data, valid_data, test_data = data_preparation(config, dataset)

user_map = {v: k for k, v in test_data.dataset.field2token_id['user_id'].items()}
output = []
for i, batch in tqdm(enumerate(test_data), total=len(test_data)):
    user = batch[0]['user_id']
    item_seq = batch[0]['item_id_list']
    lengths = batch[0]['item_length']
    for b in range(user.shape[0]):
        uid = user[b].item()
        length = lengths[b]
        seq = item_seq[b, :length]
        seq_item_emb = text_emb(seq)
        a_norm = seq_item_emb / seq_item_emb.norm(dim=1)[:, None]
        res = torch.mm(a_norm, a_norm.transpose(0, 1))
        idx = torch.triu_indices(*res.shape, 1) # upper tri without diagonal
        sim = torch.mean(res[idx[0], idx[1]])
        output.append([int(user_map[uid]), length.item(), sim.item()])

out = pd.DataFrame(output, columns=['uid', 'len', 'sim'])
out.describe()
out.head()

out[out['sim']<=0.8].describe()
out[(out['sim']<=0.84) & (out['sim']>0.8)].describe()
out[(out['sim']<=0.85) & (out['sim']>0.84)].describe()
out[(out['sim']<=0.86) & (out['sim']>0.85)].describe()
out[(out['sim']<=0.88) & (out['sim']>0.86)].describe()
out[(out['sim']<=0.9) & (out['sim']>0.88)].describe()
out[(out['sim']<=0.95) & (out['sim']>0.9)].describe()
out[out['sim']>0.95].describe()

df = pd.read_csv('recommend_topk/UniSRec-Sep-28-2022_17-19-05.csv', sep='\t')
df = pd.merge(df, out, left_on='uid', right_on='uid')
df.head()


def metrics(pos_len, pos_index):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result_ndcg = dcg / idcg
    # ndcg
    ndcg5 = round(result_ndcg[:, 4].mean(), 5)
    ndcg10 = round(result_ndcg[:, 9].mean(), 5)
    ndcg20 = round(result_ndcg[:, 19].mean(), 5)
    ndcg50 = round(result_ndcg[:, 49].mean(), 5)

    result_recall = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    recall5 = round(result_recall.iloc[:, 4].mean(), 5)
    recall10 = round(result_recall.iloc[:, 9].mean(), 5)
    recall20 = round(result_recall.iloc[:, 19].mean(), 5)
    recall50 = round(result_recall.iloc[:, 49].mean(), 5)

    return recall5, recall10, recall20, recall50, ndcg5, ndcg10, ndcg20, ndcg50, result_ndcg, result_recall

df.describe()
df1 = df[df['sim']<=0.8]
df2 = df[(df['sim']<=0.84) & (df['sim']>0.8)]
df3 = df[(df['sim']<=0.85) & (df['sim']>0.84)]
df4 = df[(df['sim']<=0.86) & (df['sim']>0.85)]
df5 = df[(df['sim']<=0.88) & (df['sim']>0.86)]
df6 = df[(df['sim']<=0.9) & (df['sim']>0.88)]
df7 = df[(df['sim']<=0.95) & (df['sim']>0.9)]
df8 = df[df['sim']>0.95]

res = metrics(np.array([1 for i in range(df1.shape[0])]), df1.iloc[:, 2:].astype(bool))
print(f'n_users:{df1.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df2.shape[0])]), df2.iloc[:, 2:].astype(bool))
print(f'n_users:{df2.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df3.shape[0])]), df3.iloc[:, 2:].astype(bool))
print(f'n_users:{df3.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df4.shape[0])]), df4.iloc[:, 2:].astype(bool))
print(f'n_users:{df4.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df5.shape[0])]), df5.iloc[:, 2:].astype(bool))
print(f'n_users:{df5.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df6.shape[0])]), df6.iloc[:, 2:].astype(bool))
print(f'n_users:{df6.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df7.shape[0])]), df7.iloc[:, 2:].astype(bool))
print(f'n_users:{df7.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')
res = metrics(np.array([1 for i in range(df8.shape[0])]), df8.iloc[:, 2:].astype(bool))
print(f'n_users:{df8.shape[0]}, hit@10:{res[1]}, hit@50:{res[3]}, ndcg@10:{res[5]}, ndcg@50:{res[7]}')

