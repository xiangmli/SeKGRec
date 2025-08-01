import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
import math
from tqdm import tqdm
from torch_geometric.nn import LGConv, GATConv
from torch_geometric.utils import dropout_edge
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class MyLoader(object):
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        self.loadData()

    def loadData(self):
        path = './data/' + self.config['dataset']
        self.train = self._load_ratings(f'{path}/train.txt')
        self.test = self._load_ratings(f'{path}/test.txt')
        self.kg = self._load_kg(f'{path}/kg_final.txt', self.config['inverse_r'])

        self.kg_org = self.kg.copy()

        df = pd.concat((self.train, self.test))
        user_num = df['userid'].max() + 1
        item_num = df['itemid'].max() + 1
        entity_num = 0
        entity_num = max([df['itemid'].max(), self.kg['head'].max(), self.kg['tail'].max()]) + 1

        self.kg_interest = pd.read_csv(f'./User_Intent/{self.config["dataset"]}/user_intent.txt', sep='\t',
                                       names=['uid', 'interest'])
        entity_num = max([entity_num, self.kg_interest['interest'].max() + 1])
        interest_num = self.kg_interest['interest'].max() - self.kg_interest['interest'].min() + 1

        self.INTERACT_RELATION = 0
        self.INTERACT_RELATION_INV = 1

        max_kg_relation = self.kg['relation'].max()

        self.INTEREST_RELATION = max_kg_relation + 1
        self.INTEREST_RELATION_INV = max_kg_relation + 2

        self.train['itemid'] = self.train['itemid'].apply(lambda x: x + user_num)
        self.test['itemid'] = self.test['itemid'].apply(lambda x: x + user_num)

        self.kg_interest['interest'] = self.kg_interest['interest'] + user_num

        self.kg['head'] = self.kg['head'].apply(lambda x: x + user_num)
        self.kg['tail'] = self.kg['tail'].apply(lambda x: x + user_num)
        self.kg_org['head'] = self.kg_org['head'].apply(lambda x: x + user_num)
        self.kg_org['tail'] = self.kg_org['tail'].apply(lambda x: x + user_num)

        kg_interactive = pd.DataFrame({
            'head': self.train['userid'].to_list(),
            'relation': [self.INTERACT_RELATION] * len(self.train),
            'tail': self.train['itemid'].to_list()
        })

        kg_interactive_inv = pd.DataFrame({
            'head': self.train['itemid'].to_list(),
            'relation': [self.INTERACT_RELATION_INV] * len(self.train),
            'tail': self.train['userid'].to_list()
        })

        kg_interest_relations = pd.DataFrame({
            'head': self.kg_interest['uid'].to_list(),
            'relation': [self.INTEREST_RELATION] * len(self.kg_interest),
            'tail': self.kg_interest['interest'].to_list()
        })

        kg_interest_relations_inv = pd.DataFrame({
            'head': self.kg_interest['interest'].to_list(),
            'relation': [self.INTEREST_RELATION_INV] * len(self.kg_interest),
            'tail': self.kg_interest['uid'].to_list()
        })

        relations_to_add = [kg_interactive, kg_interactive_inv, kg_interest_relations, kg_interest_relations_inv]
        self.kg = pd.concat([self.kg] + relations_to_add)

        self.kg_dict = defaultdict(list)
        for row in self.kg.values:
            h_id, r_id, t_id = row
            self.kg_dict[h_id].append((r_id, t_id))

        self.config['users'] = user_num
        self.config['items'] = item_num
        self.config['entities'] = entity_num
        self.config['relations'] = self.kg['relation'].max() + 1
        self.config['interests'] = interest_num

        if self.verbose:
            self.statistic(df, self.kg_org)
        self.prepare_eval()

    def _load_ratings(self, file_name):
        inter_mat = []
        lines = open(file_name, 'r').readlines()
        for l in lines:
            if l == "\n":
                continue
            tmps = l.replace("\n", "").strip()
            inters = [int(i) for i in tmps.split(' ')]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = set(pos_ids)
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
        df = pd.DataFrame(inter_mat, columns=['userid', 'itemid'])
        return df

    def _load_kg(self, file_name, inverse_r=True):
        kg = []
        lines = open(file_name, 'r').readlines()
        for l in lines:
            if l == "\n":
                continue
            tmps = l.replace("\n", "").strip()
            tup = [int(i) for i in tmps.split(' ')]
            kg.append(tup)

        can_triplets_np = np.array(kg)
        can_triplets_np = np.unique(can_triplets_np, axis=0)
        if inverse_r:
            inv_triplets_np = can_triplets_np.copy()
            inv_triplets_np[:, 0] = can_triplets_np[:, 2]
            inv_triplets_np[:, 2] = can_triplets_np[:, 0]
            inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
            can_triplets_np[:, 1] = can_triplets_np[:, 1] + 2
            inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 2
            triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
        else:
            if self.config['use_ckg']:
                can_triplets_np[:, 1] = can_triplets_np[:, 1] + 2
            triplets = can_triplets_np.copy()
        df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])
        self.config['n_triplets'] = len(df)
        self.kg_org = df.copy()
        return df

    def get_cf_loader(self, bs=1024):
        dataset = TrainDataset(self.train, self.rated_dict, self.config['users'], self.config['items'])
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=bs, pin_memory=True), dataset

    def get_eval_data(self, dtype='valid'):
        if dtype == 'valid':
            return self.rated_dict, self.item_dict
        else:
            return self.rated_dict, self.item_dict_test

    def prepare_eval(self):
        self.rated_dict = {}
        rated_group = self.train.groupby('userid')
        for u, v in rated_group:
            self.rated_dict[u] = set(v['itemid'].to_list())

        eval_users = list(self.test['userid'].unique())
        eval_df = self.test
        self.item_dict_test = {}
        item_group = eval_df.groupby('userid')
        for u, v in item_group:
            self.item_dict_test[u] = v['itemid'].values.tolist()
        return

    def statistic(self, df, kg):
        user_num = self.config['users']
        item_num = self.config['items']
        sparse = round(len(df) / (user_num * item_num), 4)
        interaction = len(df)
        tups = len(kg)
        itemset = set(df['itemid'].to_list())
        entityset = set(kg['head'].to_list()) | set(kg['tail'].to_list())
        return


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, rated_dict, user_num, item_num):
        self.pos = df
        self.pos_rec = rated_dict
        self.user_num = user_num
        self.item_num = item_num
        self.train_cf = df.values

    def generate_cf_neg(self, user):
        poss = self.pos_rec[user]
        while True:
            neg = random.randint(self.user_num, self.user_num + self.item_num - 1)
            if neg not in poss:
                break
        return neg

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, index):
        user = self.train_cf[index][0]
        pos = self.train_cf[index][1]
        neg = self.generate_cf_neg(user)
        return user, pos, neg


def init_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break
        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][1]
        relation = pos_triples[pos_triple_idx][0]
        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx, lowest_neg_idx):
    pos_triples = kg_dict[head]
    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break
        tail = np.random.randint(low=lowest_neg_idx, high=highest_neg_idx, size=1)[0]
        if (relation, tail) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails


def generate_kg_batch(kg_dict, batch_size, highest_neg_idx, lowest_neg_idx):
    exist_heads = kg_dict.keys()
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail
        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx, lowest_neg_idx)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


def clean_text(text):
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        sw = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except (ImportError, LookupError) as e:
        sw = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
              'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
              'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
              'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
              'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once'}
        lemmatizer = None

    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    return text
