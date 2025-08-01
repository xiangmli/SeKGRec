import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import pandas as pd
import os
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.pretrained_user_emb = None
        self.pretrained_item_emb = None

        if self.config.get('init_from_pretrain', False):
            dataset_name = self.config['dataset']
            user_emb_path = f"../data/{dataset_name}/user_embeddings.pt"
            item_emb_path = f"../data/{dataset_name}/item_embeddings.pt"

            try:
                user_emb = torch.load(user_emb_path)
                item_emb = torch.load(item_emb_path)
                self.embedding_user.weight.data.copy_(user_emb)
                self.embedding_item.weight.data.copy_(item_emb)
                print(f'use pretrained embeddings from {user_emb_path} and {item_emb_path}')

                if self.config.get('plug_pretrain', 0) != 0:
                    self.pretrained_user_emb = user_emb.clone()
                    self.pretrained_item_emb = item_emb.clone()
                    print(
                        f'Pretrained embeddings saved for plug functionality with weight {self.config["plug_pretrain"]}')

            except FileNotFoundError as e:
                print(f'Pretrained embedding file not found: {e}')
                print('Fall back to normal initialization')
                nn.init.normal_(self.embedding_user.weight, std=0.1)
                nn.init.normal_(self.embedding_item.weight, std=0.1)
                world.cprint('use NORMAL distribution initilizer (fallback)')
        else:
            if self.config.get('plug_pretrain', 0) != 0:
                dataset_name = self.config['dataset']
                user_emb_path = f"../data/{dataset_name}/user_embeddings.pt"
                item_emb_path = f"../data/{dataset_name}/item_embeddings.pt"
                try:
                    self.pretrained_user_emb = torch.load(user_emb_path)
                    self.pretrained_item_emb = torch.load(item_emb_path)
                    print(self.pretrained_user_emb.shape[1])
                    print(self.latent_dim)
                    if self.pretrained_user_emb.shape[1] > self.latent_dim:
                        self.pretrained_user_emb = self.semantic_space_decomposion(self.pretrained_user_emb,
                                                                                   self.latent_dim, self.config)
                        self.pretrained_item_emb = self.semantic_space_decomposion(self.pretrained_item_emb,
                                                                                   self.latent_dim, self.config)
                    print(
                        f'Pretrained embeddings loaded for plug functionality with weight {self.config["plug_pretrain"]}')
                except FileNotFoundError as e:
                    print(f'Pretrained embedding file not found for plug functionality: {e}')
                    self.pretrained_user_emb = None
                    self.pretrained_item_emb = None

            if self.config['pretrain'] == 0:
                nn.init.normal_(self.embedding_user.weight, std=0.1)
                nn.init.normal_(self.embedding_item.weight, std=0.1)
                world.cprint('use NORMAL distribution initilizer')
            else:
                self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
                self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
                print('use pretarined data')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def semantic_space_decomposion(self, language_embs, clipped_dim, config):
        if isinstance(language_embs, torch.Tensor):
            language_embs = language_embs.detach().cpu().numpy()
        self.language_mean = np.mean(language_embs, axis=0)
        cov = np.cov(language_embs - self.language_mean, rowvar=False)

        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        if clipped_dim is None:
            clipped_dim = self.language_dim
        Projection_matrix = U[..., :clipped_dim]

        if config['standardization']:
            Diagnals = np.sqrt(1 / S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals))
        clipped_language_embs = (language_embs - self.language_mean).dot(Projection_matrix)
        return torch.tensor(clipped_language_embs, dtype=torch.float32)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.config.get('plug_pretrain', 0) != 0 and self.pretrained_user_emb is not None:
            plug_weight = self.config['plug_pretrain']
            if self.pretrained_user_emb.device != users_emb.device:
                self.pretrained_user_emb = self.pretrained_user_emb.to(users_emb.device)
                self.pretrained_item_emb = self.pretrained_item_emb.to(items_emb.device)

            users_emb = users_emb + plug_weight * self.pretrained_user_emb
            items_emb = items_emb + plug_weight * self.pretrained_item_emb

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getAllEmbedding(self):
        all_users, all_items = self.computer()
        users_emb_ego = self.embedding_user.weight
        items_emb_ego = self.embedding_item.weight
        return all_users, all_items, users_emb_ego, items_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, config):
        super(Item_Embedding, self).__init__()
        self.item_num = config['n_item']
        self.construct_item_embeddings(emb_pipline, config)

    def construct_item_embeddings(self, emb_pipline, config):
        cliped_language_embs = self.semantic_space_decomposion(config["hidden_dim"], config)
        padding_emb = np.random.rand(cliped_language_embs.shape[1])  # padding ID embedding
        cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])
        self.language_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(cliped_language_embs, dtype=torch.float32),
            freeze=True,  # freeze=True
            padding_idx=self.item_num
        )
        self.init_ID_embedding(self.nullity, config["ID_embs_init_type"], config)
        # self.init_ID_embedding(self.nullity, "zeros")

    def load_language_embeddings(self, directory, scale):
        data = torch.load(directory)  # directory 直接是 pt 文件的路径
        language_embs = data['embeddings']  # 从字典中取出embeddings

        if isinstance(language_embs, torch.Tensor):
            language_embs = language_embs.numpy()

        self.item_num = len(language_embs)
        self.language_dim = language_embs.shape[1]  # 使用shape[1]更安全
        return language_embs * scale

    def init_ID_embedding(self, ID_dim, init_type, config):
        if init_type == "language_embeddings":
            language_embs = self.load_language_embeddings(config["language_embs_path"],
                                                          config["language_embs_scale"])
            if self.language_dim == ID_dim:
                padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
                language_embs = np.vstack([language_embs, padding_emb])
                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            else:
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, config)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
        else:
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num + 1,
                embedding_dim=ID_dim,
            )
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")

    def semantic_space_decomposion(self, clipped_dim, config):
        language_embs = self.load_language_embeddings(config["item_language_embs_path"],
                                                      config["language_embs_scale"])
        if not config["item_frequency_flag"]:
            # The default item distribution is a uniform distribution.
            self.language_mean = np.mean(language_embs, axis=0)
            cov = np.cov(language_embs - self.language_mean, rowvar=False)
        else:
            items_pop = np.load(os.path.join(config["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop * items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs * items_freq, axis=0)
            cov = np.cov((language_embs - self.language_mean) * np.sqrt(items_freq), rowvar=False)
            # raise NotImplementedError("Custom item distribution is not implemented yet.")
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        if config["null_thres"] is not None:
            indices_null = np.where(S <= config["null_thres"])[0]
            self.nullity = len(indices_null)
        elif config["null_dim"] is not None:
            self.nullity = config["null_dim"]
        # print("The Nullity is", self.nullity)
        # self.squared_singular_values = S
        # self.language_bases = U
        if clipped_dim is None:
            clipped_dim = self.language_dim
        if config["cover"]:
            clipped_dim = clipped_dim - self.nullity
        Projection_matrix = U[..., :clipped_dim]

        if config['standardization']:
            Diagnals = np.sqrt(1 / S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals))  # V_{\lamda} into V_1
        clipped_language_embs = (language_embs - self.language_mean).dot(Projection_matrix)
        return clipped_language_embs


class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, config):
        super(Item_Embedding, self).__init__()
        self.item_num = config['n_item']
        self.construct_item_embeddings(emb_pipline, config)

    def construct_item_embeddings(self, emb_pipline, config):
        cliped_language_embs = self.semantic_space_decomposion(config["hidden_dim"], config)
        padding_emb = np.random.rand(cliped_language_embs.shape[1])
        cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])
        self.language_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(cliped_language_embs, dtype=torch.float32),
            freeze=True,
            padding_idx=self.item_num
        )
        self.init_ID_embedding(self.nullity, config["ID_embs_init_type"], config)

    def load_language_embeddings(self, directory, scale):
        data = torch.load(directory)
        language_embs = data['embeddings']

        if isinstance(language_embs, torch.Tensor):
            language_embs = language_embs.numpy()

        self.item_num = len(language_embs)
        self.language_dim = language_embs.shape[1]
        return language_embs * scale

    def init_ID_embedding(self, ID_dim, init_type, config):
        if init_type == "language_embeddings":
            language_embs = self.load_language_embeddings(config["language_embs_path"],
                                                          config["language_embs_scale"])
            if self.language_dim == ID_dim:
                padding_emb = np.random.rand(language_embs.shape[1])
                language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            else:
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, config)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
        else:
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num + 1,
                embedding_dim=ID_dim,
            )
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")

    def semantic_space_decomposion(self, clipped_dim, config):
        language_embs = self.load_language_embeddings(config["item_language_embs_path"],
                                                      config["language_embs_scale"])
        if not config["item_frequency_flag"]:
            self.language_mean = np.mean(language_embs, axis=0)
            cov = np.cov(language_embs - self.language_mean, rowvar=False)
        else:
            items_pop = np.load(os.path.join(config["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop * items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs * items_freq, axis=0)
            cov = np.cov((language_embs - self.language_mean) * np.sqrt(items_freq), rowvar=False)

        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        if config["null_thres"] is not None:
            indices_null = np.where(S <= config["null_thres"])[0]
            self.nullity = len(indices_null)
        elif config["null_dim"] is not None:
            self.nullity = config["null_dim"]

        if clipped_dim is None:
            clipped_dim = self.language_dim
        if config["cover"]:
            clipped_dim = clipped_dim - self.nullity
        Projection_matrix = U[..., :clipped_dim]

        if config['standardization']:
            Diagnals = np.sqrt(1 / S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals))
        clipped_language_embs = (language_embs - self.language_mean).dot(Projection_matrix)
        return clipped_language_embs


class User_Embedding(nn.Module):
    def __init__(self, emb_type, config):
        super(User_Embedding, self).__init__()
        self.emb_type = emb_type

        self.user_num = config['n_user']

        if emb_type == "AF":
            self.construct_user_embeddings(emb_type, config)
        elif emb_type == "ID":
            self.init_user_ID_embedding(config["hidden_dim"], config["ID_embs_init_type"], config)

    def construct_user_embeddings(self, emb_pipeline, config):
        clipped_language_embs = self.semantic_space_decomposition(config["hidden_dim"], config)
        padding_emb = np.random.rand(clipped_language_embs.shape[1])
        clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
        self.language_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(clipped_language_embs, dtype=torch.float32),
            freeze=True,
            padding_idx=self.user_num
        )
        self.init_user_ID_embedding(self.nullity, config["ID_embs_init_type"], config)

    def load_user_language_embeddings(self, directory, scale):
        data = torch.load(directory)
        language_embs = data['embeddings']

        if isinstance(language_embs, torch.Tensor):
            language_embs = language_embs.numpy()

        self.user_num = len(language_embs)
        self.language_dim = language_embs.shape[1]
        return language_embs * scale

    def init_user_ID_embedding(self, ID_dim, init_type, config):
        if init_type == "language_embeddings":
            language_embs = self.load_user_language_embeddings(
                config["user_language_embs_path"],
                config["language_embs_scale"]
            )
            if self.language_dim == ID_dim:
                padding_emb = np.random.rand(language_embs.shape[1])
                language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.user_num
                )
            else:
                clipped_language_embs = self.semantic_space_decomposition(ID_dim, config)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.user_num
                )
        else:
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.user_num + 1,
                embedding_dim=ID_dim,
            )
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
            else:
                raise NotImplementedError("This kind of init for user ID embeddings is not implemented yet.")

    def semantic_space_decomposition(self, clipped_dim, config):
        language_embs = self.load_user_language_embeddings(
            config["user_language_embs_path"],
            config["language_embs_scale"]
        )
        if not config.get("user_frequency_flag", False):
            self.language_mean = np.mean(language_embs, axis=0)
            cov = np.cov(language_embs - self.language_mean, rowvar=False)
        else:
            users_pop = np.load(os.path.join(config["user_language_embs_path"], 'users_pop.npy'))
            users_freq_scale = 1.0 / users_pop.sum()
            users_freq = (users_pop * users_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs * users_freq, axis=0)
            cov = np.cov((language_embs - self.language_mean) * np.sqrt(users_freq), rowvar=False)

        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        if config.get("null_thres") is not None:
            indices_null = np.where(S <= config["null_thres"])[0]
            self.nullity = len(indices_null)
        elif config.get("null_dim") is not None:
            self.nullity = config["null_dim"]

        if clipped_dim is None:
            clipped_dim = self.language_dim
        if config.get("cover", False):
            clipped_dim = clipped_dim - self.nullity
        Projection_matrix = U[..., :clipped_dim]

        if config.get('standardization', False):
            Diagonals = np.sqrt(1 / S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagonals))
        clipped_language_embs = (language_embs - self.language_mean).dot(Projection_matrix)
        return clipped_language_embs





