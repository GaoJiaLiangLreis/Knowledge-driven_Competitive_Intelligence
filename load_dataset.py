from dgl.data import DGLDataset
import dgl
import pandas as pd
import networkx as nx
import torch as th
import pickle as pkl
import json


class Tourism_KG_Dataset(DGLDataset):
    def __init__(self, city):
        self.city = city
        super(Tourism_KG_Dataset, self).__init__(name='TKG_dataset_for_CI')

    def process(self):
        node_id_dict = dict()
        node_idx = 0
        bh_graph_nx = nx.read_gexf('data/{}_graph_1.1.gexf'.format(self.city)).to_directed()
        for node in bh_graph_nx.nodes:
            node_id_dict[node] = node_idx
            node_idx += 1
        # print(node_id_dict)
        pkl.dump(node_id_dict, open('data/node_id_dict_{}.pkl'.format(self.city), 'wb'))

        src_ids = []
        dst_ids = []
        edge_weights = []
        edge_dists = []
        for edge in bh_graph_nx.edges:
            src, dst = edge
            weight = bh_graph_nx.edges[edge]['weight']
            dist = bh_graph_nx.edges[edge]['distance']
            src_ids.append(node_id_dict[src])
            dst_ids.append(node_id_dict[dst])
            edge_weights.append(weight)
            edge_dists.append(dist)

        bh_graph_dgl = dgl.graph((src_ids, dst_ids), num_nodes=bh_graph_nx.number_of_nodes())
        bh_graph_dgl.edata['weight'] = th.tensor(edge_weights)
        bh_graph_dgl.edata['dist'] = th.tensor(edge_dists)
        # print(bh_graph_dgl)

        with open('data/embed_Beijing.vec', 'r', encoding='utf-8') as file:
            embed_dict = json.load(file)
        # 对节点特征进行初始化
        features = th.tensor(embed_dict['ent_embeddings.weight'])
        # print(features.shape)

        entity_idx_in_TKG = pkl.load(open('data/entity_idx_dict_Beijing.pkl', 'rb'))
        # print(entity_idx_in_TKG)
        features_map = []
        for poi_id in node_id_dict:
            entity = 'attr/' + poi_id
            if entity in entity_idx_in_TKG:
                ent_idx = entity_idx_in_TKG[entity]['node_idx']
                features_map.append(features[ent_idx].numpy())
        features_map = th.tensor(features_map)
        # print(features_map.shape)

        self.graph = bh_graph_dgl

        # self.graph.ndata["semantic_feat"] = features_map
        self.graph.ndata["semantic_feat"] = features_map
        self.graph.ndata['spatial_feat'] = th.load('data/{}_spatial_context.pt'.format(self.city))

        train_df = pd.read_csv('./data/dataset_{}/train.csv'.format(self.city), sep=',', encoding='utf-8')
        test_df = pd.read_csv('./data/dataset_{}/test.csv'.format(self.city), sep=',', encoding='utf-8')

        train_pos_u = list()
        train_pos_v = list()
        train_neg_u = list()
        train_neg_v = list()
        for record in train_df.values:
            spot_i, spot_j, intensity, competitive = record
            node_i = node_id_dict[str(int(spot_i))]
            node_j = node_id_dict[str(int(spot_j))]
            if competitive:
                train_pos_u.append(node_i)
                train_pos_v.append(node_j)
            else:
                train_neg_v.append(node_i)
                train_neg_u.append(node_j)
        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.graph.num_nodes())
        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.graph.num_nodes())

        test_pos_u = list()
        test_pos_v = list()
        test_neg_u = list()
        test_neg_v = list()
        for record in test_df.values:
            spot_i, spot_j, intensity, competitive = record
            node_i = node_id_dict[str(int(spot_i))]
            node_j = node_id_dict[str(int(spot_j))]
            if competitive:
                test_pos_u.append(node_i)
                test_pos_v.append(node_j)
            else:
                test_neg_u.append(node_i)
                test_neg_v.append(node_j)
        self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.graph.num_nodes())
        self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.graph.num_nodes())

        all_nodes = self.graph.nodes()
        all_u = []
        all_v = []
        for u in all_nodes:
            for v in all_nodes:
                all_u.append(u)
                all_v.append(v)
        self.graph_all_pairs = dgl.graph((all_u, all_v), num_nodes=self.graph.num_nodes())

    def __getitem__(self, i):
        if i == 0:
            return self.graph
        elif i == 1:
            return self.train_pos_g, self.train_neg_g
        elif i == 2:
            return self.test_pos_g, self.test_neg_g
        elif i == 3:
            return self.graph_all_pairs

    def __len__(self):
        return 4


if __name__ == '__main__':
    dataset = Tourism_KG_Dataset(city='北京')