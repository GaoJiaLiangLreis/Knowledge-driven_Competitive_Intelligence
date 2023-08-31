'''
Spatial-dependency GNN-based Model
'''

import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from scipy.stats import poisson
import numpy as np

os.environ["DGLBACKEND"] = "pytorch"


# --------- 1. spatial GAT layer --------- #
class GATLayer(nn.Module):
    '''
    单层 单头 GAT
    '''
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        e = F.leaky_relu(a)

        # print(edges.data)
        dists = edges.data['dist'].cpu().numpy()
        means = np.mean(dists)
        probs = torch.tensor([poisson.pmf(k=dist//2, mu=means) for dist in dists]).cuda()
        probs = torch.unsqueeze(probs, dim=1)
        # print(probs.shape)
        # print(e.shape)
        e = torch.mul(probs, e)

        return {"e": e}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):
        self.g = g
        # print(h.shape)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    '''
    多头注意力 GAT层
    '''
    def __init__(self, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        self.g = g

        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


# ----------- 2. create model -------------- #
# 构建两层的GAT模型
class Spatial_GraphGAT(nn.Module):
    def __init__(self, in_feat_dim, h_feat_dim, sm_dim, sp_dim, num_heads):
        super(Spatial_GraphGAT, self).__init__()
        # equation (4)
        self.T1 = nn.Linear(sm_dim, in_feat_dim)
        # equation (2)
        self.T2 = nn.Linear(sp_dim, in_feat_dim)
        self.gat_layer1 = MultiHeadGATLayer(in_dim=in_feat_dim * 2, out_dim=h_feat_dim, num_heads=num_heads)
        self.gat_layer2 = MultiHeadGATLayer(in_dim=h_feat_dim * num_heads, out_dim=h_feat_dim, num_heads=1)

    def forward(self, g, sm_feats, sp_feats):
        # equation (5)
        in_feat = torch.cat([self.T1(sm_feats), self.T2(sp_feats)], dim=1)

        h = self.gat_layer1(g, in_feat)
        h = F.elu(h)
        h = self.gat_layer2(g, h)
        return h