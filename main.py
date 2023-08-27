"""
训练、测试main接口
"""

import itertools
import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data
import dgl.function as fn

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from load_dataset import Tourism_KG_Dataset

from Spatial_K_CI import Spatial_GraphGAT

dataset = Tourism_KG_Dataset(city='Beijing')
g = dataset[0]
g = dgl.add_self_loop(g)

print(g)


train_g = g

train_g = dgl.add_self_loop(train_g)

# GPU版本运行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_g = train_g.to(device)






train_pos_g, train_neg_g = dataset[1]
train_pos_g = train_pos_g.to(device)
train_neg_g = train_neg_g.to(device)

test_pos_g, test_neg_g = dataset[2]
test_pos_g = test_pos_g.to(device)
test_neg_g = test_neg_g.to(device)





class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]




class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], -1)
        # print(h.shape)
        # print(F.relu(self.W1(h)).shape)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]




# GAT模型
model = Spatial_GraphGAT(train_g.ndata['semantic_feat'].shape[1], 64,
                 sm_dim=train_g.ndata['semantic_feat'].shape[1], sp_dim=train_g.ndata['spatial_feat'].shape[1])

# You can replace DotPredictor with MLPPredictor.
pred = MLPPredictor(64).to(device)


# pred = DotPredictor()


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)
    if scores.shape != labels.shape:
        scores = torch.squeeze(scores, dim=1)

    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).cpu().numpy()
    return roc_auc_score(labels, scores)


def compute_acc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    scores = F.sigmoid(scores).cpu().cpu().numpy()
    scores = np.around(scores)

    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).cpu().numpy()

    return accuracy_score(labels, scores)


# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=1e-5
)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100000):
    # forward
    model = model.to(device)
    h = model(train_g, train_g.ndata["semantic_feat"], train_g.ndata['spatial_feat']).to(device)
    pos_score = pred(train_pos_g, h).to(device)
    neg_score = pred(train_neg_g, h).to(device)
    loss = compute_loss(pos_score, neg_score).to(device)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 1000 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# ----------- 5. check results ------------------------ #
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))
    print("ACC", compute_acc(pos_score, neg_score))


