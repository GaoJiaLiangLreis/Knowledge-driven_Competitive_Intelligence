"""
Link Prediction using Graph Neural Networks
===========================================

In the :doc:`introduction <1_introduction>`, you have already learned
the basic workflow of using GNNs for node classification,
i.e. predicting the category of a node in a graph. This tutorial will
teach you how to train a GNN for link prediction, i.e. predicting the
existence of an edge between two arbitrary nodes in a graph.

By the end of this tutorial you will be able to

-  Build a GNN-based link prediction model.
-  Train and evaluate the model on a small DGL-provided dataset.

(Time estimate: 28 minutes)

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

######################################################################
# Overview of Link Prediction with GNN
# ------------------------------------
#
# Many applications such as social recommendation, item recommendation,
# knowledge graph completion, etc., can be formulated as link prediction,
# which predicts whether an edge exists between two particular nodes. This
# tutorial shows an example of predicting whether a citation relationship,
# either citing or being cited, between two papers exists in a citation
# network.
#
# This tutorial formulates the link prediction problem as a binary classification
# problem as follows:
#
# -  Treat the edges in the graph as *positive examples*.
# -  Sample a number of non-existent edges (i.e. node pairs with no edges
#    between them) as *negative* examples.
# -  Divide the positive examples and negative examples into a training
#    set and a test set.
# -  Evaluate the model with any binary classification metric such as Area
#    Under Curve (AUC).
#
# .. note::
#
#    The practice comes from
#    `SEAL <https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf>`__,
#    although the model here does not use their idea of node labeling.
#
# In some domains such as large-scale recommender systems or information
# retrieval, you may favor metrics that emphasize good performance of
# top-K predictions. In these cases you may want to consider other metrics
# such as mean average precision, and use other negative sampling methods,
# which are beyond the scope of this tutorial.
#
# Loading graph and features
# --------------------------
#
# Following the :doc:`introduction <1_introduction>`, this tutorial
# first loads the Cora dataset.
#


dataset = Tourism_KG_Dataset(city='Beijing')
g = dataset[0]
g = dgl.add_self_loop(g)

print(g)

######################################################################
# Prepare training and testing sets
# ---------------------------------
#
# This tutorial randomly picks 10% of the edges for positive examples in
# the test set, and leave the rest for the training set. It then samples
# the same number of edges for negative examples in both sets.
#

######################################################################
# When training, you will need to remove the edges in the test set from
# the original graph. You can do this via ``dgl.remove_edges``.
#
# .. note::
#
#    ``dgl.remove_edges`` works by creating a subgraph from the
#    original graph, resulting in a copy and therefore could be slow for
#    large graphs. If so, you could save the training and test graph to
#    disk, as you would do for preprocessing.
#

train_g = g

train_g = dgl.add_self_loop(train_g)

# GPU版本运行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_g = train_g.to(device)

######################################################################
# Define a GraphSAGE model
# ------------------------
#
# This tutorial builds a model consisting of two
# `GraphSAGE <https://arxiv.org/abs/1706.02216>`__ layers, each computes
# new node representations by averaging neighbor information. DGL provides
# ``dgl.nn.SAGEConv`` that conveniently creates a GraphSAGE layer.
#

from dgl.nn import GATv2Conv

# ----------- 2. create model -------------- #
# 构建两层的GAT模型
class GraphGAT(nn.Module):
    def __init__(self, in_feat_dim, h_feat_dim, sm_dim, sp_dim):
        super(GraphGAT, self).__init__()
        self.T1 = nn.Linear(sm_dim, in_feat_dim)
        self.T2 = nn.Linear(sp_dim, in_feat_dim)
        self.gat_layer1 = GATv2Conv(in_feat_dim*2, h_feat_dim, num_heads=8)
        self.gat_layer2 = GATv2Conv(h_feat_dim, h_feat_dim, num_heads=8)

    def forward(self, g, sm_feats, sp_feats):
        in_feat = torch.cat([self.T1(sm_feats), self.T2(sp_feats)], dim=1)
        # print(in_feat.shape)
        h = self.gat_layer1(g, in_feat)
        h = F.elu(h)
        h = torch.mean(h, dim=1)
        # print(h.shape)
        h = self.gat_layer2(g, h)
        h = torch.mean(h, dim=1)
        # print(h.shape)
        return h

######################################################################
# The model then predicts the probability of existence of an edge by
# computing a score between the representations of both incident nodes
# with a function (e.g. an MLP or a dot product), which you will see in
# the next section.
#
# .. math::
#
#
#    \hat{y}_{u\sim v} = f(h_u, h_v)
#


######################################################################
# Positive graph, negative graph, and ``apply_edges``
# ---------------------------------------------------
#
# In previous tutorials you have learned how to compute node
# representations with a GNN. However, link prediction requires you to
# compute representation of *pairs of nodes*.
#
# DGL recommends you to treat the pairs of nodes as another graph, since
# you can describe a pair of nodes with an edge. In link prediction, you
# will have a *positive graph* consisting of all the positive examples as
# edges, and a *negative graph* consisting of all the negative examples.
# The *positive graph* and the *negative graph* will contain the same set
# of nodes as the original graph.  This makes it easier to pass node
# features among multiple graphs for computation.  As you will see later,
# you can directly feed the node representations computed on the entire
# graph to the positive and the negative graphs for computing pair-wise
# scores.
#
# The following code constructs the positive graph and the negative graph
# for the training set and the test set respectively.
#

train_pos_g, train_neg_g = dataset[1]
train_pos_g = train_pos_g.to(device)
train_neg_g = train_neg_g.to(device)

test_pos_g, test_neg_g = dataset[2]
test_pos_g = test_pos_g.to(device)
test_neg_g = test_neg_g.to(device)

######################################################################
# The benefit of treating the pairs of nodes as a graph is that you can
# use the ``DGLGraph.apply_edges`` method, which conveniently computes new
# edge features based on the incident nodes’ features and the original
# edge features (if applicable).
#
# DGL provides a set of optimized builtin functions to compute new
# edge features based on the original node/edge features. For example,
# ``dgl.function.u_dot_v`` computes a dot product of the incident nodes’
# representations for each edge.
#


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


######################################################################
# You can also write your own function if it is complex.
# For instance, the following module produces a scalar score on each edge
# by concatenating the incident nodes’ features and passing it to an MLP.
#


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


######################################################################
# .. note::
#
#    The builtin functions are optimized for both speed and memory.
#    We recommend using builtin functions whenever possible.
#
# .. note::
#
#    If you have read the :doc:`message passing
#    tutorial <3_message_passing>`, you will notice that the
#    argument ``apply_edges`` takes has exactly the same form as a message
#    function in ``update_all``.
#


######################################################################
# Training loop
# -------------
#
# After you defined the node representation computation and the edge score
# computation, you can go ahead and define the overall model, loss
# function, and evaluation metric.
#
# The loss function is simply binary cross entropy loss.
#
# .. math::
#
#
#    \mathcal{L} = -\sum_{u\sim v\in \mathcal{D}}\left( y_{u\sim v}\log(\hat{y}_{u\sim v}) + (1-y_{u\sim v})\log(1-\hat{y}_{u\sim v})) \right)
#
# The evaluation metric in this tutorial is AUC.
#

# GraphSAGE模型
# model = GraphSAGE(train_g.ndata["semantic_feat"].shape[1], 64)
# GAT模型
model = GraphGAT(train_g.ndata['semantic_feat'].shape[1], 64,
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


######################################################################
# The training loop goes as follows:
#
# .. note::
#
#    This tutorial does not include evaluation on a validation
#    set. In practice you should save and evaluate the best model based on
#    performance on the validation set.
#

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

# Thumbnail credits: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
