import numpy as np
import torch


class Attn_Graph_Sampled():

    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data

    def compute_attention(data, model, enc):

        # compute masked attention matrix
        Attn_MLP = model.model.enc1.pconv.attn
        Pos_MLP = model.model.enc1.pconv.pos

        MLP_src = model.model.enc1.pconv.conv.lin_src
        MLP_dst = model.model.enc1.pconv.conv.lin_dst

        # compute attention
        x_src = Attn_MLP(MLP_src(data.x))
        x_dest = Attn_MLP(MLP_dst(data.x))

        # create NxNxnum_features difference matrices
        x_d = (x_dest[:, None] - x_src[None])
        pos_d = (data.pos[:, None] - data.pos[None])

        # view matrices
        x_n = x_d.view(-1, x_src.shape[-1])
        pos_n = pos_d.view(-1, pos_d.shape[-1])

        # create final
        pos_enc = Pos_MLP(pos_n)
        x_enc = Attn_MLP(x_n + pos_enc)

        # create NxN attention matrix
        attn = x_enc.view(x_src.shape[0], x_dest.shape[0], -1)

        # to read row_wise Attention scores of node i t node j
        attn = torch.mean(attn, dim=-1)
        attn = torch.nn.Softmax(dim=1)(attn)

        return attn

    def compute_edge_index(data, attn):
        edge_index = 1
        return edge_index

    def create_graph(data, model, enc, batch, k=16):
        # create graph
        att = att(data, model, enc)
        # extract functions from model

        # create graph from attention
        return 1

    def create_delauney_graph(data, model, batch):
        # create graph
        return 1


class Delauney_Graph():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data


class Stratified_Graph():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data


class two_level_KNN():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data
