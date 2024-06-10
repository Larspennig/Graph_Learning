import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, scatter
# from create_graph import create_graph


def generate_graph(data):
    # initalize graph
    # data.to('cpu')
    data = tg.transforms.KNNGraph(k=16)(data)
    return data


class PointTrans_Layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.linear_up = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels)

        self.attn = tgnn.models.MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            hidden_channels=out_channels,
            num_layers=2)
        self.pos = tgnn.models.MLP(
            in_channels=3,
            out_channels=out_channels,
            hidden_channels=out_channels,
            num_layers=2)

        self.conv = tgnn.PointTransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn)

    def forward(self, data):
        # put create graph here

        out = self.conv(x=data.x.float(),
                        pos=data.pos.float(),
                        edge_index=data.edge_index)
        out = self.linear_up(out)

        # create skip connection
        data.x = out + data.x
        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, grid_size=0.5):
        super().__init__()
        self.grid_size = grid_size
        self.linear = torch.nn.Linear(in_features=in_channels,
                                      out_features=out_channels)
        self.down = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels, out_features=out_channels),
                                        torch.nn.BatchNorm1d(out_channels),
                                        torch.nn.ReLU())

    def forward(self, data):
        # linear projection
        data_up = tg.data.Data(x=self.down(data.x.float()),
                               batch=data.batch, pos=data.pos, y=data.y, edge_index=data.edge_index)
        # pooling and maxpool
        max_pooled_data = tgnn.max_pool_neighbor_x(data_up)
        del max_pooled_data.edge_index
        data_out = tg.transforms.GridSampling(self.grid_size)(max_pooled_data)
        return data_out


class Enc_block(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size):
        super().__init__()
        self.downlayer = PointTrans_Layer_down(in_channels=in_channels,
                                               out_channels=out_channels,
                                               grid_size=grid_size)

        self.pconv = PointTrans_Layer(in_channels=out_channels,
                                      out_channels=out_channels)

    def forward(self, data):
        x_1 = self.downlayer(data)
        x_1 = generate_graph(x_1)
        x_2 = self.pconv(x_1)
        return x_2


class TransformerGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=32)
        self.pconv_in = PointTrans_Layer(in_channels=32, out_channels=32)

        self.enc1 = Enc_block(in_channels=32, out_channels=64, grid_size=0.1)
        self.enc2 = Enc_block(in_channels=64, out_channels=128, grid_size=0.15)
        self.enc3 = Enc_block(in_channels=128, out_channels=256, grid_size=0.2)
        self.enc4 = Enc_block(in_channels=256, out_channels=512, grid_size=0.4)

        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=512, out_features=40),
            torch.nn.BatchNorm1d(num_features=40),
            torch.nn.ReLU())
        # torch.nn.Dropout(p=0.5),
        # torch.nn.Linear(in_features=512, out_features=40))
        # torch.nn.Softmax(dim=1))

    def forward(self, data):
        # compute graph
        data.x = data.x.float()
        data.x = self.linear(data.x)
        data = generate_graph(data)
        x_1 = self.pconv_in(data)

        # encoder
        x_2 = self.enc1(x_1)
        x_3 = self.enc2(x_2)
        x_4 = self.enc3(x_3)
        x_5 = self.enc4(x_4)

        # global_pooling and output head
        x_11 = tgnn.pool.global_mean_pool(x_5.x, x_5.batch)
        pred = self.output_head(x_11)
        return pred
