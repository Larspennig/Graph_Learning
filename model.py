import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, scatter


class PointTrans_Layer(nn.Module):
    def __init__(self, in_channels=3, mid_channels=3, out_channels=3):
        super().__init__()

        self.attn = tgnn.models.MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            hidden_channels=out_channels*2,
            num_layers=2)
        self.pos = tgnn.models.MLP(
            in_channels=3,
            out_channels=out_channels,
            hidden_channels=out_channels*2,
            num_layers=2)

        self.conv = tgnn.PointTransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn)

    def forward(self, data):
        out = self.conv(x=data.x.float(),
                        pos=data.pos.float(),
                        edge_index=data.edge_index)
        data.x = out
        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, perc_down=0.5):
        super().__init__()
        self.perc_points = perc_down
        self.linear = torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels)

    def forward(self, data):
        data.x = self.linear(data.x.float())
        # uniform sampling
        index = np.random.choice(data.x.shape[0], size=int(
            np.round(data.x.shape[0]*self.perc_points)), replace=False)
        index = np.sort(index)
        # farthest point sampling
        # index = tgnn.pool.fps(
        #    data.x, ratio=self.perc_points).unique().sort().values
        max_pooled_data = tgnn.max_pool_neighbor_x(data)

        data.x = max_pooled_data.x[index, :]
        data.pos = max_pooled_data.pos[index]
        data.batch = max_pooled_data.batch[index]

        return data


class Enc_block(nn.Module):
    def __init__(self, in_channels, out_channels, perc_down):
        super().__init__()
        self.pconv = PointTrans_Layer(
            in_channels=in_channels, out_channels=out_channels)
        self.downlayer = PointTrans_Layer_down(
            in_channels=out_channels, out_channels=out_channels, perc_down=perc_down)

        self.layer = 1


class TransformerGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pconv1 = PointTrans_Layer(in_channels=3, out_channels=32)
        self.downlayer1 = PointTrans_Layer_down(
            in_channels=32, out_channels=32)

        self.pconv2 = PointTrans_Layer(in_channels=32, out_channels=64)
        self.downlayer2 = PointTrans_Layer_down(
            in_channels=64, out_channels=64)

        self.pconv3 = PointTrans_Layer(in_channels=64, out_channels=128)
        self.downlayer3 = PointTrans_Layer_down(
            in_channels=128, out_channels=128)

        self.pconv4 = PointTrans_Layer(in_channels=128, out_channels=256)
        self.downlayer4 = PointTrans_Layer_down(
            in_channels=256, out_channels=256)

        self.pconv5 = PointTrans_Layer(in_channels=256, out_channels=512)
        self.downlayer5 = PointTrans_Layer_down(
            in_channels=512, out_channels=512)

        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=2*512),
            torch.nn.BatchNorm1d(num_features=2*512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=2*512, out_features=512),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.Linear(in_features=512, out_features=40),
            torch.nn.ReLU())

    def generate_graph(self, data):
        # initalize graph
        # data.to('cpu')
        data = tg.transforms.KNNGraph(k=16)(data)
        # data.to('cuda')
        return data

    def forward(self, data):
        # compute graph
        data = self.generate_graph(data)

        # Layers
        x_1 = self.pconv1(data)
        x_2 = self.generate_graph(self.downlayer1(x_1))

        x_3 = self.pconv2(x_2)
        x_4 = self.generate_graph(self.downlayer2(x_3))

        x_5 = self.pconv3(x_4)
        x_6 = self.generate_graph(self.downlayer3(x_5))

        x_7 = self.pconv4(x_6)
        x_8 = self.generate_graph(self.downlayer4(x_7))

        x_9 = self.pconv4(x_8)
        x_10 = self.generate_graph(self.downlayer4(x_9))

        # global_pooling and output head
        x_11 = tgnn.pool.global_mean_pool(x_9.x, x_9.batch)
        pred = torch.softmax(self.output_head(x_10).squeeze())
        return pred
