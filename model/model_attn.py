import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, scatter
from point_transformer_conv_custom import PointTransformerConv_Custom
# from create_graph import create_graph


class PointTrans_Layer(nn.Module):
    def __init__(self, in_channels=3, mid_channels=3, out_channels=3):
        super().__init__()

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

        self.conv = PointTransformerConv_Custom(
            in_channels=in_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn)

    def forward(self, data):
        # put create graph here

        out = self.conv(x=data.x.float(),
                        pos=data.pos.float(),
                        edge_index=data.edge_index)
        data.x = out
        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, perc_down=0.5):
        super().__init__()
        self.perc_points = perc_down
        self.linear = torch.nn.Linear(in_features=in_channels,
                                      out_features=out_channels)
        self.down = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels, out_features=out_channels),
                                        torch.nn.BatchNorm1d(out_channels),
                                        torch.nn.ReLU())

    def forward(self, data):
        data.x = self.down(data.x.float())
        ''' # uniform sampling
        index = np.random.choice(data.x.shape[0],
                                 size=int(
                                     np.round(data.x.shape[0]*self.perc_points)),
                                 replace=False)
        index = np.sort(index)
        '''
        # farthest point sampling

        index = tgnn.pool.fps(
            data.pos, ratio=self.perc_points, batch=data.batch)
        index = index.sort().values

        # pooling
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

    def forward(self, data):
        x_1 = self.pconv(data)
        x_2 = self.downlayer(x_1)
        return x_2


class TransformerGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Enc_block(in_channels=3, out_channels=32, perc_down=0.5)
        self.enc2 = Enc_block(in_channels=32, out_channels=64, perc_down=0.5)
        self.enc3 = Enc_block(in_channels=64, out_channels=128, perc_down=0.5)
        self.enc4 = Enc_block(in_channels=128, out_channels=256, perc_down=0.5)
        self.enc5 = Enc_block(in_channels=256, out_channels=512, perc_down=0.5)

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

    def generate_graph(self, data):
        # initalize graph
        # data.to('cpu')
        data = tg.transforms.KNNGraph(k=16)(data)
        return data

    def forward(self, data):
        # compute graph
        data = self.generate_graph(data)

        # Layers
        x_1 = self.enc1(data)
        x_2 = self.generate_graph(x_1)

        x_3 = self.enc2(x_2)
        x_4 = self.generate_graph(x_3)

        x_5 = self.enc3(x_4)
        x_6 = self.generate_graph(x_5)

        x_7 = self.enc4(x_6)
        x_8 = self.generate_graph(x_7)

        x_9 = self.enc5(x_8)
        x_10 = self.generate_graph(x_9)

        # global_pooling and output head
        x_11 = tgnn.pool.global_mean_pool(x_10.x, x_10.batch)
        pred = self.output_head(x_11)
        return pred
