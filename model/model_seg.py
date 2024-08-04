import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, scatter
# from create_graph import create_graph


def generate_graph(data, device='cpu', k=16):
    # initalize graph
    data = tg.transforms.KNNGraph(k=k)(data)
    return data


class PointTrans_Layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.linear_up = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels)
        self.linear_in = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels)
        
        self.attn = tgnn.models.MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            hidden_channels=out_channels,
            num_layers=2,
            plain_last=False)
        self.pos = tgnn.models.MLP(
            in_channels=3,
            out_channels=out_channels,
            hidden_channels=out_channels,
            num_layers=2,
            plain_last=False)
        
        self.conv = tgnn.PointTransformerConv(
            in_channels=out_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn)
        
        #self.norm = tgnn.norm.LayerNorm(out_channels,mode='node')

    def forward(self, data):
        # put create graph here
        data.x = self.linear_in(data.x).relu()
        out = self.conv(x=data.x,
                        pos=data.pos.float(),
                        edge_index=data.edge_index)
        out = self.linear_up(out).relu()

        # create skip connection
        data.x = out + data.x
        data.x = self.norm(data.x)
        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, grid_size=0.5, device='cpu', subsampling = 'fps'):
        super().__init__()
        self.grid_size = grid_size
        self.perc_points = 0.5
        self.device = device
        self.linear = torch.nn.Linear(in_features=in_channels,
                                      out_features=out_channels)
        self.down = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels, out_features=out_channels),
                                        torch.nn.ReLU())
        self.subsampling = subsampling
        

    def forward(self, data):
        # linear projectionlong
        data_up = tg.data.Data(x=self.down(data.x.float()),
                               batch=data.batch.long(), pos=data.pos, y=data.y.long(), edge_index=data.edge_index)
        # pooling and maxpool
        if self.subsampling == 'grid':
            max_pooled_data = tgnn.max_pool_neighbor_x(data_up)
            del max_pooled_data.edge_index
            data_out = tg.transforms.GridSampling(self.grid_size)(max_pooled_data)
        if self.subsampling == 'fps':
            # farthest point sampling
            index = tgnn.pool.fps(data.pos, ratio=self.perc_points, batch=data.batch)
            index = index.sort().values
            # pooling
            max_pooled_data = tgnn.max_pool_neighbor_x(data_up)
            max_pooled_data.x = max_pooled_data.x[index, :]
            max_pooled_data.pos = max_pooled_data.pos[index]
            max_pooled_data.batch = max_pooled_data.batch[index]
            max_pooled_data.y = max_pooled_data.y[index]
            data_out = max_pooled_data
        return data_out


class PointTrans_Layer_up(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device='cuda', k_up=3) -> None:
        super().__init__()
        self.device= device 
        self.k_up = k_up
        self.linear1 = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels)
        self.linear2 = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels)

    def forward(self, data_1, data_2):
        # upstream input
        data_1.x = self.linear1(data_1.x.float())
        # skip connection input
        data_2.x = self.linear2(data_2.x.float())

        # interpolation
        x_int = tg.nn.unpool.knn_interpolate(x=data_1.x,
                                             pos_x=data_1.pos,
                                             pos_y=data_2.pos,
                                             batch_x=data_1.batch,
                                             batch_y=data_2.batch,
                                             k=self.k_up)

        data = tg.data.Data(x=x_int, pos=data_2.pos, batch=data_2.batch)
        return data


class Enc_block(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, config):
        super().__init__()
        self.k_down = config['k_down']
        self.device = config['device']
        self.downlayer = PointTrans_Layer_down(in_channels=in_channels,
                                               out_channels=out_channels,
                                               grid_size=grid_size,
                                               subsampling = config['subsampling'])

        self.pconv = PointTrans_Layer(in_channels=out_channels,
                                      out_channels=out_channels)

    def forward(self, data):
        x_1 = self.downlayer(data)
        x_1 = generate_graph(x_1, device=self.device, k=self.k_down)
        x_2 = self.pconv(x_1)
        return x_2


class Dec_block(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.config = config
        self.uplayer = PointTrans_Layer_up(
            in_channels=in_channels, out_channels=out_channels)
        self.pconv = PointTrans_Layer(
            in_channels=out_channels, out_channels=out_channels)

    def forward(self, data_1, data_2):
        x_1 = self.uplayer(data_1, data_2)
        x_1 = generate_graph(x_1, device=self.config['device'], k=self.config['k_up'])
        x_2 = self.pconv(x_1)
        return x_2


class TransformerGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(in_features=3, out_features=32)
        self.pconv_in = PointTrans_Layer(in_channels=32, out_channels=32)

        self.enc1 = Enc_block(in_channels=32, out_channels=64, grid_size=config['grid_size'][0], config=config)
        self.enc2 = Enc_block(in_channels=64, out_channels=128, grid_size=config['grid_size'][1], config=config)
        self.enc3 = Enc_block(in_channels=128, out_channels=256, grid_size=config['grid_size'][2], config=config)
        self.enc4 = Enc_block(in_channels=256, out_channels=512, grid_size=config['grid_size'][3], config=config)

        self.linear_mid = torch.nn.Linear(in_features=512, out_features=512)
        self.pconv_mid = PointTrans_Layer(in_channels=512, out_channels=512)

        self.dec1 = Dec_block(in_channels=512, out_channels=256, config=config)
        self.dec2 = Dec_block(in_channels=256, out_channels=128, config=config)
        self.dec3 = Dec_block(in_channels=128, out_channels=64, config=config)
        self.dec4 = Dec_block(in_channels=64, out_channels=32, config=config)

        self.linear_out = torch.nn.Linear(in_features=32, out_features=13)

        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=config['num_classes']))
        
    def generate_graph(self, data):
        # initalize graph
        data = tg.transforms.KNNGraph(k=self.config['k_down'])(data)
        return data

    def forward(self, data):
        # first_block
        data.x = data.x.float()
        data.x = self.linear(data.x)
        data = self.generate_graph(data)
        x_1 = self.pconv_in(data)

        # encoder
        x_2 = self.enc1(x_1)
        x_3 = self.enc2(x_2)
        x_4 = self.enc3(x_3)
        x_5 = self.enc4(x_4)

        # mid_layer
        x_5.x = self.linear_mid(x_5.x)
        x_6 = self.pconv_mid(x_5)

        # decoder
        x_7 = self.dec1(x_6, x_4)
        x_8 = self.dec2(x_7, x_3)
        x_9 = self.dec3(x_8, x_2)
        x_10 = self.dec4(x_9, x_1)

        # output
        x_10 = self.output_head(x_10.x.float())
        return x_10
