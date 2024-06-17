import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, scatter
import time
from model.point_transformer_conv_super import PointTransformerConv_Super
# from create_graph import create_graph


class generate_graph(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.k = 16
        self.MLP = tgnn.models.MLP(
            in_channels=in_channels,
            out_channels=10,
            hidden_channels=in_channels,
            num_layers=2)
        self.t = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, data):
        # initalize graph
        data.to('cpu')
        data = tg.transforms.KNNGraph(k=16)(data)
        data.cuda()
        # add edge_index with kNN in feature space

        # better solution? to make neighbors deterministic?
        emb_g = self.MLP(data.x)
        rand_scores = torch.rand_like(emb_g).cuda() * 0.001
        emb_g = emb_g + rand_scores
        data.soft_index_i = torch.zeros((2, 0), dtype=torch.long).cuda()
        data.soft_index_v = torch.zeros((2, 0), dtype=torch.float).cuda()

        for i in data.batch.unique():
            dist = emb_g[data.batch == i][:, None] - \
                emb_g[data.batch == i][None, :]
            dist = torch.norm(dist, dim=-1)
            # calculate connection probability
            p = torch.exp(-self.t*dist**2)

            # sample k from neighbors with gumbel loss
            gumbel_noise = - \
                torch.log(-torch.log(torch.rand_like(p) + 1e-20) + 1e-20).cuda()
            noisy_logits = torch.log(p + 1e-20) + gumbel_noise

            top_edges_v, top_edges_i = torch.topk(noisy_logits, self.k, dim=0)
            top_edges_v = torch.softmax(top_edges_v, dim=0)
            top_edges_v = top_edges_v / top_edges_v.max(dim=0).values
            min_index = (data.batch == i).nonzero().min()

            data.soft_index_i = torch.cat([data.soft_index_i, torch.stack([top_edges_i.T.flatten()+min_index, torch.arange(
                top_edges_i.shape[1]).repeat_interleave(self.k).cuda()+min_index], dim=0)], dim=1)
            data.soft_index_v = torch.cat([data.soft_index_v, torch.stack([top_edges_v.T.flatten(), torch.arange(
                top_edges_v.shape[1]).repeat_interleave(self.k).cuda()+min_index], dim=0)], dim=1)

        data.edge_index = torch.cat(
            [data.soft_index_i, data.edge_index], dim=1)
        # TO DO: remove equal edges from soft index and hard index
        return data


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

        self.conv = PointTransformerConv_Super(
            in_channels=in_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn)

    def forward(self, data):
        # put create graph here

        out = self.conv(x=data.x.float(),
                        pos=data.pos.float(),
                        edge_index=data.edge_index,
                        edge_index_soft_idx=data.soft_index_i,
                        edge_index_soft_v=data.soft_index_v)
        data.x = out
        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, grid_size=0.5):
        super().__init__()
        self.perc_points = grid_size
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
        
        # farthest point sampling
        index = tgnn.pool.fps(
            data.pos, ratio=self.perc_points, batch=data.batch)
        index = index.sort().values
        '''
        # pooling
        max_pooled_data = tgnn.max_pool_neighbor_x(data)
        '''
        data.x = max_pooled_data.x[index, :]
        data.pos = max_pooled_data.pos[index]
        data.batch = max_pooled_data.batch[index]
        '''
        del data.edge_index
        data.to('cpu')
        data = tg.transforms.GridSampling(self.perc_points)(max_pooled_data)
        data.cuda()
        return data


class Enc_block(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size):
        super().__init__()
        self.pconv = PointTrans_Layer(
            in_channels=in_channels, out_channels=out_channels)
        self.downlayer = PointTrans_Layer_down(
            in_channels=out_channels, out_channels=out_channels, grid_size=grid_size)
        self.generate_graph = generate_graph(in_channels=in_channels)

    def forward(self, data):
        x_1 = self.generate_graph(data)
        x_2 = self.pconv(x_1)
        x_3 = self.downlayer(x_2)
        return x_3


class TransformerGNN(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.enc1 = Enc_block(in_channels=3, out_channels=32, perc_down=0.5)
        self.enc2 = Enc_block(in_channels=32, out_channels=64, perc_down=0.5)
        self.enc3 = Enc_block(in_channels=64, out_channels=128, perc_down=0.5)
        self.enc4 = Enc_block(in_channels=128, out_channels=256, perc_down=0.5)
        self.enc5 = Enc_block(in_channels=256, out_channels=512, perc_down=0.5)
        '''
        self.enc1 = Enc_block(in_channels=3, out_channels=32, grid_size=0.1)
        self.enc2 = Enc_block(in_channels=32, out_channels=64, grid_size=0.15)
        self.enc3 = Enc_block(in_channels=64, out_channels=128, grid_size=0.2)
        self.enc4 = Enc_block(
            in_channels=128, out_channels=256, grid_size=0.3)
        self.enc5 = Enc_block(in_channels=256, out_channels=512, grid_size=0.4)

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

        # Layers
        x_1 = self.enc1(data)
        x_2 = self.enc2(x_1)
        x_3 = self.enc3(x_2)
        x_4 = self.enc4(x_3)
        x_5 = self.enc5(x_4)

        # global_pooling and output head
        x_11 = tgnn.pool.global_mean_pool(x_5.x, x_5.batch)
        pred = self.output_head(x_11)
        return pred
