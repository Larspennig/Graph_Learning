import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import scatter, softmax
from sklearn.neighbors import NearestNeighbors
# from create_graph import create_graph


def generate_graph(data, device='cpu', k=16):
    # initalize graph
    data = tg.transforms.KNNGraph(k=k)(data)
    return data


class global_attn(nn.Module):
    def __init__(self, channels_in, channels_out, regular_attention=False):
        super(global_attn, self).__init__()
        self.lin_q = nn.Linear(channels_in, channels_out)
        self.lin_k = nn.Linear(channels_in, channels_out)
        self.lin_v = nn.Linear(channels_in, channels_out)

        self.pos = nn.Sequential(nn.Linear(3, channels_out),
                                 nn.BatchNorm1d(channels_out),
                                 nn.ReLU(),
                                 nn.Linear(channels_out, channels_out),
                                 nn.BatchNorm1d(channels_out),
                                 nn.ReLU())
        self.attn = nn.Sequential(nn.Linear(channels_out, channels_out),
                                  nn.BatchNorm1d(channels_out),
                                  nn.ReLU(),
                                  nn.Linear(channels_out, channels_out),
                                  nn.BatchNorm1d(channels_out),
                                  nn.ReLU())
        self.regular_attention = regular_attention

    def forward(self, data):
        # Get global points via farthest point sampling
        perc = 20/data.x[data.batch == 0].shape[0]
        indices = tgnn.pool.fps(data.pos, ratio=perc, batch=data.batch)
        indices = indices.sort().values

        fps_pos = data.pos[indices]
        fps_x = data.x[indices]  # [m, c]
        fps_batch = data.batch[indices]

        edge_index = tgnn.pool.knn(
            fps_pos, data.pos, k=1, batch_x=fps_batch, batch_y=data.batch)
        # aggregate new values for global nodes
        euc_kernel = 1 / \
            (1+5*(data.pos[edge_index[0]] -
             fps_pos[edge_index[1]]).pow(2).sum(dim=1))

        # feat_kernel = torch.exp(data.x[edge_index[0]] @ fps_x[edge_index[1]].T)/torch.exp(data.x[edge_index[0]] @ fps_x[1]).sum()

        # aggregate new positions for gobal nodes
        fps_x = scatter(euc_kernel.unsqueeze(
            1)*data.x[edge_index[0]], edge_index[1], dim=0, reduce='mean')
        fps_pos = scatter(euc_kernel.unsqueeze(
            1)*data.pos[edge_index[0]], edge_index[1], dim=0, reduce='mean')

        x_q = self.lin_q(data.x)  # [n, c]
        x_v, x_k = self.lin_v(fps_x), self.lin_k(fps_x)

        # Expand batch indices for broadcasting
        local_batch_expanded = data.batch.unsqueeze(1)  # Shape: (n, 1)
        global_batch_expanded = fps_batch.unsqueeze(0)  # Shape: (1, m)

        # Create a mask where local and global tokens have the same batch index
        mask = (local_batch_expanded == global_batch_expanded)  # Shape: (n, m)

        # Get indices where the mask is True
        local_indices, global_indices = torch.nonzero(mask, as_tuple=True)

        # aggregate new values for global nodes
        euc_kernel = 1 / \
            (1+5*(data.pos[local_indices] -
             fps_pos[global_indices]).pow(2).sum(dim=1))
        # feat_kernel = data.x[local_indices] * fps_x[global_indices]

        # aggregate new positions for gobal nodes
        fps_x = scatter(euc_kernel.unsqueeze(
            1)*data.x[local_indices], global_indices, dim=0, reduce='mean')

        # Compute positional encoding #TODO: Implement CPE?? This should be way stronger
        delta = self.pos(data.pos[local_indices]-fps_pos[global_indices])

        if self.regular_attention:
            # what to do about the relative pos encoding ....
            attn = softmax(
                (x_q[local_indices] * x_k[global_indices]+delta).sum(dim=1), local_indices)
            x_v = (x_v[global_indices]+delta) * attn.unsqueeze(1)
            x_v = scatter(x_v, local_indices, dim=0, reduce='add')

        else:
            alpha = self.attn(x_q[local_indices]-x_k[global_indices]+delta)
            alpha = softmax(alpha, local_indices)
            x_v = (x_v[global_indices]+delta) * alpha
            x_v = scatter(x_v, local_indices, dim=0, reduce='add')

        return x_v


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
            num_layers=1,
            plain_last=None)

        self.conv = tgnn.PointTransformerConv(
            in_channels=out_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn)

        self.glob_attn = global_attn(out_channels, out_channels)

    def forward(self, data):
        # put create graph here
        data.x = self.linear_in(data.x).relu()
        # local attention
        out = self.conv(x=data.x,
                        pos=data.pos.float(),
                        edge_index=data.edge_index)
        out = self.linear_up(out).relu()
        # global attention
        glob_out = self.glob_attn(data)
        # create skip connection
        data.x = out + data.x + glob_out

        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, grid_size=0.5, device='cpu', subsampling='fps'):
        super().__init__()
        self.grid_size = grid_size
        self.perc_points = 0.5
        self.device = device
        self.linear = torch.nn.Linear(in_features=in_channels,
                                      out_features=out_channels)
        self.down = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels, out_features=out_channels),
                                        torch.nn.BatchNorm1d(out_channels),
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
            data_out = tg.transforms.GridSampling(
                self.grid_size)(max_pooled_data)
        if self.subsampling == 'fps':
            # farthest point sampling
            index = tgnn.pool.fps(
                data.pos, ratio=self.perc_points, batch=data.batch)
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
        # replace with sequential batchnorm and relu
        self.device = device
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
                                               subsampling=config['subsampling'])

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
        x_1 = generate_graph(
            x_1, device=self.config['device'], k=self.config['k_up'])
        x_2 = self.pconv(x_1)
        return x_2


class TransformerGNN_global(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(in_features=3, out_features=32)
        self.pconv_in = PointTrans_Layer(in_channels=32, out_channels=32)

        self.enc1 = Enc_block(in_channels=32, out_channels=64,
                              grid_size=config['grid_size'][0], config=config)
        self.enc2 = Enc_block(in_channels=64, out_channels=128,
                              grid_size=config['grid_size'][1], config=config)
        self.enc3 = Enc_block(in_channels=128, out_channels=256,
                              grid_size=config['grid_size'][2], config=config)
        self.enc4 = Enc_block(in_channels=256, out_channels=512,
                              grid_size=config['grid_size'][3], config=config)

        self.linear_mid = torch.nn.Linear(in_features=512, out_features=512)
        self.pconv_mid = PointTrans_Layer(in_channels=512, out_channels=512)

        self.dec1 = Dec_block(in_channels=512, out_channels=256, config=config)
        self.dec2 = Dec_block(in_channels=256, out_channels=128, config=config)
        self.dec3 = Dec_block(in_channels=128, out_channels=64, config=config)
        self.dec4 = Dec_block(in_channels=64, out_channels=32, config=config)

        self.linear_out = torch.nn.Linear(in_features=32, out_features=13)

        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=config['num_classes']))

    def forward(self, data):
        # first_block
        data.x = data.x.float()
        data.x = self.linear(data.x)
        data = generate_graph(data)
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
