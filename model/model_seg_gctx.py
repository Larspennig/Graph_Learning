import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch_geometric.utils import scatter, softmax, add_self_loops,remove_self_loops
from torch_geometric.nn import PointTransformerConv
from typing import Callable, Optional, Tuple, Union
from torch import Tensor
# from create_graph import create_graph

from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
RANDOM_CONNECTIONS = False
AGGREGATION = 'kernel'


def generate_graph(data, k=16):
    # initalize graph
    data = tg.transforms.KNNGraph(k=k)(data)
    if RANDOM_CONNECTIONS:
        for idx, sample in enumerate(data.batch.unique()):
            # find minimum idx with batch == sample
            min_idx = torch.where(data.batch == sample)[0].min()
            max_idx = torch.where(data.batch == sample)[0].max()
            target = torch.arange(min_idx, max_idx+1).repeat(8).to(data.x.device)
            source = torch.randint(min_idx, max_idx+1, target.shape).to(data.x.device)
            
            edge_index = torch.stack([source, target], dim=0)
            data.edge_index = torch.cat([data.edge_index, edge_index], dim=1)
    return data

class Custom_Transformer(PointTransformerConv):
    def __init__(self, in_channels, out_channels, pos_nn, attn_nn, stride):
        super().__init__(in_channels = in_channels,
                 out_channels = out_channels, pos_nn = pos_nn,
                 attn_nn = attn_nn)
        self.stride = stride
    ''''
    Method overwritten from PointTransformerConv torch_geometric class to account for grouped attention via strides
    
    '''
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
        ) -> Tensor:

        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        delta = self.pos_nn(pos_i - pos_j)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        if self.stride is not None:
            alpha = alpha.repeat(1, self.stride)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha * (x_j + delta)



class glob2loc(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(glob2loc, self).__init__()
        self.pos_d_l = nn.Sequential(nn.Linear(3, channels_out),
                        nn.BatchNorm1d(channels_out),
                        nn.ReLU(),
                        nn.Linear(channels_out, channels_out),
                        nn.BatchNorm1d(channels_out),
                        nn.ReLU())

        self.feat_mlp_loc = nn.Sequential(nn.Linear(channels_out, channels_out),
                                  nn.BatchNorm1d(channels_out),
                                  nn.ReLU(),
                                  nn.Linear(channels_out, channels_out),
                                  nn.BatchNorm1d(channels_out),
                                  nn.ReLU())
        
        self.linear_q = nn.Linear(channels_in, channels_out)
        self.linear_k = nn.Linear(channels_in, channels_out)
        self.linear_v = nn.Linear(channels_in, channels_out)
        
    def forward(self, data, edge_index, fps_pos):
        '''
        delta_feat = self.linear_k(data.x)[edge_index[0]] - scatter(self.linear_q(data.x)[edge_index[0]], edge_index[1], dim=0, reduce='mean')[edge_index[1]]

        pos_enc = self.pos_d_l(fps_pos[edge_index[1]]-data.pos[edge_index[0]])
        attn_loc = softmax(self.feat_mlp_loc(delta_feat+pos_enc), edge_index[1])

        fps_n_x = scatter(attn_loc*self.linear_v(data.x)[edge_index[0]], edge_index[1], dim=0, reduce='mean')
        
        '''
        if AGGREGATION.lower() == 'kernel':
            delt_pos = fps_pos[edge_index[1]] - data.pos[edge_index[0]]
            euc_kernel = 1 / (1+5*delt_pos.pow(2).sum(dim=1))
            weight = softmax(euc_kernel, edge_index[1])
            fps_n_x = scatter(weight.unsqueeze(1)*data.x[edge_index[0]], edge_index[1], dim=0, reduce='mean')

        elif AGGREGATION.lower() == 'attention':
            delta_feat = data.x[edge_index[0]] - scatter(data.x[edge_index[0]], edge_index[1], dim=0, reduce='mean')[edge_index[1]]

            pos_enc = self.pos_d_l(fps_pos[edge_index[1]]-data.pos[edge_index[0]])
            attn_loc = softmax(self.feat_mlp_loc(delta_feat+pos_enc), edge_index[1])

            fps_n_x = scatter(attn_loc*data.x[edge_index[0]]*pos_enc, edge_index[1], dim=0, reduce='mean')
        else:
            raise ValueError('Specify valid aggregation method')
        
        return fps_n_x



class global_attn(nn.Module):
    """
    This implements a gobal attention module with the PointTransformer attention mechanism
    """
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
        self.glob2loc = glob2loc(channels_in, channels_out)

    def forward(self, data):
        # Get global points via farthest point sampling
        perc = 15/data.x[data.batch == 0].shape[0]
        indices = tgnn.pool.fps(data.pos, ratio=perc, batch=data.batch)
        indices = indices.sort().values

        fps_pos = data.pos[indices]
        fps_x = data.x[indices]  # [m, c]
        fps_batch = data.batch[indices]

        ### Local 2 Global
        edge_index = tgnn.pool.knn(
            fps_pos, data.pos, k=1, batch_x=fps_batch, batch_y=data.batch)
        
        fps_n_x = self.glob2loc(data, edge_index, fps_pos)
        
        '''
        # aggregate new values for global nodes
        euc_kernel = 1 / \
            (1+5*(data.pos[edge_index[0]] -
             fps_pos[edge_index[1]]).pow(2).sum(dim=1))
        
        # aggregate new features for gobal nodes
        fps_x = scatter(euc_kernel.unsqueeze(
            1)*data.x[edge_index[0]], edge_index[1], dim=0, reduce='mean')
        '''
        ### Global 2 Local
        x_q = self.lin_q(data.x)  # [n, c]
        x_v, x_k = self.lin_v(fps_n_x), self.lin_k(fps_n_x)

        # Expand batch indices for broadcasting
        local_batch_expanded = data.batch.unsqueeze(1)  # Shape: (n, 1)
        global_batch_expanded = fps_batch.unsqueeze(0)  # Shape: (1, m)

        # Create a mask where local and global tokens have the same batch index
        mask = (local_batch_expanded == global_batch_expanded)  # Shape: (n, m)

        # Get indices where the mask is True
        local_indices, global_indices = torch.nonzero(mask, as_tuple=True)

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
    


class GlobalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, regular_attention=False):
        super().__init__()

        self.lin_out = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        self.glob_attn = global_attn(in_channels, out_channels, regular_attention)
        self.bn = nn.BatchNorm1d(out_channels)


    def forward(self, data):
        out = self.glob_attn(data)
        out = self.bn(self.lin_out(out)).relu()
        # create skip connection
        data.x = out + data.x
        return data


class Glob_Loc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.glob = GlobalAttention(in_channels, out_channels)
        self.loc = PointTrans_Layer(in_channels, out_channels)
        self.param = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

    def forward(self, data):
        x_glob = self.glob(data)
        x_loc = self.loc(data)

        x_loc.x = x_loc.x*self.param.sigmoid() + x_glob.x*(1-self.param.sigmoid())

        return x_loc


class PointTrans_Layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, stride=1):
        super().__init__()

        self.stride = stride

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
        
        self.conv = Custom_Transformer(
            in_channels=out_channels,
            out_channels=out_channels,
            pos_nn=self.pos,
            attn_nn=self.attn,
            stride=stride)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, data):
        # put create graph here
        data.x = self.bn1(self.linear_in(data.x)).relu()
        out = self.conv(x=data.x,
                        pos=data.pos.float(),
                        edge_index=data.edge_index)
        out = self.bn2(self.linear_up(out)).relu()

        # create skip connection
        data.x = out + data.x
        return data


class PointTrans_Layer_down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, value_down=0.5, subsampling = 'fps'):
        super().__init__()
        self.value_down = value_down
        self.down = torch.nn.Sequential(torch.nn.Linear(in_features=in_channels, out_features=out_channels),
                                        torch.nn.ReLU())
        self.subsampling = subsampling
        
    def forward(self, data):
        # linear projectionlong
        data_up = tg.data.Data(x=self.down(data.x.float()),
                               batch=data.batch.long(), pos=data.pos, y=data.y.long(), edge_index=data.edge_index)
        
        if self.value_down == 1:    
            return generate_graph(data_up)
        
        if data_up.edge_index is None:
            data_up = generate_graph(data_up)
        
        # pooling and maxpool
        if self.subsampling == 'grid':
            max_pooled_data = tgnn.max_pool_neighbor_x(data_up)
            del max_pooled_data.edge_index
            data_out = tg.transforms.GridSampling(self.value_down)(max_pooled_data)
        if self.subsampling == 'fps':
            # farthest point sampling
            index = tgnn.pool.fps(data.pos, ratio=self.value_down, batch=data.batch)
            index = index.sort().values
            # pooling
            max_p = tgnn.max_pool_neighbor_x(data_up)
            max_p.x, max_p.pos, max_p.batch, max_p.y = max_p.x[index], max_p.pos[index], max_p.batch[index], max_p.y[index]
            data_out = max_p
        return generate_graph(data_out)


class PointTrans_Layer_up(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, special='no', k_up=3) -> None:
        super().__init__()
        self.special = special
        self.k_up = k_up
        self.linear1 = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels)
        self.linear2 = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels)

    def forward(self, data):
        # upstream input
        # skip connection input
        if self.special == 'one_input':
            data.x = self.linear1(data.x.float())
            return data
        
        data_1, data_2 = data
        data_1.x = self.linear1(data_1.x.float())
        data_2.x = self.linear2(data_2.x.float())

        # interpolation
        x_int = tg.nn.unpool.knn_interpolate(x=data_1.x,
                                             pos_x=data_1.pos,
                                             pos_y=data_2.pos,
                                             batch_x=data_1.batch,
                                             batch_y=data_2.batch,
                                             k=self.k_up)

        data = tg.data.Data(x=x_int+data_2.x, pos=data_2.pos, batch=data_2.batch)
        return generate_graph(data)


class TransformerGNN_global(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels, channels = config['in_channels'], config['channels']
        value_down = config['value_down']
        subsampling = config['subsampling']
        blocks = config['blocks']

        self.enc1 = self._make_encoder(blocks = blocks[0], channels=channels[0], value_down=value_down[0], subsampling=subsampling)
        self.enc2 = self._make_encoder(blocks = blocks[1], channels=channels[1], value_down=value_down[1], subsampling=subsampling)
        self.enc3 = self._make_encoder(blocks = blocks[2], channels=channels[2], value_down=value_down[2], subsampling=subsampling)
        self.enc4 = self._make_encoder(blocks = blocks[3], channels=channels[3], value_down=value_down[3], subsampling=subsampling)
        self.enc5 = self._make_encoder(blocks = blocks[4], channels=channels[4], value_down=value_down[4], subsampling=subsampling)

        self.dec1 = self._make_decoder(blocks = 1, channels = channels[4], special = 'one_input')
        self.dec2 = self._make_decoder(blocks = 1, channels = channels[3])
        self.dec3 = self._make_decoder(blocks = 1, channels = channels[2])
        self.dec4 = self._make_decoder(blocks = 1, channels = channels[1])
        self.dec5 = self._make_decoder(blocks = 1, channels = channels[0])

        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=config['num_classes']))
    
    def _make_encoder(self, blocks, channels, value_down, subsampling):
        layers = [PointTrans_Layer_down(in_channels=self.in_channels, out_channels=channels, value_down=value_down, subsampling=subsampling)]

        self.in_channels = channels

        for idx in range(blocks):
            if idx == 0:
                layers.append(Glob_Loc(in_channels=channels, out_channels=channels))
            else:
                layers.append(PointTrans_Layer(in_channels=channels, out_channels=channels))
        return nn.Sequential(*layers)
    
    def _make_decoder(self, blocks, channels, special = 'no'):
        layers = [PointTrans_Layer_up(in_channels=self.in_channels, out_channels=channels, special = special)]
        
        self.in_channels = channels

        for idx in range(blocks):
            if idx == 0:
                layers.append(Glob_Loc(in_channels=channels, out_channels=channels))
            else:
                layers.append(PointTrans_Layer(in_channels=channels, out_channels=channels))
        return nn.Sequential(*layers)

    def forward(self, data):
        x_1 = data
        # encoder
        x_2 = self.enc1(x_1)
        x_3 = self.enc2(x_2)
        x_4 = self.enc3(x_3)
        x_5 = self.enc4(x_4)
        x_6 = self.enc5(x_5)

        # decoder
        x_7 = self.dec1(x_6)
        x_8 = self.dec2((x_7, x_5))
        x_9 = self.dec3((x_8, x_4))
        x_10 = self.dec4((x_9, x_3))
        x_11 = self.dec5((x_10, x_2))

        # output
        x_11 = self.output_head(x_11.x.float())
        return x_11