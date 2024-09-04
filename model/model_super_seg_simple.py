import torch_geometric as tg
import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
# from create_graph import create_graph

from torch_geometric.utils import add_self_loops, scatter, softmax, remove_self_loops
from torch_geometric.nn import PointTransformerConv
from typing import Callable, Optional, Tuple, Union
from torch import Tensor

from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)

DEVICE = 'cpu'

def create_graph(data, k=16):
    # initalize graph
    data = tg.transforms.KNNGraph(k=k)(data)
    return data

class StraightThrough(torch.autograd.Function):
    """
        A custom autograd function that implements the Straight-Through Estimator (STE).
    """
    @staticmethod
    def forward(ctx, alpha, factor):
        ctx.save_for_backward(alpha, factor) 
        return alpha*factor[:,None]

    @staticmethod
    def backward(ctx, grad_output):
        alpha, factor = ctx.saved_tensors
        return grad_output, (alpha*grad_output).sum(dim=1)


class generate_graph(nn.Module):
    def __init__(self, in_channels, k=16):
        super().__init__()
        self.k = k
        self.MLP = tgnn.models.MLP(
            in_channels=in_channels,
            out_channels=20,
            hidden_channels=in_channels,
            num_layers=1,
            plain_last=False)
        self.t = nn.Parameter(torch.tensor([1.0], requires_grad=False))

    def forward(self, data):
        # initalize graph
        emb_g = self.MLP(data.x)

        num_edges = min(data.x[data.batch == 0].shape[0],16)
        data = tg.transforms.KNNGraph(k=16)(data)
        edges_large = tg.nn.knn_graph(
            emb_g, k=num_edges, batch=data.batch, loop=False, flow='source_to_target', cosine=False)
        i = 0
        
        # circumvent error of having more than k neighbors if necessary
        while edges_large.shape[1] != data.edge_index.shape[1]:
            emb_g = emb_g + torch.rand_like(emb_g)*0.001
            edges_large = tg.nn.knn_graph(
                emb_g, k=16, batch=data.batch, loop=False, flow='source_to_target', cosine=False)
            print('repeated points')
            i += 1
            if i > 10:
                raise ValueError('kNN feature graph clould not be constructed')
                break

        # add edge_index with kNN in feature space
        edges_large = edges_large

        # better solution? to make neighbors deterministic?
        rand_scores = torch.rand_like(emb_g) * 0.0001
        emb_g = emb_g + rand_scores.to(DEVICE)

        dist = torch.norm(emb_g[edges_large[0]] - emb_g[edges_large[1]], dim=1)

        # clipping distances to avoid numerical instability
        dist = torch.clamp(dist, min=1e-6, max=5.0)

        # calculate connection probability
        p = torch.exp(-self.t*dist)

        edges_sparse_v = torch.stack([p, edges_large[1, :]], dim=0)
        data.soft_index_i = edges_large
        data.soft_index_v = edges_sparse_v.float()

        data.edge_index = torch.cat(
            [data.soft_index_i, data.edge_index], dim=1)
        # TO DO: remove equal edges from soft index and hard index

        return data


def generate_knn_graph(data, k=16):
    # initalize graph
    data = tg.transforms.KNNGraph(k=k)(data)
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
        edge_index_soft_idx: Tensor,
        edge_index_soft_v: Tensor
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

        self.edge_index_soft_idx =  edge_index_soft_idx 
        self.edge_index_soft_v = edge_index_soft_v

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, edge_index: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int], turn_off_pos_enc=False) -> Tensor:
        edge_index_soft_v = self.edge_index_soft_v
        delta = self.pos_nn(pos_i - pos_j)
        '''
        if turn_off_pos_enc:
            # mask positional encodings for sparse global edges 
            delta[:edge_index_soft_v.shape[1],:] = torch.zeros_like(delta[:edge_index_soft_v.shape[1],:])
        '''
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)

        factor = torch.cat([edge_index_soft_v[0, :], torch.ones(
            edge_index.shape[1]-edge_index_soft_v.shape[1]).to(DEVICE)], dim=0)
        alpha = StraightThrough.apply(alpha, factor)
        #alpha = factor[:, None]*alpha
        alpha = softmax(alpha, index, ptr, size_i)

        return alpha * (x_j + delta)


class PointTrans_Layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, stride=1):
        super().__init__()
        self.stride = stride

        if out_channels % stride != 0:
            raise ValueError('out_channels must be divisible by stride')

        self.linear_up = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels)
        self.linear_in = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels)
        
        self.attn = tgnn.models.MLP(
            in_channels=out_channels,
            out_channels=out_channels // stride,
            hidden_channels=out_channels // stride,
            num_layers=2,
            plain_last=False)
        self.pos = tgnn.models.MLP(
            in_channels=3,
            out_channels=out_channels,
            hidden_channels=out_channels // 2,
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
        data.x = self.bn1(self.linear_in(data.x))
        out = self.conv(x=data.x.float(),
                            pos=data.pos.float(),
                            edge_index=data.edge_index,
                            edge_index_soft_idx=data.soft_index_i,
                            edge_index_soft_v=data.soft_index_v)
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
        self.graph = generate_graph(out_channels)
        
    def forward(self, data):
        # linear projectionlong
        data_up = tg.data.Data(x=self.down(data.x.float()),
                               batch=data.batch.long(), pos=data.pos, y=data.y.long(), edge_index=data.edge_index)
        
        if self.value_down == 1:    
            return self.graph(data_up)
        
        if data_up.edge_index is None:
            data_up = create_graph(data_up)
        
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
        return self.graph(data_out)


class PointTrans_Layer_up(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, special='no', k_up=3) -> None:
        super().__init__()
        self.special = special
        self.k_up = k_up
        self.linear1 = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels)
        self.linear2 = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels)
        self.graph = generate_graph(out_channels)

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
        return self.graph(data)


class TransformerGNN_super_simple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels, channels = config['in_channels'], config['channels']
        value_down = config['value_down']
        subsampling = config['subsampling']
        blocks = config['blocks']
        strides = config['strides']

        self.enc1 = self._make_encoder(blocks = blocks[0], channels=channels[0], value_down=value_down[0], subsampling=subsampling, stride=strides[0])
        self.enc2 = self._make_encoder(blocks = blocks[1], channels=channels[1], value_down=value_down[1], subsampling=subsampling, stride=strides[1])
        self.enc3 = self._make_encoder(blocks = blocks[2], channels=channels[2], value_down=value_down[2], subsampling=subsampling, stride=strides[2])
        self.enc4 = self._make_encoder(blocks = blocks[3], channels=channels[3], value_down=value_down[3], subsampling=subsampling, stride=strides[3])
        self.enc5 = self._make_encoder(blocks = blocks[4], channels=channels[4], value_down=value_down[4], subsampling=subsampling, stride=strides[4])

        self.dec1 = self._make_decoder(blocks = 1, channels = channels[4], special = 'one_input', stride=strides[4])
        self.dec2 = self._make_decoder(blocks = 1, channels = channels[3], stride=strides[3])
        self.dec3 = self._make_decoder(blocks = 1, channels = channels[2], stride=strides[2])
        self.dec4 = self._make_decoder(blocks = 1, channels = channels[1], stride=strides[1])
        self.dec5 = self._make_decoder(blocks = 1, channels = channels[0], stride=strides[0])

        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=config['num_classes']))
    
    def _make_encoder(self, blocks, channels, value_down, subsampling, stride = 1):
        layers = [PointTrans_Layer_down(in_channels=self.in_channels, out_channels=channels, value_down=value_down, subsampling=subsampling)]

        self.in_channels = channels

        for _ in range(blocks):
            layers.append(PointTrans_Layer(in_channels=channels, out_channels=channels, stride=stride))
        return nn.Sequential(*layers)
    
    def _make_decoder(self, blocks, channels, special = 'no', stride = 1):
        layers = [PointTrans_Layer_up(in_channels=self.in_channels, out_channels=channels, special = special)]
        
        self.in_channels = channels

        for _ in range(blocks):
            layers.append(PointTrans_Layer(in_channels=channels, out_channels=channels, stride=stride))
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