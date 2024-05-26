import numpy as np
import torch
import torch_geometric as tg
from scipy.stats import multivariate_normal


class Skewed_Gauss_Graph():

    def __init__(self, model, enc, k=16):
        self.model = model
        self.enc = enc
        self.k = k

    def compute_attention(self, x_i, pos_i, x, pos, model, enc):
        with torch.no_grad():
            # compute masked attention matrix

            enc_layer = getattr(model.model, enc)
            Attn_MLP = enc_layer.pconv.attn
            Pos_MLP = enc_layer.pconv.pos

            MLP_src = enc_layer.pconv.conv.lin_src
            MLP_dst = enc_layer.pconv.conv.lin_dst

            # compute attention
            x_src = Attn_MLP(MLP_src(x))
            x_dest = Attn_MLP(MLP_dst(x_i).view(1, -1))

            # create NxNxnum_features difference matrices
            x_d = (x_dest[:, None] - x_src[None])
            pos_d = (pos_i[:, None].view(1, -1) - pos[None])

            # view matrices
            x_n = x_d.view(-1, x_src.shape[-1])
            pos_n = pos_d.view(-1, pos_d.shape[-1])

            # create final
            pos_enc = Pos_MLP(pos_n)
            attn = Attn_MLP(x_n + pos_enc)

            # to read row_wise Attention scores of node i to node j
            attn = torch.mean(attn, dim=-1)
            attn = torch.nn.Softmax(dim=-1)(attn)

        return attn.squeeze()

    def skew_gaussian_3d(self, x, factor=1):
        init = np.array([1, 0, 0])
        norm_x = np.linalg.norm(x)
        x = x/norm_x
        axis = np.cross(init, x)
        angle = np.arccos(np.dot(x, init))

        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis
        R = np.array([[t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
                      [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
                      [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c]])

        # should the total variance stay the same?

        S = np.array([[1+norm_x*factor, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        cov = R@S@S.T@R.T

        return cov

    def create_graph(self, data, factor=8):
        # start by defining KNN_Graph
        data = tg.transforms.KNNGraph(k=self.k)(data)

        # compute attn scores to egde_index per sample

        edge_index_n = torch.zeros_like(data.edge_index)
        self.model.eval()
        for i in range(data.x.shape[0]):
            # get attention scores
            # check if this is correct
            edge_index_i = data.edge_index[:, data.edge_index[1] == i]
            x = data.x[edge_index_i[0, :], :]
            pos = data.pos[edge_index_i[0, :], :]

            attn_scores = self.compute_attention(
                data.x[i, :], data.pos[i, :], x, pos, self.model, self.enc)

            # get mean direction
            pos_knn = data.pos[edge_index_i[0, :], :]
            pos_diff = pos_knn - data.pos[i]
            diff_mean = (pos_diff*attn_scores[:, None]).mean(0)

            # skew Gaussian
            cov = self.skew_gaussian_3d(diff_mean, factor=factor)

            prob = multivariate_normal.pdf(data.pos, mean=data.pos[i], cov=cov)

            # take most probable k indices
            indices = torch.argsort(torch.tensor(
                prob), descending=True)[:self.k+1]
            indices.sort()

            # drop self loop
            indices = indices[indices != i]

            # compute new edge_index
            edge_index_n[:, data.edge_index[1] ==
                         i] = torch.stack([indices, torch.full_like(indices, i)], dim=0)
        self.model.train()
        return edge_index_n


class Delauney_Graph():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data

    def create_delauney_graph(data, model, batch):
        # create graph
        return 1


class Stratified_Graph():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data

    def create_graph(self, data):
        # create graph by combining ISS keypoints and KNN
        return 1


class Glob_Attn_Graph():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data

    def create_graph(self, data):
        # create graph by computing global attention and top k Gumbel sampling
        return 1


class Loc_Attn_Graph():
    def __init__(self, data, model, enc):
        self.model = model
        self.enc = enc
        self.data = data

    def create_graph(self, data):
        # create graph by computing local attention and top k Gumbel sampling
        return 1
