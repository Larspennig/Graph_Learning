import torch
import torch_geometric as tg
from Mddataloader import Modelnet40
from model import TransformerGNN
from geometric import Extract_Geometric
import yaml
import numpy as np

# Load array with params from config.yml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
# params = config['params']

# Set up and process Dataset and Dataloader
dataset = Modelnet40(root='data/Modelnet/',
                     classes_yml='classes.yml', split='train')
dataloader = tg.data.DataLoader(dataset, batch_size=1, shuffle=False)

data = next(dataloader.__iter__())


'''
# geometric feature_extraction
geometric = Extract_Geometric(data=data)
geometric.load_to_open3d()
keypoints_iss = geometric.compute_iss_keypoints()
normals = geometric.compute_edges()
'''

# feature extraction still to come

# first test if knn even remotely works
GNN_model = TransformerGNN()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'Model has {count_parameters(GNN_model)} parameters.')


# perform one forward pass
GNN_model(data)
