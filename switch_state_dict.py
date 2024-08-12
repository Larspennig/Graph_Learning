import torch
import numpy
import os
import yaml
import torch_geometric
import tqdm
from model.GNN_inf_seg import Lightning_GNN
import time

with open('configs/config_SNpart.yml', 'r') as f:
    config = yaml.safe_load(f)

GNN_model = Lightning_GNN(config=config)
GNN_model.eval()
example1 = torch.load('/home/lars/Graph_Learning/data/shapenet_core/processed/02773838/1b9ef45fefefa35ed13f430b2941481.pt')
example1.batch = torch.ones(example1.num_nodes)
GNN_model(example1)

a = 1
'''
GNN_model.to('cuda')
GNN_model.load_state_dict(torch.load('model_checkpoints/2024-06-19_14.45.46/epoch=11-train_loss=0.66.ckpt'))
GNN_model.to('cpu')

torch.save(GNN_model.state_dict(), 'GNN_model_params.pt')
'''