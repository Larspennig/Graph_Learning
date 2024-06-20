import torch
import numpy
import os
import yaml
import torch_geometric as tg
import tqdm
from model.GNN_inf import Lightning_GNN
import time

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

GNN_model = Lightning_GNN(config=config)
GNN_model.to('cuda')
GNN_model.load_state_dict(torch.load('model_checkpoints/2024-06-19_14.45.46/epoch=11-train_loss=0.66.ckpt')['state_dict'])
GNN_model.to('cpu')

torch.save(GNN_model.state_dict(), 'GNN_model_params.pt')