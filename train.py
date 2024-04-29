import numpy as np
import lightning as pl
import datetime
import yaml
import os

import torch_geometric as tg
from sklearn.model_selection import train_test_split

from model.model import TransformerGNN
from geometric import Extract_Geometric
from Mddataloader import Modelnet40
import pytorch_lightning as pl


# Load array with params from config.yml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Data setup
dataset = Modelnet40(root=config['root'],
                     classes_yml='classes.yml',
                     split=config['split'])

train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                      test_size=0.2,
                                      shuffle=True)
train_loader = tg.loader.DataLoader(dataset[train_idx])
val_loader = tg.loader.DataLoader(dataset[val_idx])


'''
dataloader = tg.loader.DataLoader(dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False)

data = next(dataloader.__iter__())

# geometric feature_extraction
geometric = Extract_Geometric(data=data)
geometric.load_to_open3d()
keypoints_iss = geometric.compute_iss_keypoints()
normals = geometric.compute_edges()
'''

# Model setup
GNN_model = TransformerGNN()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'Model has {count_parameters(GNN_model)} parameters.')

# Setup output dir
run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
output_dir = os.path.join(config['checkpoints'], run_time)
