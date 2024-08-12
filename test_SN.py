from lightning.pytorch.callbacks import ModelCheckpoint
from loaders.Sndataloader2 import SNpart_Dataset
from model.GNN_inf_seg import Lightning_GNN
import torch_geometric as tg
import numpy as np
import lightning as pl
import datetime
import yaml
import os
import wandb
import torch

with open('/home/lars/Graph_Learning/configs/config_SNpart.yml', 'r') as f:
    config = yaml.safe_load(f)

config['batch_size'] = 20

# Data setup
dataset_test = SNpart_Dataset(root=config['root'],
                                 split='test')

test_loader = tg.loader.DataLoader(dataset_test,
                                  batch_size=config['batch_size'],
                                  num_workers=2,
                                  shuffle = True)

# Model setup
GNN_model = Lightning_GNN(config=config)
GNN_model.load_state_dict(torch.load('/home/lars/2024-08-12_13.56.12SN_part_blocks_base/epoch=196-train_loss=0.16.ckpt')['state_dict'])
GNN_model.to('cpu')

# Test
trainer = pl.Trainer(max_epochs=1,
                     accelerator='cpu',
                     log_every_n_steps=1)

trainer.test(GNN_model,
            dataloaders=test_loader) 

# Getting dataset level metrics
miou, macc, oa, ious, accs = GNN_model.cm.all_metrics()
print('miou: ', miou)
print('macc: ', macc)
print('oa: ', oa)