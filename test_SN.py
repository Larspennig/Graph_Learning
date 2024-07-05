from lightning.pytorch.callbacks import ModelCheckpoint
from loaders.Sndataloader import SNpart_Dataset
from model.GNN_inf_seg import Lightning_GNN
import torch_geometric as tg
import numpy as np
import lightning as pl
import datetime
import yaml
import os
import wandb
import torch

with open('configs/config_SNpart.yml', 'r') as f:
    config = yaml.safe_load(f)

config['batch_size'] = 10

# Data setup
dataset_test = SNpart_Dataset(root=config['root'],
                                 split='test')

test_loader = tg.loader.DataLoader(dataset_test,
                                  batch_size=config['batch_size'],
                                  num_workers=2)

# Model setup
GNN_model = Lightning_GNN(config=config)
GNN_model.load_state_dict(torch.load('/home/lars/SN_outputs/output_SN_standard/2024-07-03_19.53.29/epoch=41-train_loss=0.22.ckpt')['state_dict'])
GNN_model.to('cpu')

# Test
trainer = pl.Trainer(max_epochs=1,
                     accelerator='cpu',
                     log_every_n_steps=1)

trainer.test(GNN_model,
            dataloaders=test_loader) 