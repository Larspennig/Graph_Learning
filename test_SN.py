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

config['batch_size'] = 10

# Data setup
dataset_test = SNpart_Dataset(root=config['root'],
                                 split='test')

test_loader = tg.loader.DataLoader(dataset_test,
                                  batch_size=config['batch_size'],
                                  num_workers=2,
                                  shuffle = False)

# Model setup
GNN_model = Lightning_GNN(config=config).cuda()
GNN_model.load_state_dict(torch.load('/home/lars/2024-08-13_10.21.20SN_part_blocks_base/epoch=199-train_loss=0.15.ckpt')['state_dict'])
GNN_model.to('cpu')

# Test
trainer = pl.Trainer(max_epochs=1,
                     accelerator='cpu',
                     log_every_n_steps=1,)

trainer.test(GNN_model,
            dataloaders=test_loader)

# Compute all final metrics
print('FINAL METRICS')
# Get instance metrics
instance_miou = trainer.logged_metrics['test_miou']
print('instance_miou: ', instance_miou.item())
# Getting category metrics
cat_mious = GNN_model.cat_ious
all_ious = []
for key in cat_mious.keys():
    cat_mious[key]['miou'] = torch.mean(torch.stack(cat_mious[key]['ious']))
    all_ious.append(cat_mious[key]['miou'])
    category = dataset_test.number2name[key]
cat_miou = torch.mean(torch.stack(all_ious))
print('cat_miou: ', cat_miou.item())
# Getting dataset level metrics
miou, macc, oa, ious, accs = GNN_model.cm.all_metrics()
print('DATASET LEVEL METRICS')
print('miou: ', miou)
print('macc: ', macc)
print('oa: ', oa)