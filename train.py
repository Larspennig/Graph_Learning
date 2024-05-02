import numpy as np
import lightning as pl
import datetime
import yaml
import os

import torch_geometric as tg
from sklearn.model_selection import train_test_split

from model.GNN_inf import Lightning_GNN
from geometric import Extract_Geometric
from loaders.Mdndataloader import Modelnet40
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


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

train_loader = tg.loader.DataLoader(dataset[train_idx],
                                    batch_size=config['batch_size'],
                                    num_workers=2)

val_loader = tg.loader.DataLoader(dataset[val_idx],
                                  batch_size=config['batch_size'],
                                  num_workers=2)

# Model setup
GNN_model = Lightning_GNN(config=config)

data = next(iter(train_loader))
GNN_model(data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'Model has {count_parameters(GNN_model)} parameters.')

# Setup output dir
run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
output_dir = os.path.join(config['checkpoints'], run_time)
checkpoint_filename = "{epoch:02d}-{train_loss:.2f}"

logger = CSVLogger(save_dir=output_dir,
                   flush_logs_every_n_steps=10)

checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                      monitor='val_loss',
                                      mode='min',
                                      dirpath=output_dir,
                                      filename=checkpoint_filename)

# Train
trainer = pl.Trainer(max_epochs=config['max_epochs'],
                     check_val_every_n_epoch=1,
                     callbacks=[checkpoint_callback],
                     default_root_dir=output_dir,
                     accelerator='cpu',
                     logger=logger,
                     log_every_n_steps=5,
                     limit_train_batches=2,
                     limit_val_batches=2)

trainer.fit(GNN_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
