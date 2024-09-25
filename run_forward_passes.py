from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from loaders.Sdataloader import Stanford_Dataset
from model.GNN_inf_seg import Lightning_GNN
from sklearn.model_selection import train_test_split
import torch_geometric as tg
import numpy as np
import torch
import lightning as pl
import datetime
import yaml
import os
import wandb
import itertools

with open('configs/config_S3DIS.yml', 'r') as f:
    config = yaml.safe_load(f)
dataset_test = Stanford_Dataset(root=config['root'],
                                split='test')
with open('configs/config_S3DIS_global.yml', 'r') as f:
    config_glob = yaml.safe_load(f)

test_loader = tg.loader.DataLoader(dataset_test,
                                    batch_size=2,
                                    num_workers=2,
                                    shuffle=False,)

# Load standard model
GNN_model = Lightning_GNN(config=config)
GNN_model.load_state_dict(torch.load('model_checkpoints/2024-09-15_18.03.45s3dis_base/epoch=599-train_loss=0.06.ckpt')['state_dict'])

# Load global model
GNN_model_glob = Lightning_GNN(config=config_glob)
GNN_model_glob.load_state_dict(torch.load('/home/lars/models/2024-09-17_19.20.45s3dis_global/epoch=609-train_loss=0.08.ckpt')['state_dict'])

for i,sample in enumerate(test_loader):

    with torch.no_grad():
        GNN_model.eval()
        out_pc = GNN_model(sample.cuda()).cpu()

    prediction = torch.argmax(out_pc, dim=1)
    accr_list = []
    for i in range(config['batch_size']):
        accr = torch.sum(prediction[sample.batch == i] == sample.y[sample.batch == i]).item() / len(sample.y[sample.batch == i])
        accr_list.append((accr, i))
    lowest_accuracy_indices = [x[1] for x in sorted(accr_list, key=lambda x: x[0])[:3]]
    print(lowest_accuracy_indices)

    for j in range(2):
        batch_idx = j
        ax = 0

        pos = sample.pos[sample.batch == batch_idx]
        pos = pos - pos.mean(dim=0)
        color = sample.x[sample.batch == batch_idx][:,:3][pos[:,ax] < 0]
        label = sample.y[sample.batch == batch_idx][pos[:,ax] < 0]
        pred = prediction[sample.batch == batch_idx][pos[:,ax] < 0]
        pos = pos[pos[:,ax] < 0]

        out_name = f'cloud_{j}_batch_{i}'

        # Save all pointclouds
        os.makedirs('S3DIS_out/' + out_name,exist_ok=True)  
        np.savetxt('S3DIS_out/'+out_name+'/input.txt', np.concatenate((pos.numpy(),color.numpy()), axis=1))
        np.savetxt('S3DIS_out/'+out_name+'/label.txt', np.concatenate((pos.numpy(),label.unsqueeze(1).numpy()),axis=1))
        np.savetxt('S3DIS_out/'+out_name+'/pred.txt', np.concatenate((pos.numpy(),pred.unsqueeze(1).numpy()),axis=1))           

        glob_sampel = tg.data.Data(
            x=sample.x[sample.batch == batch_idx],
            pos=sample.pos[sample.batch == batch_idx],
            batch=torch.zeros(sample.x[sample.batch == batch_idx].shape[0], dtype=torch.long),
            y=sample.y[sample.batch == batch_idx]
        )
        out_pc = GNN_model(glob_sampel.cuda()).cpu()

        accr_global = torch.sum(torch.argmax(out_pc, dim=1) == sample.y[sample.batch == batch_idx]).item() / len(sample.y[sample.batch == batch_idx])
        print(accr_global)

        pos = glob_sampel.pos
        pos = pos - pos.mean(dim=0)
        preds = torch.argmax(out_pc, dim=1)[pos[:,ax] < 0]
        pos = pos[pos[:,ax] < 0]
        np.savetxt('S3DIS_out/'+out_name+'/pred_global.txt', np.concatenate((pos.numpy(),preds.unsqueeze(1).numpy()),axis=1))           