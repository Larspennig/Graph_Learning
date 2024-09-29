from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loaders.Sndataloader2 import SNpart_Dataset
from model.GNN_inf_seg import Lightning_GNN
from sklearn.model_selection import train_test_split
import torch_geometric as tg
import numpy as np
import lightning as pl
import datetime
import yaml
import os
import wandb
import torch


def main():
    # Load array with params from config.yml
    with open('configs/config_SNpart.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    if config['debug']:
        wandb.init('disable')
        config['device'] = 'cpu'
        config['batch_size'] = 3
    else: 
        wandb.login(key='446bb0e42e6ee0d7b7a2224d3b524a036009d8ad')

    # Data setup
    dataset_train = SNpart_Dataset(root=config['root'],
                                   split='train')

    dataset_val = SNpart_Dataset(root=config['root'],
                                 split='val')

    train_loader = tg.loader.DataLoader(dataset_train,
                                        batch_size=config['batch_size'],
                                        num_workers=2,
                                        shuffle=True)

    val_loader = tg.loader.DataLoader(dataset_val,
                                      batch_size=15,
                                      num_workers=2)

    # Model setup
    GNN_model = Lightning_GNN(config=config)
    GNN_model.to(config['device'])
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {count_parameters(GNN_model)} parameters.')

    # Setup output dir
    run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = os.path.join(
        config['checkpoints'], run_time+config['run_name'])
    checkpoint_filename = "{epoch:02d}-{train_loss:.2f}"

    # copy model to output dir
    #os.makedirs(output_dir, exist_ok=True)
    #os.system('cp -r model/model_seg_gctx.py' + output_dir)

    # Set up logger
    wandb_logger = WandbLogger(
        project=config['project_name'], name=config['run_name'])
    wandb_logger.experiment.config['k_down'] = 16

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                          monitor='val_acc',
                                          mode='max',
                                          dirpath=output_dir,
                                          filename=checkpoint_filename)

    # Set up early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_acc',
        patience=20,
        mode='max',
        verbose=True
    )

    # Train
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         check_val_every_n_epoch=1,
                         callbacks=[checkpoint_callback],
                         default_root_dir=output_dir,
                         accelerator=config['device'],
                         logger=wandb_logger,
                         log_every_n_steps=1,)
    trainer.fit(GNN_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    if config['test']:
        dataset_test = SNpart_Dataset(root=config['root'],
                                      split='test')

        test_loader = tg.loader.DataLoader(dataset_test,
                                           batch_size=config['batch_size'],
                                           num_workers=2,
                                           shuffle=False)

        # retrieve the path to the best model checkpoint
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            raise ValueError("No best model found")

        # load the best model checkpoint
        GNN_model.load_state_dict(torch.load(best_model_path)['state_dict'])

        # initialize a new Trainer for testing
        trainer = pl.Trainer(
            accelerator=config['device'],
            logger=wandb_logger,)

        # test the model
        test_results = trainer.test(GNN_model, dataloaders=test_loader)
        
        print('TEST RESULTS Lightning')
        print(test_results)

        # Compute all final metrics
        print('FINAL METRICS')

        # Get instance metrics
        instance_miou = trainer.logged_metrics['test_miou']
        print('instance_miou: ', instance_miou.item())

        # Get category metrics
        cat_mious = GNN_model.cat_ious
        all_ious = []
        for key in cat_mious.keys():
            cat_mious[key]['miou'] = torch.mean(torch.stack(cat_mious[key]['ious']))
            all_ious.append(cat_mious[key]['miou'])
            category = dataset_test.number2name[key]
            print(f'{category} miou: ', cat_mious[key]['miou'].item())
        cat_miou = torch.mean(torch.stack(all_ious))
        print('cat_miou: ', cat_miou.item())

        # Get dataset level metrics
        miou, macc, oa, ious, accs = GNN_model.cm.all_metrics()
        print('DATASET LEVEL METRICS ON PART LEVEL')
        print('miou: ', miou)
        print('macc: ', macc)
        print('oa: ', oa)
        print('ious: ', ious)
        print('accs: ', accs)


if __name__ == '__main__':
    main()
