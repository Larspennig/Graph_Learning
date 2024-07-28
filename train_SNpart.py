from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loaders.Sndataloader import SNpart_Dataset
from model.GNN_inf_seg import Lightning_GNN
from sklearn.model_selection import train_test_split
import torch_geometric as tg
import numpy as np
import lightning as pl
import datetime
import yaml
import os
import wandb
wandb.login(key='446bb0e42e6ee0d7b7a2224d3b524a036009d8ad')
# wandb.login()


def main():
    # Load array with params from config.yml
    with open('configs/config_SNpart.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Data setup
    dataset_train = SNpart_Dataset(root=config['root'],
                                    split='train')

    dataset_val = SNpart_Dataset(root=config['root'],
                                split = 'val')

    train_loader = tg.loader.DataLoader(dataset_train,
                                        batch_size=config['batch_size'],
                                        num_workers=2,
                                        shuffle=True)

    val_loader = tg.loader.DataLoader(dataset_train,
                                    batch_size=config['batch_size'],
                                    num_workers=2)

    # Model setup
    GNN_model = Lightning_GNN(config=config)

    #data = next(iter(train_loader))
    #GNN_model(data)


    GNN_model.to(config['device'])


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'Model has {count_parameters(GNN_model)} parameters.')

    # Setup output dir
    run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = os.path.join(config['checkpoints'], run_time)
    checkpoint_filename = "{epoch:02d}-{train_loss:.2f}"

    # logger = CSVLogger(save_dir=output_dir,flush_logs_every_n_steps=10)

    wandb_logger = WandbLogger(project=config['project_name'],name=config['run_name'])
    wandb_logger.experiment.config['learning_rate'] = config['learning_rate']
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
        patience=8,
        mode='max',
        verbose=True
        )

    # Train
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                        check_val_every_n_epoch=1,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        default_root_dir=output_dir,
                        accelerator=config['device'],
                        logger=wandb_logger,
                        log_every_n_steps=1)

    trainer.fit(GNN_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    if config['test']:
        dataset_test = SNpart_Dataset(root=config['root'],
                                    split='test')

        test_loader = tg.loader.DataLoader(dataset_test,
                                    batch_size=config['batch_size'],
                                    num_workers=2,
                                    shuffle = True)
        
        # retrieve the path to the best model checkpoint
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            raise RuntimeError("Best model checkpoint not found. Ensure that the checkpoint callback is correctly configured.")

        # load the best model checkpoint
        GNN_model = Lightning_GNN(config=config)
        GNN_model.load_from_checkpoint(best_model_path)

        # initialize a new Trainer for testing
        trainer = pl.Trainer(
            accelerator=config['device'],
            logger=wandb_logger
        )
        
        # Perform testing
        test_results = trainer.test(model=GNN_model, dataloaders=test_loader)
        print(f"Test Results: {test_results}")

        # Optionally, save the test results to WandB or local files
        wandb_logger.experiment.log({"test_results": test_results})

        trainer.test(test_dataloaders=test_loader)

if __name__ == '__main__':
    main()