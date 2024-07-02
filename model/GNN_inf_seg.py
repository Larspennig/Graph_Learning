from lightning import LightningModule
import torch
from model.model_seg import TransformerGNN


class Lightning_GNN(LightningModule):
    def __init__(self, config):
        self.dev = config['device']
        super().__init__()
        self.model = TransformerGNN(config=config)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.config = config

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs = batch
        target = batch.y.type(torch.LongTensor)
        output = self(inputs)
        loss = self.loss_fn(output, target.to(self.dev))
        self.log('train_loss', loss.item(), on_epoch=True,
                 batch_size=self.config['batch_size'])
        self.log('curr_train_loss', loss.item(), on_step=True,
                 batch_size=self.config['batch_size'])
        self.log('batch_size', 2)
        values = output.max(dim=1).indices
        accr = torch.sum(values == target.to(self.dev))/len(target.to(self.dev))
        self.log('train_acc', accr, on_epoch=True,
                 batch_size=self.config['batch_size'])
        return loss

    def validation_step(self, batch):
        inputs = batch
        target = batch.y.type(torch.LongTensor)
        output = self(inputs)
        loss = self.loss_fn(output, target.to(self.dev))
        self.log('val_loss', loss.item(), on_epoch=True,
                 batch_size=self.config['batch_size'])
        values = output.max(dim=1).indices
        accr = torch.sum(values == target.to(self.dev))/len(target.to(self.dev))
        self.log('val_acc', accr, on_epoch=True,
                 batch_size=self.config['batch_size'])
        return loss

    def test_step(self, batch):
        inputs = batch
        target = batch.y.type(torch.LongTensor)
        output = self(inputs)
        values = output.max(dim=1).indices
        accr = torch.sum(values == target)/len(target)
        self.log('test_acc', accr, on_epoch=True,
                 batch_size=self.config['batch_size'])
        return accr

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9, weight_decay=0.0001)
