from lightning import LightningModule
import torch
from model.model_seg import TransformerGNN
from model.model_super_seg_simple import TransformerGNN_super_simple
from model.model_seg_double_knn import TransformerGNN_double
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class Lightning_GNN(LightningModule):
    def __init__(self, config):
        self.dev = config['device']
        super().__init__()
        if config['model'] == 'standard':
            self.model = TransformerGNN(config=config)
        #elif config['model'] == 'super':
        #    self.model = TransformerGNN_super(config=config)
        elif config['model'] == 'super_simple':
            self.model = TransformerGNN_super_simple(config=config)
        elif config['model'] == 'double':
            self.model = TransformerGNN_double(config=config)
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
        '''
        # Compute MIoU
        num_classes = self.config['num_classes']
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=self.device)

        for t, p in zip(target.view(-1), values.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        IoU = torch.zeros(num_classes, device=self.device)
        for cls in range(num_classes):
            TP = confusion_matrix[cls, cls]
            FP = confusion_matrix[:, cls].sum() - TP
            FN = confusion_matrix[cls, :].sum() - TP
            IoU[cls] = TP / (TP + FP + FN + 1e-10)  # Avoid division by zero

        MIoU = IoU[IoU.nonzero()].mean()
        self.log('val_miou', MIoU, on_epoch=True, batch_size=self.config['batch_size'])
        '''
        return loss

    def test_step(self, batch):
        inputs = batch
        target = batch.y.type(torch.LongTensor).to(self.dev)
        output = self(inputs)
        values = output.max(dim=1).indices
        # compute MIoU

        accr = torch.sum(values == target)/len(target)
        self.log('test_acc', accr, on_epoch=True,
                 batch_size=self.config['batch_size'])
        
        # Compute MIoU
        num_classes = self.config['num_classes']
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=self.device)

        for t, p in zip(target.view(-1), values.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        IoU = torch.zeros(num_classes, device=self.device)
        for cls in range(num_classes):
            TP = confusion_matrix[cls, cls]
            FP = confusion_matrix[:, cls].sum() - TP
            FN = confusion_matrix[cls, :].sum() - TP
            IoU[cls] = TP / (TP + FP + FN + 1e-10)  # Avoid division by zero

        MIoU = IoU[IoU.nonzero()].mean()
        self.log('test_miou', MIoU, on_epoch=True, batch_size=self.config['batch_size'])
        
        return accr
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9, weight_decay=0.0001)
        
        scheduler = {
            'scheduler': StepLR(optimizer, gamma=0.3, step_size=35),
            'interval': 'epoch', 
            'frequency': 2
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
'''
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9, weight_decay=0.0001)
'''