from lightning import LightningModule
import torch
from model.model_seg import TransformerGNN
from model.model_super_seg_simple import TransformerGNN_super_simple
from model.model_seg_double_knn import TransformerGNN_double
from model.model_seg_gctx import TransformerGNN_global
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from utils.metrics import ConfusionMatrix
from torchmetrics import JaccardIndex
import math



def compute_ins_miou(target, values, num_classes):
    confusion_matrix = torch.zeros(
        num_classes, num_classes, dtype=torch.int64, device='cuda')

    for t, p in zip(target.view(-1), values.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    IoU = torch.zeros(num_classes, device='cuda')
    for cls in range(num_classes):
        TP = confusion_matrix[cls, cls]
        FP = confusion_matrix[:, cls].sum() - TP
        FN = confusion_matrix[cls, :].sum() - TP
        IoU[cls] = TP / (TP + FP + FN + 1e-10)  # Avoid division by zero

    MIoU = IoU[IoU.nonzero()].mean()
    if math.isnan(MIoU): # if no part is correctly predicted
        MIoU = torch.tensor(0.).cuda()
    return MIoU


class Lightning_GNN(LightningModule):
    def __init__(self, config):
        self.dev = config['device']
        super().__init__()
        self.dataset = config['data']
        if config['model'] == 'standard':
            self.model = TransformerGNN(config=config)
        # elif config['model'] == 'super':
        #    self.model = TransformerGNN_super(config=config)
        elif config['model'] == 'super_simple':
            self.model = TransformerGNN_super_simple(config=config)
        elif config['model'] == 'double':
            self.model = TransformerGNN_double(config=config)
        elif config['model'] == 'global':
            self.model = TransformerGNN_global(config=config)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.config = config
        self.cm = ConfusionMatrix(config['num_classes'])
        if self.dataset == 'ShapeNetPart':
            self.cat_ious = {i: {'ious': [], 'num': 0} for i in range(config['num_categories'])}

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
        accr = torch.sum(values == target.to(self.dev)) / \
            len(target.to(self.dev))
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
        accr = torch.sum(values == target.to(self.dev)) / \
            len(target.to(self.dev))
        self.log('val_acc', accr, on_epoch=True,
                 batch_size=self.config['batch_size'])
    

        # Compute MIoU
        # here iterate over batch and compute miou per sample
        # save mIoU per sample and per category
        # only compute all 10 epochs 
        if (self.current_epoch+1) % 10 == 0:
            batch_ious = []
            for i,sample in enumerate(inputs.batch.unique()):
                mask = inputs.batch == sample
                target_sample = target[mask.cpu()]
                values_sample = values[mask.cpu()]
                sample_iou = compute_ins_miou(target_sample, values_sample, self.config['num_classes'])
                if self.dataset == 'ShapeNetPart':
                    self.cat_ious[inputs.cat_id[i].item()]['ious'].append(sample_iou)
                    self.cat_ious[inputs.cat_id[i].item()]['num'] += 1
                batch_ious.append(sample_iou)
            
            ins_MIoU = torch.mean(torch.stack(batch_ious))
            self.log('val_miou', ins_MIoU, on_epoch=True,
                    batch_size=self.config['batch_size'])
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
        
        batch_ious = []
    
        # Compute MIoU
        # here iterate over batch and compute miou per sample
        # save mIoU per sample and per category
        for i,sample in enumerate(inputs.batch.unique()):
            mask = inputs.batch == sample
            target_sample = target[mask]
            values_sample = values[mask]
            sample_iou = compute_ins_miou(target_sample, values_sample, self.config['num_classes'])
            if self.dataset == 'ShapeNetPart':
                self.cat_ious[inputs.cat_id[i].item()]['ious'].append(sample_iou)
                self.cat_ious[inputs.cat_id[i].item()]['num'] += 1
            batch_ious.append(sample_iou)
        
        ins_MIoU = torch.mean(torch.stack(batch_ious))
        self.log('test_miou', ins_MIoU, on_epoch=True,
                 batch_size=self.config['batch_size'])
    
        self.cm.update(values, target)
        return accr

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.config['learning_rate'],momentum=0.9, weight_decay=0.0001)

        scheduler = {
            'scheduler': StepLR(optimizer, gamma=self.config['step_gamma'], step_size=self.config['step_size']),
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

