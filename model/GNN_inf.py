import pytorch_lightning as pl
import torch
from model.model import TransformerGNN


class Lightning_GNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerGNN()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch):
        inputs, target = batch
        output = self(inputs)

        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def validation_step(self, batch):
        a = 1
        return a

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
