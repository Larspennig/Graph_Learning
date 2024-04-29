from lightning import LightningModule
import torch
from model.model import TransformerGNN


class Lightning_GNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerGNN()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs = batch
        target = batch.y.type(torch.LongTensor)
        output = self(inputs)
        loss = self.loss_fn(output, target)
        return loss

    def validation_step(self, batch):
        a = 1
        return a

    def configure_optimizers(self):
        return torch.optim.adam(self.model.parameters(), lr=0.1)
