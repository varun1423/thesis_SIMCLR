import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from src import lars_optim
import pytorch_lightning as pl
import wandb

class LightingModel(pl.LightningModule):

    def __init__(self, p_model, lr, loss_func, arg_opti, weight_decay, batch_size):
        super().__init__()
        self.lr = lr
        self.loss_func = loss_func
        self.p_model = p_model
        self.arg_opti = arg_opti
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def forward(self, x):
        return self.p_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # self() calls self.forward()
        loss = self.loss_func(self(x), self(y))
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        #wandb.log({"nce_loss" : loss})
        return loss

    def configure_optimizers(self):
        if self.arg_opti == "ADAM":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        elif self.arg_opti == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        elif self.arg_opti == "LARS":
            learning_rate = 0.3 * self.batch_size / 256
            optimizer = lars_optim.LARS(
                self.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )
        return optimizer

