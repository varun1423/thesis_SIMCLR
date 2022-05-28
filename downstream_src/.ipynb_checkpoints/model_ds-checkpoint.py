import os
import time
import torch
from torch import nn

from src import model
from src import lars_optim
from src import data_loader
from torch.utils.data import DataLoader
from src import functions_file
import wandb


class Downstream_model(nn.Module):
    """created a downstream model.

        Args:
            pretrain (model): pretrained model is passed here.
            num_classes (int): number of classes
            in_features (int): output of emcoder/input of linear classifies
            encoder_fine_tune (bool): boolean to turn of stop gradient for encoder and train only linear classfier
            classifier_type (str): Linear / Non Linear
        Returns:
            forward function calculates the forward pass
    """
    def __init__(self, pretrain, num_classes, in_features, classifier_type = "Linear", encoder_fine_tune = False):
        super().__init__()

        self.pretrain = pretrain
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.in_features = in_features
        self.encoder_fine_tune = encoder_fine_tune

        for p in self.pretrain.parameters():
            p.requires_grad = encoder_fine_tune

        for p in self.pretrain.projector.parameters():
            p.requires_grad = encoder_fine_tune
        if classifier_type == "Linear":
            self.classifier = nn.Linear(self.in_features, self.num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_features, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.num_classes))


    def forward(self, x):
        out = self.pretrain.encoder(x)
        out = self.classifier(out)

        return out