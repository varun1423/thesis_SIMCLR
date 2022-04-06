import os
import time
import torch
from src import model
from src import lars_optim
from src import data_loader
from torch.utils.data import DataLoader
import wandb
from pl_model import LightingModel
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project='MNIST', log_model= True)

DDPStrategy = DDPPlugin

pl.seed_everything(123)

os.environ["WANDB_SILENT"] = "true"

default_hyperparameter = dict(batch_size=4,
                              epochs=300,
                              base_encoder="ResNet18",
                              weight_decay=1e-6,
                              lr=0.2,
                              Lars_optimizer=True,
                              arg_optimizer="SGD",
                              temperature=0.3,
                              p_in_features=512,
                              p_hidden_features=2048,
                              p_out_features=224,
                              p_head_type="nonlinear",
                              crop_size=224,
                              conv_1_channel=3,
                              nr=0,
                              learning_rate_ds= 0.01,
                              train_dir="D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime",
                              csv_file="pretraining_trainset.csv",
                              data_dir="data_with_Lifetime",
                              gpus = 1)

wandb.init(project="simCLR_scratch", entity="varun-s", config=default_hyperparameter)
config = wandb.config
run_name = wandb.run.name

simCLR_encoder = model.PreModel(config.base_encoder,
                                config.conv_1_channel,
                                config.p_in_features,
                                config.p_hidden_features,
                                config.p_out_features,
                                config.p_head_type)

data_loader_tbc = data_loader.ThermalBarrierCoating(phase='train',train_dir=config.train_dir,
                                                    csv_file='meta_data_with_lifetime.csv',
                                                    data_dir= 'data_morph_noise',
                                                    crop_size=config.crop_size)
train_set = DataLoader(data_loader_tbc,
                       batch_size=config.batch_size,
                       shuffle=True,
                       drop_last=True)
loss_criterion = model.SimCLR_Loss(batch_size=config.batch_size,
                                   temperature=config.temperature)

model = LightingModel(p_model=simCLR_encoder,
                      lr=config.lr,
                      loss_func=loss_criterion,
                      arg_opti=config.arg_optimizer,
                      batch_size=config.batch_size,
                      weight_decay=config.weight_decay)

print(model)

trainer = pl.Trainer(
    max_epochs=config.epochs,
    gpus=0,
    log_every_n_steps=50
)

trainer.fit(
    model,
    train_dataloaders=train_set,
)