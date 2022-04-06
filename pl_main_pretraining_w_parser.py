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
from pytorch_lightning.callbacks import ModelCheckpoint


DDPStrategy = DDPPlugin

if __name__ == "__main__":

    pl.seed_everything(123)

    #os.environ["WANDB_SILENT"] = "true"

    config = dict(batch_size=80,
                  epochs=2,
                                  base_encoder="ResNet18",
                                  weight_decay=1e-6,
                                  lr=0.3,
                                  Lars_optimizer=True,
                                  arg_optimizer="SGD",
                                  temperature=0.3,
                                  p_in_features=512,
                                  p_hidden_features=2048,
                                  p_out_features=224,
                                  p_head_type="nonlinear",
                                  crop_size=224,
                                  conv_1_channel=3,
                                  transform=3,
                                  nr=0,
                                  learning_rate_ds= 0.01,
                                  hdf5_file="/p/project/hai_consultantfzj/set_up/simclr_with_down_stream/thesis_SIMCLR/hdf5/tbc_train_data_hdf5.h5",
                                  train_dir="D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime",
                                  csv_file="pretraining_trainset.csv",
                                  data_dir="data_with_Lifetime",
                                  gpus = 1)

    # wandb.init(project="simCLR_scratch", entity="varun-s", config=default_hyperparameter)
    # config = wandb.config
    # run_name = wandb.run.name

    simCLR_encoder = model.PreModel(config["base_encoder"],
                                    config["conv_1_channel"],
                                    config["p_in_features"],
                                    config["p_hidden_features"],
                                    config["p_out_features"],
                                    config["p_head_type"])

    data_loader_tbc = data_loader.TBC_H5(phase='train',
                                         hdf5_file=config["hdf5_file"],
                                         crop_size=config["crop_size"],
                                         channel=config["conv_1_channel"])

    train_set = DataLoader(data_loader_tbc,
                           batch_size=config["batch_size"],
                           shuffle=True,
                           drop_last=True,
                           num_workers=1)

    loss_criterion = model.SimCLR_Loss(batch_size=config["batch_size"],
                                       temperature=config["temperature"])

    model = LightingModel(p_model=simCLR_encoder,
                          lr=config["lr"],
                          loss_func=loss_criterion,
                          arg_opti=config["arg_optimizer"],
                          batch_size=config["batch_size"],
                          weight_decay=config["weight_decay"])

    #wandb_logger = WandbLogger(project= "ddp" , entity="varun-s", offline="offline")
    wandb_logger = WandbLogger(project="wandb_logging",
                               entity="varun-s",
                               offline=True)
    wandb_logger.watch(model, log="gradients", log_freq=100)

    # ckpt_path = "/p/home/jusers/shitole1/juwels/shared/set_up/simclr_with_down_stream/pytorch_lightning/ckpt/simCLR"
    checkpoint_callback = ModelCheckpoint(dirpath="/p/home/jusers/shitole1/juwels/shared/set_up/simclr_with_down_stream/pytorch_lightning/ckpt/simCLR",
                                          filename="ssl_pretraining-{epoch:02d}")

    accelerator="ddp"
    trainer = pl.Trainer(logger=wandb_logger,
        progress_bar_refresh_rate=20,
        callbacks=[checkpoint_callback],
        max_epochs=config["epochs"],
        accelerator="ddp",
        gpus=4,
        plugins=DDPPlugin(find_unused_parameters=False) if accelerator == "ddp" else None,
        log_every_n_steps=50
    )

    trainer.fit(
        model,
        train_set)