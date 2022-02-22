import os
import time
import torch
from src import model
from src import lars_optim
from src import data_loader
from torch.utils.data import DataLoader
from src import functions_file
import wandb

default_hyperparameter = dict(batch_size=700,
                              epochs=300,
                              base_encoder="ResNet18",
                              weight_decay=1e-6, lr=0.2,
                              Lars_optimizer=True,
                              arg_optimizer="SGD",
                              temperature=0.3,
                              p_in_features=512,
                              p_hidden_features=2048,
                              p_out_features=128,
                              p_head_type="nonlinear",
                              crop_size=128,
                              conv_1_channel=3,
                              nr=0,
                              train_dir="D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime/",
                              csv_file="pretraining_trainset.csv",
                              data_dir= "tbc_with_lifetime/data_with_Lifetime"
                              )

wandb.init(project="simCLR_scratch", entity="varun-s", config=default_hyperparameter)
config = wandb.config

print(wandb.run.name, flush=True)

# initialize_model
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs! :)",  flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using dataparallel..!', flush=True)
else:
    print(f"Let's use {torch.cuda.device_count()} GPU only :(!", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

simCLR_encoder = model.PreModel(config.base_encoder,
                                config.conv_1_channel,
                                config.p_in_features,
                                config.p_hidden_features,
                                config.p_out_features,
                                config.p_head_type)
simCLR_encoder = torch.nn.DataParallel(simCLR_encoder)

simCLR_encoder.to(device)


# data_loader train
data_loader_tbc = data_loader.ThermalBarrierCoating(phase='train',train_dir=config.train_dir,
                                                    csv_file=config.csv_file,
                                                    data_dir= config.data_dir,
                                                    crop_size=config.crop_size,
                                                    channel=config.conv_1_channel)

train_set = DataLoader(data_loader_tbc,
                       batch_size=config.batch_size,
                       shuffle=True,
                       drop_last=True)


# optimizer
optimizer, _ = lars_optim.load_optimizer(config.arg_optimizer,
                                         simCLR_encoder,
                                         config.batch_size,
                                         config.epochs,
                                         config.weight_decay)



warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0)

mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     500,
                                                                     eta_min=0.05,
                                                                     last_epoch=-1,
                                                                     verbose=True)
# loss initilization
criterion = model.SimCLR_Loss(batch_size=config.batch_size,
                              temperature=config.temperature)

tr_loss = []
current_epoch = 0


if not os.path.exists('ckpt_sgd_300'):
    os.makedirs('./ckpt_sgd_300')

# training loop
for epoch in range(config.epochs):
    print(f"Epoch [{epoch}/{config.epochs}]\t", flush=True)
    stime = time.time()
    simCLR_encoder.train()

    if config.nr == 0 and epoch < 10:
        warmupscheduler.step()
    if config.nr == 0 and epoch >= 10:
        mainscheduler.step()
    # training
    tr_loss_epoch = functions_file.train(train_set,
                                         simCLR_encoder,
                                         criterion,
                                         optimizer,
                                         config.nr,
                                         device)

    if config.nr == 0 and (epoch + 1) % 20 == 0:
        functions_file.save_model(simCLR_encoder,
                                  optimizer,
                                  mainscheduler,
                                  current_epoch,
                                  "./ckpt_sgd_300/SimCLR_TBC_ckpt_intermediate_", wandb.run.name)

    lr = optimizer.param_groups[0]["lr"]

    tr_loss.append(tr_loss_epoch / len(train_set))
    wandb.log({"train_epoch_loss": tr_loss_epoch / len(train_set), "Epochs": epoch+1,})
    print(f"Epoch [{epoch}/{config.epochs}]\t Training Loss: {tr_loss_epoch / len(train_set)}\t lr: {round(lr, 5)},", flush=True)
    current_epoch += 1

functions_file.save_model(simCLR_encoder, optimizer, mainscheduler, current_epoch, "./ckpt_sgd_300/SimCLR_TBC_ckpt_", wandb.run.name)

wandb.finish()