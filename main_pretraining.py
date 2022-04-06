"""
                             train_dir="D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime/",
"""
import os
import torch
from src import model
from src import lars_optim
from src import data_loader
from torch.utils.data import DataLoader
from src import functions_file
import wandb
os.environ["WANDB_SILENT"] = "true"


torch.manual_seed(14)

default_hyperparameter = dict(batch_size=4,
                              epochs=30,
                              base_encoder="ResNet18",
                              weight_decay=1e-6, lr=0.2,
                              Lars_optimizer=True,
                              arg_optimizer="LARS",
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
                              data_dir="data_with_Lifetime")

# init_wandb
wandb.init(project="simCLR_scratch",
           entity="varun-s",
           config=default_hyperparameter)

config = wandb.config



print(f"using batch size : {config.batch_size}")



# initialize_model
simCLR_encoder = model.PreModel(config.base_encoder,
                                config.conv_1_channel,
                                config.p_in_features,
                                config.p_hidden_features,
                                config.p_out_features,
                                config.p_head_type)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs! :)",  flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using dataparallel..!', flush=True)
    simCLR_encoder = torch.nn.DataParallel(simCLR_encoder)
else:
    print(f"Let's use 1 GPU only :(!", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    simCLR_encoder = model.PreModel(config.base_encoder,
                                    config.conv_1_channel,
                                    config.p_in_features,
                                    config.p_hidden_features,
                                    config.p_out_features,
                                    config.p_head_type)

simCLR_encoder.to(device)


# data_loader train
"""
data_loader_tbc = data_loader.ThermalBarrierCoating(phase='train',train_dir=config.train_dir,
                                                    csv_file=config.csv_file,
                                                    data_dir=config.data_dir,
                                                    crop_size=config.crop_size,
                                                    channel=config.conv_1_channel)

train_set = DataLoader(data_loader_tbc,
                       batch_size=config.batch_size,
                       shuffle=True,
                       drop_last=True)
"""

data_loader_tbc = data_loader.TBC_H5(phase='train',
                                     hdf5_file= "D:/TUD/TU_Dresden/WiSe_2021/SimCLR/tbc_hdf5.h5" ,
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
                                         config.weight_decay,
                                         sgd_adam_lr= config.learning_rate_ds)

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

# ckpt folder and saving ckpt
wandb_id = wandb.run.id
ckpt_folder = functions_file.ckpt_folder('ckpt_all', wandb_id)
# fetch run id
run_name =  wandb.run.id

# training loop
for epoch in range(config.epochs):
    print(f"Epoch [{epoch}/{config.epochs}]\t", flush=True)
    #stime = time.time()
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
                                  ckpt_folder, run_name)

    lr = optimizer.param_groups[0]["lr"]

    tr_loss.append(tr_loss_epoch / len(train_set))
    wandb.log({"train_epoch_loss": tr_loss_epoch / len(train_set), "Epochs": epoch+1,})
    print(f"Epoch [{epoch}/{config.epochs}]\t Training Loss: {tr_loss_epoch / len(train_set)}\t lr: {round(lr, 5)},", flush=True)
    current_epoch += 1

functions_file.save_model(simCLR_encoder,
                          optimizer,
                          mainscheduler,
                          current_epoch,
                          ckpt_folder,
                          "final")

wandb.finish()