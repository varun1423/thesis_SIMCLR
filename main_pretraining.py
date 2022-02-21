import time
import torch
from src import model
from src import lars_optim
from src import data_loader
from torch.utils.data import DataLoader
from src import functions_file
import matplotlib.pyplot as plt


# config
batch_size = 10
current_epoch = 0
epochs = 100
base_encoder = "ResNet18"
weight_decay = 1e-6
lr = 0.2
Lars_optimizer = True
arg_optimizer = "LARS"
temperature = 0.5
p_in_features = 512
p_hidden_features = 2048
p_out_features = 128
p_head_type = "nonlinear"
crop_size =  128
nr = 0
conv_1_channel = 3
train_dir = "D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime/"


#initialize_model
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs! :)",  flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simCLR_encoder = model.PreModel(base_encoder,
                                    conv_1_channel,
                                    p_in_features,
                                    p_hidden_features,
                                    p_out_features,
                                    p_head_type)
    simCLR_encoder.to(device)
else:
    print(f"Let's use {torch.cuda.device_count()} GPU only :(!", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simCLR_encoder = model.PreModel(base_encoder,
                                    conv_1_channel,
                                    p_in_features,
                                    p_hidden_features,
                                    p_out_features,
                                    p_head_type)
    simCLR_encoder.to(device)


#data_loader train
data_loader_tbc = data_loader.ThermalBarrierCoating(phase='train',train_dir=train_dir,
                                                    csv_file='meta_data_with_lifetime.csv',
                                                    data_dir= 'data_morph_noise',
                                                    crop_size=crop_size)
"""
for slurm files
data_loader_tbc = data_loader.ThermalBarrierCoating(phase='train',train_dir='/p/project/hai_consultantfzj/set_up/solo-learn_SSL/dataset_crop_imgs',
                                                    csv_file='meta_data_with_lifetime.csv',
                                                    data_dir= 'tbc_with_lifetime/data_with_Lifetime')
"""

train_set = DataLoader(data_loader_tbc,
                       batch_size=batch_size,
                       shuffle=True,
                       drop_last=True)


#optimizer
optimizer,_ = lars_optim.load_optimizer(arg_optimizer,simCLR_encoder, batch_size, epochs, weight_decay )



warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0)

mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     500,
                                                                     eta_min=0.05,
                                                                     last_epoch=-1,
                                                                     verbose=True)
# loss initilization
criterion = model.SimCLR_Loss(batch_size=batch_size,
                              temperature=temperature)

tr_loss = []

# training loop
for epoch in range(100):
    print(f"Epoch [{epoch}/{epochs}]\t", flush=True)
    stime = time.time()
    simCLR_encoder.train()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()
    # training
    tr_loss_epoch = functions_file.train(train_set,
                                         simCLR_encoder,
                                         criterion,
                                         optimizer,
                                         nr,
                                         device)

    if nr == 0 and (epoch + 1) % 20 == 0:
        functions_file.save_model(simCLR_encoder,
                                  optimizer,
                                  mainscheduler,
                                  current_epoch,
                                  "SimCLR_TBC_checkpoint_{}.pt")

    lr = optimizer.param_groups[0]["lr"]

    tr_loss.append(tr_loss_epoch / len(train_set))

    print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(train_set)}\t lr: {round(lr, 5)},", flush=True)
    current_epoch += 1

functions_file.save_model(simCLR_encoder, optimizer, mainscheduler, current_epoch, "SimCLR_TBC_final_checkpoint_{}_260621.pt")