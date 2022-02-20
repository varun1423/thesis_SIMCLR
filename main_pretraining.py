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
nr = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(f"The model is using: {device}")

#initialize_model
simCLR_encoder = model.PreModel('ResNet18').to(device)

#data_loader train
data_loader_tbc = data_loader.ThermalBarrierCoating(phase='train',train_dir='D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime/',
                                                    csv_file='meta_data_with_lifetime.csv',
                                                    data_dir= 'data_morph_noise')

train_set = DataLoader(data_loader_tbc,batch_size = batch_size, shuffle=True, drop_last=True)

#data_loader validation
validation_data_loader = data_loader.ThermalBarrierCoating(phase='validation',train_dir='D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime/',
                                                    csv_file='test_csv_lt.csv',
                                                    data_dir= 'data_morph_noise')
validation_set = DataLoader(validation_data_loader,batch_size = 128,drop_last=True)

#optimizer
optimizer = lars_optim.LARS(
    [params for params in simCLR_encoder.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0)
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min= 0.05, last_epoch=-1, verbose=True)


criterion = model.SimCLR_Loss(batch_size = batch_size, temperature = 0.5)

tr_loss = []
val_loss = []

# training loop
for epoch in range(100):
    print(f"Epoch [{epoch}/{epochs}]\t")
    stime = time.time()
    simCLR_encoder.train()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()
    # training
    tr_loss_epoch = functions_file.train(train_set, simCLR_encoder, criterion, optimizer, nr, device)

    if nr == 0 and (epoch + 1) % 50 == 0:
        functions_file.save_model(simCLR_encoder, optimizer, mainscheduler, current_epoch,
                   "SimCLR_TBC_checkpoint_{}_260621.pt")

    lr = optimizer.param_groups[0]["lr"]
    # validation
    simCLR_encoder.eval()
    with torch.no_grad():
        val_loss_epoch = functions_file.valid(validation_set, simCLR_encoder, criterion, nr, device)

    tr_loss.append(tr_loss_epoch / len(train_set))

    val_loss.append(val_loss_epoch / len(validation_set))

    print(
        f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(train_set)}\t lr: {round(lr, 5)}"
    )
    print(
        f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(validation_set)}\t lr: {round(lr, 5)}"
    )
    current_epoch += 1

functions_file.save_model(simCLR_encoder, optimizer, mainscheduler, current_epoch, "SimCLR_TBC_final_checkpoint_{}_260621.pt")
