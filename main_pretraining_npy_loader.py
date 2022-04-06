"""
                             train_dir="D:/TUD/TU_Dresden/WiSe_2021/Thesis_FZJ/tbc_with_lifetime/",
            ------cropping strategy-----------------------------------------
                    cropping strategy option:
                             crops_no_overlap_only
                             crops_overlap_only
                             crops_overlap_and_non_overlap
            ----------------------------------------------------------------
"""
import torch
from src import model
from src import lars_optim
from src import data_loader
from torch.utils.data import DataLoader
from src import functions_file
import wandb

torch.manual_seed(14)

default_hyperparameter = dict(batch_size=300,
                              epochs=10000,
                              base_encoder="ResNet18",
                              weight_decay=1e-6,
                              lr=0.5,
                              arg_optimizer="LARS",
                              temperature=0.3,
                              p_in_features=512,
                              p_hidden_features=2048,
                              p_out_features=256,
                              p_head_type="nonlinear",
                              crop_size=224,
                              conv_1_channel=3,
                              transform=3,
                              nr=0,
                              train_dir="/p/project/hai_consultantfzj/set_up/solo-learn_SSL/dataset_crop_imgs/",
                              npy_file="orig_images_200_iter.npy",
                              data_dir="train_orignal_data",
                              cropping_strategy="crops_overlap_and_non_overlap",
                              comment="batch 300, using 2 times data in npy cosine step size 1000")

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

print(f"gpus count {torch.cuda.device_count()  }")
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
data_loader_tbc = data_loader.NPY_loader(train_dir=config.train_dir,
                                         npy_file=config.npy_file,
                                         crop_size=config.crop_size,
                                         channel=config.conv_1_channel,
                                         cropping_strategy=config.cropping_strategy
                                        )

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
                                         sgd_adam_lr=config.lr)


warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0)

mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     1000,
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
run_name = wandb.run.id

functions_file.save_model(simCLR_encoder,
                          optimizer,
                          mainscheduler,
                          current_epoch,
                          ckpt_folder,
                          run_name)

# training loop
for epoch in range(config.epochs):
    print(f"Epoch [{epoch}/{config.epochs}]\t", flush=True)
    simCLR_encoder.train()

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

    if config.nr == 0 and epoch < 10:
        warmupscheduler.step()
    if config.nr == 0 and epoch >= 10:
        mainscheduler.step()
    tr_loss.append(tr_loss_epoch / len(train_set))
    wandb.log({"train_epoch_loss": tr_loss_epoch / len(train_set), "Epochs": epoch+1, "learning_rate":lr})
    print(f"Epoch [{epoch}/{config.epochs}]\t Training Loss: {tr_loss_epoch / len(train_set)}\t lr: {round(lr, 5)},", flush=True)
    current_epoch += 1

functions_file.save_model(simCLR_encoder,
                          optimizer,
                          mainscheduler,
                          current_epoch,
                          ckpt_folder,
                          "_final")

print(f"ckp saved at  {ckpt_folder}")
wandb.finish()