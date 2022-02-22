import os
import torch
from torch import nn
from src import functions_file
from src import model
from src import lars_optim
from downstream_src import data_loader_ds
from downstream_src import model_ds
from torch.utils.data import DataLoader
import wandb

torch.manual_seed(14)
# os.environ["WANDB_SILENT"] = "true"

default_hyperparameter = dict(batch_size=20,
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
                              csv_file_train="pretraining_trainset.csv",
                              csv_file_validation="fine_tune_validation.csv",
                              data_dir= "data_with_Lifetime",
                              ckpt = "ckpt/ckpt_ResNet18_300_139_None_.pt",
                              num_classes = 10,
                              classifier_type = "Linear",
                              learning_rate_ds = 0.01,
                              momentum = 0.09,
                              encoder_fine_tune = False
                              )

wandb.init(project="simCLR_scratch", entity="varun-s", config=default_hyperparameter)
config = wandb.config

#if not os.path.exists(run_name):
#    os.makedirs(run_name)

# initialize encoder
simCLR_encoder = model.PreModel(config.base_encoder,
                                config.conv_1_channel,
                                config.p_in_features,
                                config.p_hidden_features,
                                config.p_out_features,
                                config.p_head_type)

# load weights
state = torch.load(config.ckpt)['model_state_dict'] # module.encoder.conv1.... --> encoder.conv1.weight
for k in list(state.keys()):
        if "module" in k:
            state[k.replace("module.", "")] = state[k]
        del state[k]
simCLR_encoder.load_state_dict(state, strict=True)
print(f"weights loaded from {config.ckpt} file succesfully..!!")


downstream_model = model_ds.Downstream_model(pretrain=simCLR_encoder,
                                             num_classes=config.num_classes,
                                             in_features=config.p_in_features,
                                             classifier_type=config.classifier_type,
                                             encoder_fine_tune = config.encoder_fine_tune)

# data parallel
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs! :)",  flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using dataparallel..!', flush=True)
    downstream_model = torch.nn.DataParallel(downstream_model)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using Single GPU..!', flush=True)
downstream_model.to(device)

# data_loader train
data_loader_tbc_train = data_loader_ds.ThermalBarrierCoating(phase='train',train_dir=config.train_dir,
                                                             csv_file=config.csv_file_train,
                                                             data_dir= config.data_dir,
                                                             crop_size=config.crop_size,
                                                             channel=config.conv_1_channel)
train_set = DataLoader(data_loader_tbc_train,
                       batch_size=config.batch_size,
                       shuffle=True,
                       drop_last=True)

data_loader_tbc_validation = data_loader_ds.ThermalBarrierCoating(phase='valid',train_dir=config.train_dir,
                                                                  csv_file=config.csv_file_validation,
                                                                  data_dir= config.data_dir,
                                                                  crop_size=config.crop_size,
                                                                  channel=config.conv_1_channel)
validation_set = DataLoader(data_loader_tbc_validation,
                            batch_size=config.batch_size,
                            shuffle=True,
                            drop_last=False)

# optimizer
down_stream_optimizer = functions_file.down_stream_optimizer(downstream_model=downstream_model, arg_optimizer=config.arg_optimizer, sgd_adam_lr=config.learning_rate_ds)


# ckpt folder and saving ckpt
wandb_id = wandb.run.id
ckpt_folder = functions_file.ckpt_folder('ckpt_all/', wandb_id)
# fetch run id

run_name =  wandb.run.id +"_down_stream_" + config.ckpt.split('_')[-2]
print(run_name)
# loss
criterion = nn.CrossEntropyLoss()

tr_loss = []
val_loss = []


current_epoch = 0
for epoch in range(config.epochs):
    #train
    downstream_model.train()
    loss, correct_prediction = functions_file.training_ds(model=downstream_model,
                                                          data_loader=train_set,
                                                          optimizer= downstream_model,
                                                          criterion=criterion,
                                                          device=device)
    tr_loss.append(loss / len(train_set))

    accuracy = 100 * (correct_prediction / len(data_loader_tbc_train)).to(device)
    wandb.log({"train_epoch_accuracy": accuracy.round(), "Epochs": epoch + 1, })
    print(f"Epoch [{epoch}/{config.epochs}]\t Training Loss: {loss / len(train_set)}\t train_accuracy: {accuracy.round()}", flush=True)
    wandb.log({"train_epoch_loss": loss / len(train_set), "Epochs": epoch + 1, })

    #validation
    validation_loss, val_acc = functions_file.validation_ds(model=downstream_model,
                                                            data_loader=validation_set,
                                                            criterion=criterion,
                                                            device=device)
    val_accuracy = 100 * (val_acc / len(data_loader_tbc_validation)).to(device)
    wandb.log({"val_epoch_accuracy": val_accuracy.round(), "Epochs": epoch + 1, })
    wandb.log({"val_epoch_loss": validation_loss / len(validation_set), "Epochs": epoch + 1, })
    print(f"Epoch [{epoch}/{config.epochs}]\t Validation Loss: {validation_loss / len(validation_set)}\t val_accuracy: {val_accuracy.round()}", flush=True)
    current_epoch += 1
    if epoch == config.epochs - 10:
        functions_file.save_model(downstream_model, downstream_model, current_epoch, ckpt_folder, run_name)
functions_file.save_model(downstream_model, downstream_model, current_epoch, ckpt_folder, run_name)