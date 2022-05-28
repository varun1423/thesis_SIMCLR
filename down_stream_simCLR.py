import os
import torch
from torch import nn
from src import functions_file
from src import model
from downstream_src import data_loader_ds
from downstream_src import model_ds
from torch.utils.data import DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse

torch.manual_seed(1)




parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, required=True, 
                    help="number of classes in the dataset")
parser.add_argument('--epochs', type=int, required=False, default = 50,  
                    help="epochs"))
parser.add_argument('--ckpt', type=str, required=True 
                    help="file dir with ckpt file"))
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--encoder_fine_tune', default=False, action="store_true",
                    help="If activated, it fine tunes the encoder itherwise Linear evaluation")
parser.add_argument('--project', type=str, default="with_metrics")
parser.add_argument('--proj_out_feats', type=int, required=False, default=256)
parser.add_argument('--pretrained', default=False, action="store_true",
                    help="encoder is initialized with imagenet weights")
parser.add_argument('--micronet', default=False, action="store_true", 
                    help = "(7,7) kernel is used in conv1 layer of encoder for micronet dataset transfer learning")


args = parser.parse_args()

if args.num_classes == 10:
    csv_file_train="striding_crops_train_set.csv"
    csv_file_validation="striding_crops_val_set.csv"
else:
    csv_file_train="striding_crops_train_set_4_way.csv"
    csv_file_validation="striding_crops_val_set_4_way.csv"

print(args.encoder_fine_tune)


default_hyperparameter = dict(batch_size=args.batch_size,
                              epochs=args.epochs,
                              base_encoder="ResNet18",
                              weight_decay=1e-6,
                              Lars_optimizer=True,
                              arg_optimizer="SGD",
                              temperature=0.3,
                              p_in_features=512,
                              p_hidden_features=2048,
                              p_out_features=args.proj_out_feats,
                              p_head_type="nonlinear",
                              crop_size=224,
                              conv_1_channel=3,
                              nr=0,
                              train_dir="/p/project/hai_consultantfzj/set_up/solo-learn_SSL/dataset_crop_imgs/", 
                              csv_file_train=csv_file_train,
                              csv_file_validation=csv_file_validation,
                              data_dir="tbc_with_lifetime/data_with_Lifetime",
                              ckpt=args.ckpt,
                              num_classes=args.num_classes,
                              classifier_type="Linear",
                              learning_rate_ds=0.005,
                              momentum=0.09,
                              encoder_fine_tune=args.encoder_fine_tune,
                              step_size=15,
                              comment ="mix_crops",
                              micronet = args.micronet
                              )

wandb.init(project=args.project, entity="varun-s", config=default_hyperparameter)
config = wandb.config

# initialize pretrained model
simCLR_encoder = model.PreModel(config.base_encoder,
                                config.conv_1_channel,
                                config.p_in_features,
                                config.p_hidden_features,
                                config.p_out_features,
                                config.p_head_type,
                                pretrained = args.pretrained,
                                micronet=config.micronet)

# load weights
state = torch.load(config.ckpt)['model_state_dict'] # module.encoder.conv1.... --> encoder.conv1.weight
for k in list(state.keys()):
        if "module" in k:
            state[k.replace("module.", "")] = state[k]
        del state[k]
simCLR_encoder.load_state_dict(state, strict=True)
print(f"weights loaded from {config.ckpt} file succesfully......!!")

# build a downstream model- attach linear classifier on top.
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
                       drop_last=False)

# data_loader validation
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
scheduler = torch.optim.lr_scheduler.StepLR(down_stream_optimizer,
                                            config.step_size,
                                            gamma=0.1,
                                            last_epoch=- 1,
                                            verbose=True)

# ckpt folder and saving ckpt
wandb_id = wandb.run.id
ckpt_folder = functions_file.ckpt_folder('ckpt_all_downstream/', wandb_id)

run_name = wandb.run.id + "_down_stream_" + config.ckpt.split('_')[-2]

# loss
criterion = nn.CrossEntropyLoss()

tr_loss = []
val_loss = []

current_epoch = 0
for epoch in range(config.epochs):
    downstream_model.train()
    # train loop
    loss, correct_prediction = functions_file.training_ds(model=downstream_model,
                                                          data_loader=train_set,
                                                          optimizer= down_stream_optimizer,
                                                          criterion=criterion,
                                                          device=device)
    tr_loss.append(loss / len(train_set))

    accuracy = 100 * (correct_prediction / len(data_loader_tbc_train)).to(device)
    wandb.log({"train_epoch_accuracy": accuracy.round(), "Epochs": epoch + 1, })
    print(f"Epoch [{epoch}/{config.epochs}]\t Training Loss: {loss / len(train_set)}\t train_accuracy: {accuracy.round()}", flush=True)
    wandb.log({"train_epoch_loss": loss / len(train_set), "Epochs": epoch + 1, })

    #validation loop
    validation_loss, val_acc = functions_file.validation_ds(model=downstream_model,
                                                            data_loader=validation_set,
                                                            criterion=criterion,
                                                            device=device)
    scheduler.step()
    val_accuracy = 100 * (val_acc / len(data_loader_tbc_validation)).to(device)
    wandb.log({"val_epoch_accuracy": val_accuracy.round(), "Epochs": epoch + 1, })
    wandb.log({"val_epoch_loss": validation_loss / len(validation_set), "Epochs": epoch + 1, })
    if (epoch+1) == 2:
        functions_file.get_metrics(downstream_model, validation_set, device, config.num_classes)


    print(f"Epoch [{epoch}/{config.epochs}]\t Validation Loss: {validation_loss / len(validation_set)}\t val_accuracy: {val_accuracy.round()}", flush=True)
    current_epoch += 1

functions_file.get_metrics(downstream_model, validation_set, device, config.num_classes)
functions_file.save_model(model=downstream_model,
                          optimizer=down_stream_optimizer,
                          current_epoch=current_epoch,
                          scheduler=None,
                          name=ckpt_folder,
                          run_name=run_name)
