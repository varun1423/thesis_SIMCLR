import torch
from src import model
from src import lars_optim
from src import data_loader
from src import functions_file
import wandb
import argparse

torch.manual_seed(14)

parser = argparse.ArgumentParser()

# model parameters
parser.add_argument('--batch_size', type=int, required=True,
                    help="batch size used for the model")
parser.add_argument('--steps', type=int, required=True,
                    help="Iteration for iter_data loader; in other terms epoch")
parser.add_argument('--base_encoder', type=str, required=True, 
                    help="base encoder needed to be used, example- ----base_encoder ResNet18, ----base_encoder ResNet34, --base_encoder ResNet50")
parser.add_argument('--cropping_strategy', type=str, required=True, 
                    help="possible cropping strategies: baseline, crops_no_overlap_only,crops_overlap_only, crops_overlap_and_non_overlap, multi_instance, crops_overlap_50_percent")
parser.add_argument('--image_input_channel', type=int, required=True,
                    help="Number of channels in Image: will decide the channels for convolution")
parser.add_argument('--npy_file', type=str, required=False,
                    help="NPY file for TBC dataset")
parser.add_argument('--proj_out_feats', type=int, required=False, default=256,
                    help="Projector head output dimension; this is the input for Contrastive loss")
parser.add_argument('--crop_size', type=int, required=True,
                    help="crop_size used for SimCLR Training")
parser.add_argument('--proj_hidden_feats', type=int, required=True,
                    help="Hidden layer dimension in projector")
parser.add_argument('--temperature', type=float, required=True,
                    help="hyper-parameter in loss function of SimCLR")
parser.add_argument('--train_dir', type=str, required=False, default="/p/project/hai_consultantfzj/set_up/solo-learn_SSL/dataset_crop_imgs/", 
                    help="file path train dir; points to csv/npy file")
parser.add_argument('--data_dir', type=str, required=False, default="train_orignal_data", 
                    help="file path to load data; points --> train_dir/data_dir")

# transformation parameters
parser.add_argument('--rotation_prob', type=float, required=False, default=0.5,
                    help="Rotation probability in Random Rotation Transfrom: should be between 0.0 to 1.0")
parser.add_argument('--blur_prob', type=float, required=False, default=0.5,
                    help="blur probability in gaussian blur Transfrom: should be between 0.0 to 1.0")
parser.add_argument('--horizontal_flip_prob', type=float, required=False, default=0.5,
                    help="horizontal flip probability in gaussian blur Transfrom: should be between 0.0 to 1.0")

# Other wandb and initialization 
parser.add_argument('--pretrained', default=False, action="store_true",
                    help="activate this option for initializing the SimCLR encodder with imagenet pretraining")
parser.add_argument('--wandb_project', type=str, required=False, default="simCLR_scratch", 
                    help="wandb project ID")
parser.add_argument('--comment', type=str, required=False, default="simCLR pretraining")


args = parser.parse_args()

# load npy file for TBC dataset
if args.cropping_strategy == "multi_instance":
    args.npy_file = "orig_train_data_list_class_wise.npy"
else:
    args.npy_file = "orig_images_200_iter.npy "

# config file
default_hyperparameter = dict(batch_size=args.batch_size,
                              epochs=args.steps,
                              base_encoder=args.base_encoder,
                              weight_decay=1e-6,
                              lr=0.5,
                              arg_optimizer="LARS",
                              temperature=args.temperature,
                              p_in_features=512,
                              p_hidden_features=args.proj_hidden_feats,
                              p_out_features=args.proj_out_feats,
                              p_head_type="nonlinear",
                              crop_size=args.crop_size,
                              conv_1_channel=args.image_input_channel,
                              train_dir=args.train_dir,
                              npy_file=args.npy_file,
                              data_dir=args.data_dir,
                              cropping_strategy=args.cropping_strategy,
                              comment=args.comment,
                              rotation_prob=args.rotation_prob,
                              blur_prob=args.blur_prob,
                              horizontal_flip_prob=args.horizontal_flip_prob
                             )

# init_wandb
wandb.init(project=args.wandb_project,
           entity="varun-s",
           config=default_hyperparameter)
config = wandb.config


# initialize_model
simCLR_encoder = model.PreModel(config.base_encoder,
                                config.conv_1_channel,
                                config.p_in_features,
                                config.p_hidden_features,
                                config.p_out_features,
                                config.p_head_type,
                                pretrained=args.pretrained)

# cuda
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs! :)",  flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using dataparallel..!', flush=True)

    # wrap the model in DP
    simCLR_encoder = torch.nn.DataParallel(simCLR_encoder)
else:
    print(f"Let's use 1 GPU only :(!", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
simCLR_encoder.to(device)


# data_loader train

data_loader_tbc = data_loader.iter_data_loader(npy_file=config.npy_file,
                                               crop_size=config.crop_size,
                                               channel=config.conv_1_channel,
                                               batch_size=config.batch_size,
                                               train_dir=config.train_dir,
                                               cropping_strategy=config.cropping_strategy,
                                               rotation_prob=config.rotation_prob,
                                               blur_prob=config.blur_prob,
                                               horizontal_flip_prob=config.horizontal_flip_prob)

# optimizer and schedulers
optimizer, _ = lars_optim.load_optimizer(config.arg_optimizer,
                                         simCLR_encoder,
                                         config.batch_size,
                                         config.epochs,
                                         config.weight_decay,
                                         sgd_adam_lr=config.lr)

warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0)
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     args.steps,
                                                                     eta_min=0.05,
                                                                     last_epoch=-1,
                                                                     verbose=True)

# loss criterion
criterion = model.SimCLR_Loss(batch_size=config.batch_size,
                              temperature=config.temperature)


# ckpt folder and saving ckpt
wandb_id = wandb.run.id
ckpt_folder = functions_file.ckpt_folder('ckpt_all', wandb_id)
# fetch run id
run_name = wandb.run.id
current_epoch = 0

# training loop
for epoch in range(config.epochs):
    print(f"Epoch [{epoch}/{config.epochs}]\t", flush=True)

    # set model to train()
    simCLR_encoder.train()

    # training loop SimCLR --> return loss for the epoch
    tr_loss_epoch = functions_file.train_simCLR(data_loader_tbc,
                                                simCLR_encoder,
                                                criterion,
                                                optimizer,
                                                device)

    if (epoch + 1)%2000==0:
        functions_file.save_model(simCLR_encoder,
                                  optimizer,
                                  mainscheduler,
                                  current_epoch,
                                  ckpt_folder, run_name)

    # tracking change in lr
    lr = optimizer.param_groups[0]["lr"]

    if epoch < 10:
        warmupscheduler.step()
    if epoch >= 10:
        mainscheduler.step()
    # log wandb: loss
    wandb.log({"train_epoch_loss": tr_loss_epoch, "Epochs": epoch+1, "learning_rate":lr})
    print(f"Epoch [{epoch}/{config.epochs}]\t Training Loss: {tr_loss_epoch}\t lr: {round(lr, 5)},", flush=True)

    # increase the counter
    current_epoch += 1

# save ckpt at end of training
functions_file.save_model(simCLR_encoder,
                          optimizer,
                          mainscheduler,
                          current_epoch,
                          ckpt_folder,
                          run_name="_final")
print(f"ckp saved at  {ckpt_folder}")

wandb.finish()