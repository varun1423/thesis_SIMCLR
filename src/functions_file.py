import torch
import os
from PIL import Image
import random
import wandb
import numpy
from sklearn import metrics
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


"""
x_i = x_i
        x_j = x_j
        print(f"size of x_i is : {x_i.size()}", flush=True)
        for step, (x_i_2, x_j_2) in enumerate(train_loader):
            x_i = torch.cat((x_i, x_i_2), 0)
            x_j = torch.cat((x_j, x_j_2), 0)
"""

def train_npy(train_loader, model, criterion, optimizer, nr, device):
    loss_epoch = 0
    x_i = torch.tensor([])
    x_j = torch.tensor([])
    for i in range(2):
        for step, (z1, z2) in enumerate(train_loader):
            x_i = torch.cat((x_i, z1), 0)
            x_j = torch.cat((x_j, z2), 0)
    #for step, (x_i, x_j) in enumerate(train_loader):
        print(f"size of x_i is : {x_i.size()}", flush=True)
        print(f"size of x_i is : {x_j.size()}", flush=True)

        optimizer.zero_grad()
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
        wandb.log({"train_step_loss": loss.item(),})
        loss_epoch += loss.item()
    return loss_epoch

def train(train_loader, model, criterion, optimizer, nr, device):
    loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
        wandb.log({"train_step_loss": loss.item(),})
        loss_epoch += loss.item()
    return loss_epoch


def train_multi_instance(train_loader, model, criterion, optimizer, nr, device):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        print(x_i.size(), x_j.size())

        optimizer.zero_grad()

        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        # if nr == 0 and step % 50 == 0:
        #    print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
        wandb.log({"train_step_loss": loss.item(),})
        loss_epoch += loss.item()
    return loss_epoch


def valid(valid_loader, model, criterion, nr):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(valid_loader):

        x_i = x_i.squeeze().float()
        x_j = x_j.squeeze().float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)

        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(), 5)}")

        loss_epoch += loss.item()
    return loss_epoch


def save_model(model, optimizer, scheduler, current_epoch, name, run_name):
    out = f"{name}/ckpt_{current_epoch}_{run_name}_.pt"
    #out = os.path.join('./',name.format(current_epoch))
    if scheduler == None:

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, out)
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()}, out)


def down_stream_optimizer(downstream_model, arg_optimizer, sgd_adam_lr):

    if arg_optimizer == "Adam":
        optimizer = torch.optim.Adam([params for params in downstream_model.parameters() if params.requires_grad], lr=sgd_adam_lr)  # TODO: LARS
    elif arg_optimizer == "SGD":
        optimizer = torch.optim.SGD([params for params in downstream_model.parameters() if params.requires_grad], lr=sgd_adam_lr)
    else:
        raise NotImplementedError

    return optimizer


def training_ds(model, data_loader, optimizer, criterion,device):
    loss_epoch = 0
    correct = 0
    for step, (x,y) in enumerate(data_loader):
        x = x.squeeze().to(device).float()
        label = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, 1)
        correct_per_batch = (pred == label).float()
        wandb.log({"train_step_loss": loss.item(),})
        wandb.log({"train_accuracy_step" : 100*(correct_per_batch.sum()/len(correct_per_batch)),})
        if step % 50 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t Loss: {round(loss.item(), 5)}\t accuracy : {100*(correct_per_batch.sum()/len(correct_per_batch))}")

        loss_epoch += loss.item()
        correct += correct_per_batch.sum()

    return loss_epoch, correct

def validation_ds(model, data_loader, criterion,device):

    model.eval()
    loss_epoch = 0
    correct = 0
    with torch.no_grad():
        for step, (x,y) in enumerate(data_loader):
            x = x.squeeze().to(device).float()
            label = y.to(device)
            output = model(x)
            loss = criterion(output, label)
            _, pred = torch.max(output, 1)
            correct_per_batch = (pred == label).float()
            wandb.log({"val_step_loss": loss.item(), })
            wandb.log({"val_accuracy_step": 100 * (correct_per_batch.sum() / len(correct_per_batch)), })
            if step % 50 == 0:
                print(
                    f"Step [{step}/{len(data_loader)}]\t val_Loss: {round(loss.item(), 5)}\t val_accuracy : {100 * (correct_per_batch.sum() / len(correct_per_batch))}")

            loss_epoch += loss.item()
            correct += correct_per_batch.sum()
        return loss_epoch, correct


def ckpt_folder(ckpt_all,wandb_id):
    if not os.path.exists(f"./{ckpt_all}"):
        os.makedirs(ckpt_all)
    if not os.path.exists(f"{ckpt_all}/ckpt__{wandb_id}/"):
        os.makedirs(f"{ckpt_all}/ckpt__{wandb_id}/")
    ckpt_folder = f'./{ckpt_all}/ckpt__{wandb_id}/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    return ckpt_folder


def random_crop_2048(image, crop_size, overlap):
    #inp_img = Image.open(image, mode='r')
    if type(image) == numpy.ndarray:
        inp_img = Image.fromarray(image)
    else:
        inp_img = Image.open(image)
    resolution_x = 2048
    resolution_y = 2048
    end_limit_x = (resolution_x - crop_size)
    end_limit_y = (resolution_y - crop_size)

    x = random.randint(0, end_limit_x)
    y = random.randint(0, end_limit_y)

    if (x + crop_size > 2048) or (y + crop_size > 2048):
        resample = True
    else:
        resample = False
    while resample:
        x = random.randint(0, end_limit_x)
        y = random.randint(0, end_limit_y)
        if (x + crop_size < 2048) and (y + crop_size < 2048):
            resample = False
    main_Crop = ((x, y, x + crop_size, y + crop_size))
    main_Crop_image = inp_img.crop(main_Crop)

    if overlap:
        overlapping_crop_start_list = []

        if (x-(crop_size/2) > 0) and (y-(crop_size/2) > 0): # nrth west
            overlapping_crop_start_list.append((x-(crop_size/2), y-(crop_size/2)))
        if ((x+crop_size)+(crop_size/2) < resolution_x) and (y-(crop_size/2) > 0): # north east
            overlapping_crop_start_list.append((x+(crop_size/2), y-(crop_size/2)))
        if ((x+crop_size)+(crop_size/2) < resolution_x) and ((y+crop_size)+(crop_size/2) < resolution_y): # south east
            overlapping_crop_start_list.append((x+(crop_size/2),y+(crop_size/2)))
        if (x-(crop_size/2) > 0) and ((y+crop_size)+(crop_size/2) < resolution_y): # west south
            overlapping_crop_start_list.append((x-(crop_size/2), y+(crop_size/2)))

        co_ordinates = random.choice(overlapping_crop_start_list)
        second_overlap = ((co_ordinates[0], co_ordinates[1], co_ordinates[0] + crop_size, co_ordinates[1] + crop_size))
        second_image = inp_img.crop(second_overlap)

        return main_Crop_image, second_image
    else:
        non_overlapping_crop_start_list = []

        if (x-(2*crop_size) > 0) and (y-(2*crop_size)>0): # nrth west
            non_overlapping_crop_start_list.append((x-(2*crop_size), y-(2*crop_size)))
        if ((x+crop_size)+(2*crop_size) < resolution_x) and (y-(2*crop_size) > 0): # north east
            non_overlapping_crop_start_list.append((x+(2*crop_size), y-(2*crop_size)))
        if ((x+crop_size)+(2*crop_size) < resolution_x) and ((y+crop_size)+(2*crop_size) < resolution_y): # south east
            non_overlapping_crop_start_list.append((x+(2*crop_size), y+(2*crop_size)))
        if (x-(2*crop_size) > 0) and ((y+crop_size)+(2*crop_size) < resolution_y): # south west
            non_overlapping_crop_start_list.append((x-(2*crop_size), y+(2*crop_size)))

        if len(non_overlapping_crop_start_list)>0:
            co_ordinates = random.choice(non_overlapping_crop_start_list)
            second_non_overlap = ((co_ordinates[0], co_ordinates[1], co_ordinates[0] + crop_size, co_ordinates[1] + crop_size))
            second_image_non_overlap = inp_img.crop(second_non_overlap)
        else:
            main_Crop_image, second_image_non_overlap = random_crop_2048(image, 256, False)

        return main_Crop_image, second_image_non_overlap


def random_crop_712(path_img, crop_size, overlap):
    #inp_img = Image.open(path_img)

    if type(path_img) == numpy.ndarray:
        inp_img = Image.fromarray(path_img)
    else:
        inp_img = Image.open(path_img)

    resolution_x = 1024
    resolution_y = 712
    end_limit_x = (resolution_x - crop_size)
    end_limit_y = (resolution_y - crop_size)

    x = random.randint(0, end_limit_x)
    y = random.randint(0, end_limit_y)

    if (x + crop_size > 1024) or (y + crop_size > 712):
        resample = True
    else:
        resample = False
    while resample:
        x = random.randint(0, end_limit_x)
        y = random.randint(0, end_limit_y)
        if (x + crop_size < 2048) and (y + crop_size < 2048):
            resample = False

    main_Crop = ((x, y, x + crop_size, y + crop_size))
    main_Crop_image = inp_img.crop(main_Crop)

    if overlap:
        overlapping_crop_start_list = []

        if (x-(crop_size/2) > 0) and (y-(crop_size/2) > 0): # nrth west
            overlapping_crop_start_list.append((x-(crop_size/2), y-(crop_size/2)))
        if ((x+crop_size)+(crop_size/2) < resolution_x) and (y-(crop_size/2) > 0): # north east
            overlapping_crop_start_list.append((x+(crop_size/2), y-(crop_size/2)))
        if ((x+crop_size)+(crop_size/2) < resolution_x) and ((y+crop_size)+(crop_size/2) < resolution_y): # south east
            overlapping_crop_start_list.append((x+(crop_size/2),y+(crop_size/2)))
        if (x-(crop_size/2) > 0) and ((y+crop_size)+(crop_size/2) < resolution_y): # west south
            overlapping_crop_start_list.append((x-(crop_size/2), y+(crop_size/2)))

        co_ordinates = random.choice(overlapping_crop_start_list)
        second_overlap = ((co_ordinates[0], co_ordinates[1], co_ordinates[0] + crop_size, co_ordinates[1] + crop_size))
        second_image = inp_img.crop(second_overlap)

        return main_Crop_image, second_image
    else:
        non_overlapping_crop_start_list = []

        if (x-(2*crop_size) > 0) and (y-(crop_size)>0): # nrth west
            non_overlapping_crop_start_list.append((x-(2*crop_size), y-(crop_size)))
        if ((x+crop_size)+(2*crop_size) < resolution_x) and (y-(crop_size) > 0): # north east
            non_overlapping_crop_start_list.append((x+(2*crop_size), y-(crop_size)))
        if ((x+crop_size)+(2*crop_size) < resolution_x) and ((y+crop_size)+(crop_size) < resolution_y): # south east
            non_overlapping_crop_start_list.append((x+(2*crop_size),y+(crop_size)))
        if (x-(2*crop_size) > 0) and ((y+crop_size)+(crop_size) < resolution_y): # west south
            non_overlapping_crop_start_list.append((x-(2*crop_size), y+(crop_size)))

        if len(non_overlapping_crop_start_list) > 0:
            co_ordinates = random.choice(non_overlapping_crop_start_list)
            second_non_overlap = ((co_ordinates[0], co_ordinates[1], co_ordinates[0] + crop_size, co_ordinates[1] + crop_size))
            second_image_non_overlap = inp_img.crop(second_non_overlap)
        else:
            main_Crop_image, second_image_non_overlap = random_crop_712(path_img, crop_size, False)

        return main_Crop_image, second_image_non_overlap


def get_all_preds(model, loader, device):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    for step, (x,y) in enumerate(loader):
        images = x.squeeze().to(device).float()
        labels = y
        labels = labels.to(device)
        preds = model(images)
        _, preds = torch.max(preds, 1)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )

        all_labels = torch.cat(
            (all_labels, labels),
            dim=0
        )

    all_preds = all_preds.type(torch.int64)
    all_labels = all_labels.type(torch.int64)
    return all_preds, all_labels

def get_metrics(model, loader, device,num_classes):
    with torch.no_grad():
        model.eval()
        #prediction_set = DataLoader(data_loader_tbc_validation,
        #                        batch_size=config.batch_size,
        #                        shuffle=False,
        #                        drop_last=False)
        validation_set_preds, targets = get_all_preds(model,
                                                      loader,
                                                      device)

        print("inference done", flush=True)
        ground_truth = targets.detach().cpu().numpy()
        predicted = validation_set_preds.detach().cpu().numpy()

        cmt_sklearn_normalize = metrics.confusion_matrix(ground_truth, predicted, normalize="true")
        cmt_sklearn_normalize = np.around(cmt_sklearn_normalize, decimals=5, out=None)
        cmt_sklearn_normalize = cmt_sklearn_normalize * 100
        print("confusion matrix calculated", flush=True)

        df_cm = pd.DataFrame(cmt_sklearn_normalize,
                             index=range(num_classes),
                             columns=range(num_classes))
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title('Confusion matrix (%)')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        wandb.log({"Confusion matrix": wandb.Image(plt)})

        all_metrics = metrics.classification_report(ground_truth, predicted, output_dict=True)
        wandb.log(all_metrics)