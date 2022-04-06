import torch
import os

import wandb


def train(train_loader, model, criterion, optimizer, nr, device):
    loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.squeeze().float()
        x_j = x_j.squeeze().float()

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
    out = f"{name}_ckpt_{current_epoch}_{run_name}_.pt"
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

    ckpt_folder = f'./ckpt_all/ckpt__{wandb_id}/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    return ckpt_folder
