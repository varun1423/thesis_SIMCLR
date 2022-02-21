import torch
import os

import wandb


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


def valid(valid_loader, model, criterion, nr, device):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(valid_loader):

        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)

        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(), 5)}")

        loss_epoch += loss.item()
    return loss_epoch

def save_model(model, optimizer, scheduler, current_epoch, name, run_name):
    out = f"{name}_{current_epoch}_{run_name}_.pt"
    #out = os.path.join('./',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)
