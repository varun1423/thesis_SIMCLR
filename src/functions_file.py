import torch
import os
from PIL import Image, ImageFilter
import random
import wandb
import numpy
from sklearn import metrics
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms


def train_simCLR(train_loader, model, criterion, optimizer, device):
    """Runs a training loop for SimCLR model.

        Args:
            train_loader (Dataloader): Dataloader.
            model (model): a Pytorch model.
            criterion (loss): a loss criterion
            optimizer (optimizer): an optimizer.
            device (gpu/cpu): based on availablity of the system, gpu recommended

        Returns:
            torch.Tensor: loss per iteration/epoch.
    """
    loss_epoch = 0  # initialized
    for step, (x_i, x_j) in enumerate(train_loader):
        print(x_i.size(), x_j.size())

        optimizer.zero_grad()

        # positive pair
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        z_i = model(x_i)
        z_j = model(x_j)

        # loss based on batch size
        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        wandb.log({"train_step_loss": loss.item()})
        loss_epoch += loss.item()
    return loss_epoch


def save_model(model, optimizer, scheduler, current_epoch, name, run_name):
    """ saves a check-point

        Args:
            model (model): a Pytorch model.
            scheduler : lr rate scheduler if any
            optimizer (optimizer): an optimizer.
            current_epoch (int): an int value of current epoch
            name (str) : folder name
            run_name (str) : unique name for saved ckpt file

        Returns:
            None.
    """
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
    """returns optimizer

        Args:
            downstream_model (model): a Pytorch model.
            arg_optimizer (optimizer): an optimizer.
            sgd_adam_lr (float): learning rate

        Returns:
            optimzer
    """

    if arg_optimizer == "Adam":
        optimizer = torch.optim.Adam([params for params in downstream_model.parameters() if params.requires_grad], lr=sgd_adam_lr)  # TODO: LARS
    elif arg_optimizer == "SGD":
        optimizer = torch.optim.SGD([params for params in downstream_model.parameters() if params.requires_grad], lr=sgd_adam_lr)
    else:
        raise NotImplementedError

    return optimizer


def training_ds(model, data_loader, optimizer, criterion, device):
    """Runs a training loop for downstream task ex: classification.

        Args:
           model (model): a Pytorch model.
            data_loader (Dataloader): Dataloader.
            optimizer (optimizer): an optimizer.
            criterion (loss): a loss criterion
            device (gpu/cpu): based on availablity of the system, gpu recommended

        Returns:
            loss_epoch (torch.Tensor) : loss per epoch.
            correct (int) : number of coorectly classified images per epoch
    """
    loss_epoch = 0
    correct = 0
    for step, (x, y) in enumerate(data_loader):
        x = x.squeeze().to(device).float()
        label = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # arg max index (pred class label)
        _, pred = torch.max(output, 1)
        correct_per_batch = (pred == label).float() # per step

        # wandb log
        wandb.log({"train_step_loss": loss.item()})
        wandb.log({"train_accuracy_step": 100*(correct_per_batch.sum()/len(correct_per_batch))})

        if step % 50 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t Loss: {round(loss.item(), 5)}\t accuracy : {100*(correct_per_batch.sum()/len(correct_per_batch))}")

        loss_epoch += loss.item()  # aggregate for epoch
        correct += correct_per_batch.sum()  # aggregate for epoch

    return loss_epoch, correct


def validation_ds(model, data_loader, criterion, device):
    """Runs a validation loop for downstream task ex: classification.

        Args:
           model (model): a Pytorch model.
            data_loader (Dataloader): Dataloader.
            criterion (loss): a loss criterion
            device (gpu/cpu): based on availablity of the system, gpu recommended

        Returns:
            loss_epoch (torch.Tensor) : loss per epoch.
            correct (int) : number of coorectly classified images per epoch
    """
    model.eval()  # set to eval() 
    loss_epoch = 0
    correct = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader):
            x = x.squeeze().to(device).float()
            label = y.to(device)
            output = model(x)
            loss = criterion(output, label)
            _, pred = torch.max(output, 1)
            correct_per_batch = (pred == label).float() # per step
            wandb.log({"val_step_loss": loss.item(), })
            wandb.log({"val_accuracy_step": 100 * (correct_per_batch.sum() / len(correct_per_batch)), })
            if step % 50 == 0:
                print(
                    f"Step [{step}/{len(data_loader)}]\t val_Loss: {round(loss.item(), 5)}\t val_accuracy : {100 * (correct_per_batch.sum() / len(correct_per_batch))}")

            loss_epoch += loss.item() # aggregate for epoch
            correct += correct_per_batch.sum() # aggregate for epoch
        return loss_epoch, correct


def ckpt_folder(ckpt_all, wandb_id):
    if not os.path.exists(f"./{ckpt_all}"):
        os.makedirs(ckpt_all)
    if not os.path.exists(f"{ckpt_all}/ckpt__{wandb_id}/"):
        os.makedirs(f"{ckpt_all}/ckpt__{wandb_id}/")
    ckpt_folder = f'./{ckpt_all}/ckpt__{wandb_id}/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    return ckpt_folder


def positive_pair(image, crop_size, overlap, resolution_x, resolution_y):
    """created a positive pair from high res single image.

        Args:
            image (PIL Image or ndarray): Image in form of PIL image or array.
            crop_size (int): size of the output image which will be used by model to learn.
            resolution_x (int): horizontal Image size
            resolution_y (int): vertical Image size

        Returns:
            main_Crop_image (PIL.Image) : first crop.
            second_image (PIL.Image) : second crop
    """
    # check and convert image to PIL Image.
    if type(image) == numpy.ndarray:
        inp_img = Image.fromarray(image)
    else:
        inp_img = Image.open(image)
    end_limit_x = (resolution_x - crop_size)
    end_limit_y = (resolution_y - crop_size)

    # generate a random x, y to  make the first crop
    x = random.randint(0, end_limit_x)
    y = random.randint(0, end_limit_y)

    # checking if Image cropped randomly sampled points lies on the Image and resamples if otherwise 
    if (x + crop_size > 2048) or (y + crop_size > 2048):
        resample = True
    else:
        resample = False
    while resample:
        x = random.randint(0, end_limit_x)
        y = random.randint(0, end_limit_y)
        if (x + crop_size < 2048) and (y + crop_size < 2048):
            resample = False

    # makes the first crop
    main_Crop = ((x, y, x + crop_size, y + crop_size))
    main_Crop_image = inp_img.crop(main_Crop)

    # overlap = True check conditions and return feasible space for second crop
    if overlap:
        overlapping_crop_start_list = []  # append all feasible options to the list

        if (x-(crop_size/2) > 0) and (y-(crop_size/2) > 0): # nrth west
            overlapping_crop_start_list.append((x-(crop_size/2), y-(crop_size/2)))
        if ((x+crop_size)+(crop_size/2) < resolution_x) and (y-(crop_size/2) > 0): # north east
            overlapping_crop_start_list.append((x+(crop_size/2), y-(crop_size/2)))
        if ((x+crop_size)+(crop_size/2) < resolution_x) and ((y+crop_size)+(crop_size/2) < resolution_y): # south east
            overlapping_crop_start_list.append((x+(crop_size/2),y+(crop_size/2)))
        if (x-(crop_size/2) > 0) and ((y+crop_size)+(crop_size/2) < resolution_y): # west south
            overlapping_crop_start_list.append((x-(crop_size/2), y+(crop_size/2)))

        co_ordinates = random.choice(overlapping_crop_start_list)  # select one feasible crop from given list
        second_overlap = ((co_ordinates[0], co_ordinates[1], co_ordinates[0] + crop_size, co_ordinates[1] + crop_size))
        second_image = inp_img.crop(second_overlap)  # make the second crop

        return main_Crop_image, second_image  # return a pair of crops
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


def positive_pair_50_overlap(image, crop_size, resolution_x, resolution_y):
    """created a positive pair with 50% overlap.

        Args:
            image (PIL Image or ndarray): Image in form of PIL image or array.
            crop_size (int): size of the output image which will be used by model to learn.
            resolution_x (int): horizontal Image size
            resolution_y (int): vertical Image size

        Returns:
            main_Crop_image (PIL.Image) : first crop.
            second_image (PIL.Image) : second crop
    """
    if type(image) == numpy.ndarray:
        inp_img = Image.fromarray(image)
    else:
        inp_img = Image.open(image)

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

    overlapping_crop_start_list = []
    if (x-(crop_size/2) > 0) and (y-(crop_size/2) > 0):
        overlapping_crop_start_list.append((x-(round(crop_size/3.4)), y-(round(crop_size/3.4))))
    if ((x+crop_size)+(crop_size/2) < resolution_x) and (y-(crop_size/2) > 0): # north east
        overlapping_crop_start_list.append((x+(round(crop_size/3.4)), y-(round(crop_size/3.4))))
    if ((x+crop_size)+(crop_size/2) < resolution_x) and ((y+crop_size)+(crop_size/2) < resolution_y): # south east
        overlapping_crop_start_list.append((x+(round(crop_size/3.4)),y+(round(crop_size/3.4))))
    if (x-(crop_size/2) > 0) and ((y+crop_size)+(crop_size/2) < resolution_y): # west south
        overlapping_crop_start_list.append((x-(round(crop_size/3.4)), y+(round(crop_size/3.4))))

    co_ordinates = random.choice(overlapping_crop_start_list)
    second_overlap = ((co_ordinates[0], co_ordinates[1], co_ordinates[0] + crop_size, co_ordinates[1] + crop_size))
    second_image = inp_img.crop(second_overlap)

    return main_Crop_image, second_image


def get_all_preds(model, loader, device):
    """created tensor of all prediction and label.

        Args:
            model :  pytorch model
            loader (data loader):
            device (gpu/cpu) :

        Returns:
            all_preds (torch.Tensor) : prediction of all validation Images.
            all_labels (torch.Tensor) : labels of all validation Images
    """
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    for step, (x, y) in enumerate(loader):
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


def get_metrics(model, loader, device, num_classes):
    """calculate metrics for downstream task and logs in wandb.

        Args:
            model :  pytorch model
            loader (data loader):
            device (gpu/cpu) :
            num_classes (int) : number of classes in dataset

        Returns:
                None
    """
    with torch.no_grad():
        model.eval()
        # get all predictions
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

        # to plot a confusion matrix
        df_cm = pd.DataFrame(cmt_sklearn_normalize,
                             index=range(num_classes),
                             columns=range(num_classes))

        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title('Confusion matrix (%)')
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        wandb.log({"Confusion matrix": wandb.Image(plt)})
        all_metrics = metrics.classification_report(ground_truth, predicted, output_dict=True)
        wandb.log(all_metrics)


def tbc_transform(channel,
                  rotation_prob,
                  blur_prob,
                  horizontal_flip_prob,
                  bach=False
                  ):
    """creates a transformation pipeline

        Args:
            channel (int)               : 1 means the model takes single channel input, 3 means - 3 channel input  
            rotation_prob (float)       : probability value between 0.0 to 1.0
            blur_prob (float)           : probability value between 0.0 to 1.0
            horizontal_flip_prob (float): probability value between 0.0 to 1.0
            bach (bool)                 : bool value if working with BACH dataset

        Returns:
                None
    """

    if channel == 1:
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
             transforms.RandomApply([GaussianBlur()], p=blur_prob),
             transforms.RandomApply([transforms.RandomRotation(30)], p=rotation_prob),
             transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
             ]
        )
    elif channel == 3:
        if bach:
            transform = transforms.Compose(
                [transforms.RandomApply([GaussianBlur()], p=blur_prob),
                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                 transforms.RandomApply([transforms.RandomRotation(random.randint(0, 180))], p=rotation_prob),
                 transforms.ToTensor(),
                 ]
            )
        else:
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=3),
                 transforms.RandomApply([GaussianBlur()], p=blur_prob),
                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                 transforms.RandomApply([transforms.RandomRotation(random.randint(0, 180))], p=rotation_prob),
                 transforms.ToTensor(),
                 ]
            )
    else:
        raise NotImplementedError
    return transform


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
