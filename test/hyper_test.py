import torch
from torch import nn
from pathlib import Path
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from config import *
import numpy as np
from trainer.utils import EarlyStopping

def main():
    cfg = OmegaConf.load(Path(config_path).parent.parent.as_posix() + '/train_configurations/test_cnn3d.yaml')
    cfg.dataset.coords_path = "/davinci-1/home/morellir/artificial_intelligence/repos/fdir/data/hyper/albania/mosaic/train_val_dict_merged_12k_psize15.pkl"
    cfg.dataset.test_coords_path = "/davinci-1/home/morellir/artificial_intelligence/repos/fdir/data/hyper/albania/mosaic/train_val_dict_merged_12k_psize15.pkl"
    trainloader, valloader, weights, metrics = get_dataset(cfg, batch_size=cfg.opt.batch_size,patch_size=cfg.dataset.patch_size,
                                                                          from_dictionary=cfg.dataset.from_dictionary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg, num_filter=cfg.model.num_filter, act=cfg.model.act,
                            filter_size=cfg.model.filter_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8,
                                                                patience=cfg.opt.lr_patience, threshold=0.0001, threshold_mode='rel',
                                                                cooldown=0, min_lr=9e-8, verbose=True)
    if cfg.model.class_number > 1:
        if cfg.opt.unbalanced_resampling:
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    else:
        if cfg.opt.unbalanced_resampling:
            criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)

    early_stopping = EarlyStopping(patience=cfg.opt.early_patience)

    best_val_loss = 10**6
    for epoch in range(cfg.opt.epochs):
        current_epoch = epoch
        ####Train Loop####
        """
        Set the models to the training mode first and train
        """
        train_loss = []
        # each step should be last all the epoch times if you don't return after validation step
        model.train()
        temp_train_loss = 0.0
        train_steps = 0
        for i, (x, y) in enumerate(trainloader):

            x = x.float().to(device)
            y = y.float().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y_hat = model(x).squeeze(-1)
            w = int(y.shape[1] / 2)
            h = int(y.shape[2] / 2)
            central_pixel = y[:, w, h].type(torch.LongTensor).to(device)

            # Compute and print loss
            loss = criterion(y_hat, central_pixel.float())
            loss.backward()
            optimizer.step()

            temp_train_loss += loss
            train_steps += 1
            if i % 10 == 0:
                print(" training Loss: {}".format(loss.item()))

        train_loss = temp_train_loss / train_steps
        train_loss_cpu = train_loss.cpu().item()
        print(" training Loss at the end of epoch {} is : {}".format(current_epoch,train_loss_cpu))
        ###############################################
        # eval mode for evaluation on validation dataset
        ###############################################
        # Validation loss
        temp_val_loss = 0.0
        val_steps = 0
        y_hat_tensor = torch.zeros_like(y_hat)
        central_pixel_tensor = torch.zeros_like(central_pixel)
        model.eval()
        for i, (x, y) in enumerate(valloader, 0):
            with torch.no_grad():
                x = x.float().to(device)
                y = y.float().to(device)

                # forward + backward + optimize
                y_hat = model(x).squeeze(-1)

                w = int(y.shape[1] / 2)
                h = int(y.shape[2] / 2)
                central_pixel = y[:, w, h].type(torch.LongTensor).to(device)

                if i == 0:
                    y_hat_tensor = torch.zeros_like(y_hat)
                    central_pixel_tensor = torch.zeros_like(central_pixel)
                else:
                    y_hat_tensor = torch.cat((y_hat_tensor.cpu(), y_hat.detach().cpu()), 0)
                    central_pixel_tensor = torch.cat((central_pixel_tensor.cpu(), central_pixel.detach().cpu()), 0)

                # Compute and print loss
                temp_val_loss += criterion(y_hat, central_pixel.float())

                y_hat_tensor = torch.cat((y_hat_tensor.cpu(), y_hat.detach().cpu()), 0)
                central_pixel_tensor = torch.cat((central_pixel_tensor.cpu(), central_pixel.detach().cpu()), 0)

                val_steps += 1

        val_loss = temp_val_loss / val_steps
        val_loss_cpu = val_loss.cpu().item()
        if val_loss_cpu < best_val_loss:
            print('val loss improved from {} to {}'.format(best_val_loss, val_loss_cpu))
            best_val_loss = val_loss_cpu
            print("this is the checkpoint dir {}".format(cfg.model.checkpoint))
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss_cpu,
                'cfg': cfg
            }, f"{cfg.model.checkpoint}model.pt")

        try:
            f1_score = model.f1_score(y_hat_tensor.to(device), central_pixel_tensor.to(device)).cpu().item()
            acc = model.acc(y_hat_tensor.to(device), central_pixel_tensor.to(device)).cpu().item()
            print("val loss {} val acc: {} and f1_score {}".format(val_loss_cpu, acc, f1_score))
        except:
            print("val loss {} ".format(val_loss_cpu, acc, f1_score))

        scheduler.step(val_loss_cpu)
        early_stopping(temp_val_loss)
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    main()