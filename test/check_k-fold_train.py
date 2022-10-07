import torch
from torch import nn
from pathlib import Path
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from config import *
import numpy as np

def main():
    cfg = OmegaConf.load(Path(config_path).parent.parent.as_posix() + '/train_configurations/test.yaml' )
    cfg.dataset.coords_path = "/home/roberto/Documents/backup_rob/esa/fdir/data/hyper/albania/mosaic/merged_3_patches.pkl"
    cfg.dataset.test_coords_path = "/home/roberto/Documents/backup_rob/esa/fdir/data/hyper/albania/mosaic/merged_3_patches.pkl"
    trainloader_list, valloader_list, weights_list, metrics = get_dataset(cfg, batch_size=cfg.opt.batch_size,patch_size=cfg.dataset.patch_size,
                                                                          from_dictionary=cfg.dataset.from_dictionary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = [get_model(cfg, num_filter=cfg.model.num_filter, act=cfg.model.act,
                            filter_size=cfg.model.filter_size).to(device) for i in range(cfg.opt.k_fold_cv)]
    optimizer_list = [torch.optim.Adam(model.parameters(), lr=cfg.opt.lr) for model in model_list]
    #scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_list[i], 'min', factor=0.8,
    #                                                            patience=cfg.opt.lr_patience, threshold=0.0001, threshold_mode='rel',
    #                                                            cooldown=0, min_lr=9e-8, verbose=True) for i in range(len(model_list))]
    if cfg.model.class_number > 1:
        criterion_list = [nn.CrossEntropyLoss(weight=weights_list[i]).to(device) for i in
                               range(len(model_list))]
    else:
        criterion_list = [nn.BCEWithLogitsLoss(pos_weight=weights_list[i]).to(device) for i in
                               range(len(model_list))]

    from_dictionary = cfg.dataset.from_dictionary
    for epoch in range(cfg.opt.epochs):
        # model.apply(reset_weights)
        fold_overall_train_loss = []
        fold_overall_val_loss = []
        fold_overall_val_f1 = []
        fold_overall_val_acc = []
        for fold, (train_dl, val_dl) in enumerate(zip(trainloader_list, valloader_list)):
            temp_train_loss = 0.0
            train_steps = 0
            model_list[fold].train()
            for i, (x, y) in enumerate(train_dl):

                x = x.float().to(device)
                y = y.float().to(device)

                # zero the parameter gradients
                optimizer_list[fold].zero_grad()

                # forward + backward + optimize
                model = model_list[fold]
                y_hat = model(x).squeeze(-1)
                w = int(y.shape[1] / 2)
                h = int(y.shape[2] / 2)
                central_pixel = y[:, w, h].type(torch.LongTensor).to(device)
                print("shapeeeeeeeeee", y_hat.shape, central_pixel.shape)

                # Compute and print loss
                loss = criterion_list[fold](y_hat, central_pixel.float())
                loss.backward()
                optimizer = optimizer_list[fold]
                optimizer.step()

                temp_train_loss += loss
                train_steps += 1
                if i % 10 == 0:
                    print(" training Loss: {} on fold {}".format(loss.item(), fold))

            train_loss = temp_train_loss / train_steps
            train_loss_cpu = train_loss.cpu().item()
            fold_overall_train_loss.append(train_loss_cpu)  # buffer for each fold tr
            ###############################################
            # eval mode for evaluation on validation dataset
            ###############################################
            # Validation loss
            temp_val_loss = 0.0
            val_steps = 0
            y_hat_tensor = torch.zeros_like(y_hat)
            central_pixel_tensor = torch.zeros_like(central_pixel)
            model.eval()
            for i, (x, y) in enumerate(val_dl, 0):
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
                    temp_val_loss += criterion_list[fold](y_hat, central_pixel.float())

                    y_hat_tensor = torch.cat((y_hat_tensor.cpu(), y_hat.detach().cpu()), 0)
                    central_pixel_tensor = torch.cat((central_pixel_tensor.cpu(), central_pixel.detach().cpu()), 0)
                    val_steps += 1

            val_loss = temp_val_loss / val_steps
            val_loss_cpu = val_loss.cpu().item()
            fold_overall_val_loss.append(val_loss_cpu)
            print(" val Loss: {} on fold {}".format(val_loss_cpu, fold))

        print(" averaging the train loss: {} ".format(np.mean(fold_overall_train_loss)))
        print(" averaging the val loss: {} ".format(np.mean(fold_overall_val_loss)))

if __name__ == "__main__":
    main()