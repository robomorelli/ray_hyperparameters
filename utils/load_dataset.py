from preprocessing.nls_kdd_preprocessing import prep_nls_kdd_test, prep_nls_kdd_train_val
from preprocessing.albania_preprocessing import prep_albania
from dataset.nls_kdd_dataloader import numpyArray
from dataset.albania_dataloader import Supervised

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import pickle
import numpy as np
import multiprocessing as mp

from config import *

def get_dataset(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    if cfg.dataset.name == "nls_kdd":
        # Preprocessing step (there is also testx to use for example in test accuray in trainer step
        trainx, valx, ohe = prep_nls_kdd_train_val(0.2, nls_kdd_cols, nls_kdd_cat_cols,
                                          preprocessing='log', exclude_cat=True)
        # Dataset for dataloader definition
        dataset_train = numpyArray(trainx)
        dataset_val = numpyArray(valx)
        # Dataloader definition
        trainloader = DataLoader(dataset_train, batch_size=kwargs['batch_size'], shuffle=True)
        valloader = DataLoader(dataset_val, batch_size=kwargs['batch_size'], shuffle=True)

        return trainloader, valloader, len(nls_kdd_cols), len(nls_kdd_cat_cols), trainx.shape[1]

    if cfg.dataset.name == "albania_supervised":
        # Preprocessing step (load mixed coords of negative and positive)
        open_file = open(cfg.dataset.coords_path, "rb")
        selected_pixels = pickle.load(open_file)
        open_file.close()

        metrics = ["accuracy", "f1_score"] # evaluate to pass from here the entire metric object to the trainer

        # use case: kwarg cames after loading data a preprocessing step (the can also come from object trainer (cnn3d trainer for example)
        #Kwargs information:
        c_train, l_train, path_train, c_val, l_val, path_val = prep_albania(selected_pixels, dataset_train_split=cfg.dataset.train_split)
        c_test, l_test, path_test = prep_albania(selected_pixels, test=True)

        transform = T.Compose([
                T.ToTensor(),
            ])

        dataset_train = Supervised( n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=True,
                                            # From Kwargs:
                                           patch_size= kwargs['patch_size'], batch_size = kwargs['batch_size'],
                                           transform=transform, samples_coords_train=c_train,
                                           labels_train=l_train, patch_path_train=path_train, samples_coords_val=c_val,
                                           labels_val=l_val, patch_path_val=path_val)
        dataset_val = Supervised(n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=False,
                                            # From Kwargs:
                                           patch_size= kwargs['patch_size'], batch_size = kwargs['batch_size'],
                                           transform=transform, samples_coords_train=c_train,
                                           labels_train=l_train, patch_path_train=path_train, samples_coords_val=c_val,
                                           labels_val=l_val,  patch_path_val=path_val)
        dataset_test = Supervised(#patch_size=cfg.dataset.patch_size
                                            n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=False,
                                            test=True,
                                            # From Kwargs:
                                            patch_size=kwargs['patch_size'], batch_size = kwargs['batch_size'],
                                            transform=transform, samples_coords_test=c_test,
                                            labels_test=l_test, patch_path_test=path_test)

        #if cfg.opt.num_workers is None:
        #    num_workers = mp.cpu_count()
        #else:
        #    num_workers = cfg.opt.num_workers

        train_loader = DataLoader(dataset_train, batch_size=kwargs['batch_size'],  shuffle=True) #num_workers=num_workers,
        val_loader = DataLoader(dataset_val,  batch_size=kwargs['batch_size'], shuffle=False) #num_workers=num_workers,
        test_loader = DataLoader(dataset_test,  batch_size=kwargs['batch_size'], shuffle=False) #num_workers=num_workers,

        weights = get_class_weights(dataset_train)

        return train_loader, val_loader, test_loader, weights, metrics




def get_class_weights(dataset):
    if dataset.class_number > 0:
        class_frequency = class_frequency_center_pixel(list(dataset), dataset.class_number)
        weights = [1 / (x / np.sum(class_frequency)) for x in class_frequency]
        weights = torch.FloatTensor(weights / np.max(weights))
        if dataset.class_number == 1:
            pos_weight = weights[1]/weights[0]
            print(f"Class frequency: {class_frequency}")
            print(f"Class weights: {weights}")
            print(f"reweight: {pos_weight}")
            return pos_weight
        print(f"Class frequency: {class_frequency}")
        print(f"Class weights: {weights}")
        return weights
    else:
        return None

def class_frequency_center_pixel(patch_list, n_classes) -> np.array:
    """
    Compute the frequency of the central pixel
    :param patch_list: list of patches
    :return: array of the same length of the classes with their relative frequency
    """
    print("Computing class frequency...")
    target_list = []
    for patch in patch_list:
        target = patch[1]
        if target.shape[0] > 1:
            w = int(target.shape[0] / 2)
            h = int(target.shape[1] / 2)
            central_pixel = target[w, h]
        else:
            central_pixel = target[0]
        target_list.append(int(central_pixel))

    if n_classes > 1:
        target_arr = np.ones(n_classes)
    elif n_classes == 1:
        target_arr = np.ones(n_classes+1)

    for t in target_list:
        target_arr[t] += 1

    return target_arr
